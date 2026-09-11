import asyncio
import getpass
import json
import os
from pathlib import Path
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

from nersc_ollama_manager.core import Manager, atomic_json, confirm
from nersc_ollama_manager.cli import parser


class ManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.m = Manager.initialize(root / 'config.json', root / 'runtime', root / 'ollama')
        self.record = dict(schema_version=1, id='test-123-abcdef', name='test', uid=os.getuid(),
                           job_id='123', host='nid000001', port=23456, state='ready', heartbeat=time.time())
        self.job = dict(id='123', user=getpass.getuser(), state='RUNNING', nodes='nid000001', remaining='29:00')

    def test_cpu_and_gpu_commands(self):
        cpu = self.m.allocation_command('cpu', 'cpu')
        self.assertNotIn('--gpus', cpu)
        self.assertIn('srun', cpu)
        with self.assertRaises(ValueError):
            self.m.allocation_command('gpu', 'gpu')
        gpu = self.m.allocation_command('gpu', 'gpu', 'example_g')
        self.assertEqual(gpu.count('--gpus-per-node'), 1)
        self.assertEqual(gpu.count('--gpus'), 1)
        self.assertNotIn('--gpu-bind', gpu)
        self.assertIn('example_g', gpu)
        self.assertIn('00:30:00', gpu)

    def test_gpu_spread_disables_step_gpu_binding(self):
        self.m.set_gpu_spread(True)
        gpu = self.m.allocation_command('gpu', 'gpu', 'example_g')
        self.assertIn('--gpu-bind', gpu)
        self.assertEqual(gpu[gpu.index('--gpu-bind') + 1], 'none')

    def test_no_implicit_scheduler_approval(self):
        with patch('os.isatty', return_value=False), self.assertRaises(RuntimeError):
            confirm('Allocate?')

    def test_validate_stale_and_wrong_node(self):
        with patch.object(self.m, 'job', return_value=self.job):
            self.m.validate_record(self.record)
            with self.assertRaises(RuntimeError):
                self.m.validate_record({**self.record, 'heartbeat': time.time() - 100})
            with self.assertRaises(RuntimeError):
                self.m.validate_record({**self.record, 'host': 'nid000002'})
        with patch.object(self.m, 'job', return_value=None), self.assertRaises(RuntimeError):
            self.m.validate_record(self.record)

    def test_malformed_and_expired_discovery(self):
        (self.m.records / 'bad.json').write_text('{')
        atomic_json(self.m.records / 'old.json', self.record)
        with patch.object(self.m, 'job', return_value=None):
            records = self.m.list_servers()
        self.assertEqual(len(records), 2)
        self.assertFalse(any(r['available'] for r in records))

    def test_atomic_private_records(self):
        path = self.m.records / 'test.json'
        atomic_json(path, self.record)
        self.assertEqual(path.stat().st_mode & 0o777, 0o600)
        atomic_json(path, {**self.record, 'state': 'stopped'})
        self.assertEqual(json.loads(path.read_text())['state'], 'stopped')

    def test_ambiguous_selection(self):
        with patch.object(self.m, 'list_servers', return_value=[{**self.record, 'available': True},
                  {**self.record, 'id': 'second', 'available': True}]):
            with self.assertRaises(RuntimeError):
                self.m.select()
            self.assertEqual(self.m.select('second')['id'], 'second')

    def test_tunnel_reuses_only_healthy_endpoint(self):
        control, meta = Path(self.tmp.name) / 'sock', Path(self.tmp.name) / 'meta.json'
        atomic_json(meta, dict(server_id=self.record['id'], remote_port=23456, port=34567))
        with patch.object(self.m, 'validate_record'), patch.object(self.m, 'tunnel_paths', return_value=(control, meta)), \
             patch('nersc_ollama_manager.core.subprocess.run', return_value=Mock(returncode=0)) as proc, \
             patch('nersc_ollama_manager.core.request', return_value={'version': 'test'}):
            self.assertEqual(self.m.ensure_tunnel(self.record), 34567)
            self.assertEqual(proc.call_count, 1)

    def test_dead_tunnel_recreated(self):
        control, meta = Path(self.tmp.name) / 'sock', Path(self.tmp.name) / 'meta.json'
        atomic_json(meta, dict(server_id=self.record['id'], remote_port=23456, port=34567))
        with patch.object(self.m, 'validate_record'), patch.object(self.m, 'tunnel_paths', return_value=(control, meta)), \
             patch('nersc_ollama_manager.core.subprocess.run', return_value=Mock(returncode=0)), \
             patch('nersc_ollama_manager.core.request', side_effect=[OSError('dead'), {'version': 'test'}]), \
             patch('nersc_ollama_manager.core.free_port', return_value=34568):
            self.assertEqual(self.m.ensure_tunnel(self.record), 34568)

    def test_codex_model_and_endpoint(self):
        with patch('nersc_ollama_manager.core.shutil.which', return_value='/mock/bin/codex'), \
             patch.object(self.m, 'ensure_tunnel', return_value=34567), \
             patch('nersc_ollama_manager.core.request', side_effect=[{'models': [{'name': 'test:latest'}]},
                                                                   {'capabilities': ['tools']} ]):
            command = self.m.codex_command(self.record, 'test', ['--no-alt-screen'])
            self.assertIn('model_providers.nersc_ollama.base_url="http://127.0.0.1:34567/v1"', command)
            self.assertEqual(command[-1], '--no-alt-screen')
        with self.assertRaises(ValueError):
            self.m.check_model('test:cloud')

    def test_missing_model(self):
        with patch('nersc_ollama_manager.core.shutil.which', return_value='/mock/bin/codex'), \
             patch.object(self.m, 'ensure_tunnel', return_value=34567), \
             patch('nersc_ollama_manager.core.request', return_value={'models': []}), self.assertRaisesRegex(RuntimeError, 'not installed'):
            self.m.codex_command(self.record, 'missing')

    def test_login_node_rejected(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(self.m, 'job', return_value=self.job), \
             patch('socket.gethostname', return_value='login01'), self.assertRaises(RuntimeError):
            self.m.serve('test', 'cpu')

    def test_server_timeout_stops_owned_process(self):
        proc = Mock()
        proc.poll.return_value = None
        proc.__enter__ = Mock(return_value=proc)
        proc.__exit__ = Mock(return_value=False)
        proc.pid = 12345
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(self.m, 'job', return_value=self.job), \
             patch('socket.gethostname', return_value='nid000001'), \
             patch('nersc_ollama_manager.core.free_port', return_value=23456), \
             patch('nersc_ollama_manager.core.subprocess.Popen', return_value=proc), \
             patch('nersc_ollama_manager.core.request', side_effect=OSError('not ready')), \
             patch('nersc_ollama_manager.core.time.monotonic', side_effect=[0, 301]), \
             self.assertRaisesRegex(RuntimeError, 'timed out'):
            self.m.serve('test', 'cpu')
        proc.terminate.assert_called_once()
        records = list(self.m.records.glob('*.json'))
        self.assertEqual(json.loads(records[0].read_text())['state'], 'stopped')

    def test_disconnect_expired_record(self):
        atomic_json(self.m.records / 'expired.json', self.record)
        self.assertEqual(self.m.select_for_disconnect(self.record['id'])['id'], self.record['id'])

    def test_missing_gpu_visibility(self):
        with patch.dict(os.environ, {'SLURM_JOB_ID': '123', 'CUDA_VISIBLE_DEVICES': ''}), \
             patch.object(self.m, 'job', return_value=self.job), \
             patch('socket.gethostname', return_value='nid000001'), self.assertRaisesRegex(RuntimeError, 'GPU visibility'):
            self.m.serve('test', 'gpu')

    def test_passthrough(self):
        args = parser().parse_args(['codex', '--model', 'test', '--', '--no-alt-screen'])
        self.assertEqual(args.codex_args, ['--', '--no-alt-screen'])


class DashboardTests(unittest.IsolatedAsyncioTestCase):
    async def test_dashboard_models_and_tunnel(self):
        from nersc_ollama_manager.tui import Dashboard
        from textual.widgets import DataTable, Select
        manager = Mock()
        manager.config = {'profiles': {'cpu': {'time': '00:30:00'}, 'gpu': {'time': '00:30:00'}}}
        manager.runtime = Path('/example/runtime')
        manager.root = Path('/example/ollama')
        manager.binary.is_file.return_value = True
        record = dict(id='test', name='test', available=True)
        manager.list_servers.return_value = [record]
        manager.select.return_value = record
        manager.downloaded_models.return_value = [{'name': 'test:latest'}]
        manager.ensure_tunnel.return_value = 34567
        app = Dashboard(manager)
        async with app.run_test(size=(120, 40)) as pilot:
            await pilot.pause()
            self.assertEqual(app.query_one(DataTable).row_count, 1)
            app.selected = 'test'
            await pilot.click('#load_models')
            await pilot.pause()
            self.assertTrue(manager.downloaded_models.called)
            await pilot.click('#tunnel')
            await pilot.pause()
            manager.ensure_tunnel.assert_called_once_with(record)


if __name__ == '__main__':
    unittest.main()

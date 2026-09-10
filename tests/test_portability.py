import contextlib
import io
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import Mock, patch

from nersc_ollama_manager.core import Manager, config_location, atomic_json
from nersc_ollama_manager.cli import main


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.env = patch.dict(os.environ, {'XDG_CONFIG_HOME': str(self.root/'config'),
                'XDG_DATA_HOME': str(self.root/'data'), 'NERSC_OLLAMA_CONFIG': ''})
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_first_launch_private_and_no_external_commands(self):
        with patch('nersc_ollama_manager.core.run') as run:
            manager = Manager.open()
        run.assert_not_called()
        self.assertEqual(manager.config_path, self.root/'config/nersc-ollama/config.json')
        self.assertEqual(manager.config_path.stat().st_mode & 0o777, 0o600)
        self.assertEqual(manager.runtime, self.root/'data/nersc-ollama')
        self.assertNotIn('python', manager.config)
        self.assertNotIn('henv', manager.config)
        self.assertIsNone(manager.config['profiles']['gpu']['account'])

    def test_precedence_and_missing_explicit(self):
        with patch.dict(os.environ, {'NERSC_OLLAMA_CONFIG': str(self.root/'env.json')}):
            self.assertEqual(config_location()[0], self.root/'env.json')
            self.assertEqual(config_location(self.root/'cli.json')[0], self.root/'cli.json')
            with self.assertRaisesRegex(RuntimeError, 'setup'):
                Manager.open()
            self.assertFalse((self.root/'env.json').exists())

    def test_home_fallback_relative_xdg(self):
        with patch.dict(os.environ, {'XDG_CONFIG_HOME': 'relative', 'XDG_DATA_HOME': ''}), patch('pathlib.Path.home', return_value=self.root):
            manager = Manager.open()
        self.assertEqual(manager.config_path, self.root/'.config/nersc-ollama/config.json')
        self.assertEqual(manager.runtime, self.root/'.local/share/nersc-ollama')

    def test_concurrent_initialization(self):
        with ThreadPoolExecutor(max_workers=4) as pool:
            configs = list(pool.map(lambda _: Manager.open().config, range(12)))
        self.assertTrue(all(c == configs[0] for c in configs))
        self.assertEqual(len(list((self.root/'config/nersc-ollama').iterdir())), 1)

    def test_invalid_config_is_not_replaced(self):
        manager = Manager.open()
        for contents in ('{', '[]', '{"schema_version": 200}'):
            manager.config_path.write_text(contents)
            with self.assertRaises(ValueError):
                Manager.open()
            self.assertEqual(manager.config_path.read_text(), contents)

    def test_help_and_version_do_not_initialize(self):
        for command in ('--help', '--version'):
            with contextlib.redirect_stdout(io.StringIO()), self.assertRaises(SystemExit) as exc:
                main([command])
            self.assertEqual(exc.exception.code, 0)
        self.assertFalse((self.root/'config').exists())

    def test_legacy_interpreter_ignored_without_rewrite(self):
        manager = Manager.open()
        atomic_json(manager.config_path, {**manager.config, 'python': '/old/environment/python', 'henv': True})
        before = manager.config_path.read_bytes()
        with patch.object(sys, 'executable', '/new/environment/python'):
            manager = Manager.open()
            command = manager.allocation_command('test', 'cpu')
        self.assertIn('/new/environment/python', command)
        self.assertNotIn('/old/environment/python', command)
        self.assertNotIn('henv', command)
        self.assertEqual(manager.config_path.read_bytes(), before)

    def test_codex_uses_path_and_preserves_arguments(self):
        manager = Manager.open()
        with patch('shutil.which', return_value='/chosen/bin/codex'), patch.object(manager, 'ensure_tunnel', return_value=23456), \
             patch('nersc_ollama_manager.core.request', side_effect=[{'models':[{'name':'test:latest'}]}, {'capabilities':['tools']}]):
            command = manager.codex_command({}, 'test', ['--no-alt-screen'])
        self.assertEqual(command[0], '/chosen/bin/codex')
        self.assertEqual(command[-1], '--no-alt-screen')
        with patch('shutil.which', return_value=None), self.assertRaisesRegex(RuntimeError, 'PATH'):
            manager.codex_command({}, 'test')

    def test_setup_explicit_and_storage_review(self):
        config = self.root/'custom.json'
        with contextlib.redirect_stdout(io.StringIO()):
            self.assertEqual(main(['--config', str(config), 'setup']), 0)
        manager = Manager(config)
        manager.set_storage(self.root/'runtime2', self.root/'models2')
        self.assertEqual(Manager(config).root, self.root/'models2')
        manager.binary.parent.mkdir(parents=True)
        manager.binary.touch()
        with self.assertRaises(RuntimeError):
            manager.set_storage(self.root/'runtime3', self.root/'models3')

    def test_installation_error_is_actionable(self):
        manager = Manager.open()
        with self.assertRaisesRegex(RuntimeError, 'setup --version VERSION'):
            manager.require_installation()


class OnboardingTests(unittest.IsolatedAsyncioTestCase):
    async def test_save_storage_without_download_or_job(self):
        from nersc_ollama_manager.tui import Dashboard
        from textual.widgets import Input
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manager = Manager.initialize(base/'config.json', base/'runtime', base/'ollama')
            app = Dashboard(manager)
            with patch.object(manager, 'install') as install, patch.object(manager, 'allocate') as allocate:
                async with app.run_test(size=(140, 50)) as pilot:
                    self.assertTrue(app.query_one('#onboarding').display)
                    app.query_one('#runtime_path', Input).value = str(base/'newruntime')
                    app.query_one('#root_path', Input).value = str(base/'newroot')
                    await pilot.click('#save_storage')
                    await pilot.pause()
                    self.assertEqual(manager.root, base/'newroot')
                install.assert_not_called()
                allocate.assert_not_called()

    async def test_explicit_install_button(self):
        from nersc_ollama_manager.tui import Dashboard
        from textual.widgets import Input
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manager = Manager.initialize(base/'config.json', base/'runtime', base/'ollama')
            app = Dashboard(manager)
            with patch.object(manager, 'install') as install, patch.object(app, 'suspend', side_effect=contextlib.nullcontext):
                async with app.run_test(size=(140, 50)) as pilot:
                    app.query_one('#release', Input).value = '0.34.0'
                    await pilot.click('#install')
                    await pilot.pause()
                    install.assert_called_once_with('0.34.0')
                    self.assertFalse(app.query_one('#onboarding').display)


class AllocationTimeTests(unittest.IsolatedAsyncioTestCase):
    async def test_profile_defaults_override_and_invalid_time(self):
        from nersc_ollama_manager.tui import Dashboard
        from textual.widgets import Input, Select
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manager = Manager.initialize(base/'config.json', base/'runtime', base/'ollama')
            manager.config['profiles']['gpu']['time'] = '00:45:00'
            app = Dashboard(manager)
            with patch.object(manager, 'allocate') as allocate, patch.object(app, 'suspend', side_effect=contextlib.nullcontext):
                async with app.run_test(size=(160, 50)) as pilot:
                    field = app.query_one('#walltime', Input)
                    self.assertEqual(field.value, '00:30:00')
                    app.query_one('#profile', Select).value = 'gpu'
                    await pilot.pause()
                    self.assertEqual(field.value, '00:45:00')
                    field.value = '01:00:00'
                    await pilot.click('#allocate')
                    await pilot.pause()
                    allocate.assert_called_once_with('default', 'gpu', account=None, walltime='01:00:00')
                    allocate.reset_mock()
                    for invalid in ('', '00:00:00', '01:99:00', 'tomorrow'):
                        field.value = invalid
                        await pilot.click('#allocate')
                        await pilot.pause()
                    allocate.assert_not_called()


class DownloadTests(unittest.TestCase):
    def test_download_requests_only_cpu_and_requires_confirmation(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manager = Manager.initialize(base/'config.json', base/'runtime', base/'ollama')
            with patch.dict(os.environ, {'SLURM_JOB_ID': ''}), patch.object(manager, 'require_installation'), \
                 patch('nersc_ollama_manager.core.confirm') as confirm, patch('nersc_ollama_manager.core.run') as run:
                manager.download('test:latest', walltime='01:00:00', backend='cpu')
                args = run.call_args.args[0]
                self.assertNotIn('--gpus', args)
                self.assertEqual(args[args.index('--constraint')+1], 'cpu')
                self.assertEqual(args[-2:], ['--download-model', 'test:latest'])
                confirm.assert_called_once()

    def test_worker_exits_and_stops_server_after_pull(self):
        import getpass
        from unittest.mock import MagicMock
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp)
            manager = Manager.initialize(base/'config.json', base/'runtime', base/'ollama')
            server, pull = MagicMock(), MagicMock()
            for proc in (server, pull):
                proc.__enter__.return_value = proc
            server.pid = 42
            server.poll.return_value = None
            pull.poll.return_value = 0
            pull.returncode = 0
            job = {'state': 'RUNNING', 'nodes': 'nid000001', 'user': getpass.getuser()}
            with patch.dict(os.environ, {'SLURM_JOB_ID': '123'}), patch.object(manager, 'job', return_value=job), \
                 patch('socket.gethostname', return_value='nid000001'), \
                 patch('nersc_ollama_manager.core.free_port', return_value=23456), \
                 patch('nersc_ollama_manager.core.request', return_value={}), \
                 patch('nersc_ollama_manager.core.subprocess.Popen', side_effect=[server,pull]) as spawn:
                manager.serve('download', 'cpu', 'test:latest')
            server.terminate.assert_called_once()
            self.assertEqual(spawn.call_args_list[1].args[0][-2:], ['pull', 'test:latest'])
            record=json.loads(next(manager.records.glob('*.json')).read_text())
            self.assertEqual(record['state'], 'stopped')

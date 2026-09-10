import json
from pathlib import Path
import shlex
import tempfile
import unittest
from unittest.mock import MagicMock, patch

from nersc_ollama_manager.core import Manager
from nersc_ollama_manager import dtn_worker


class DTNTests(unittest.TestCase):
    def test_command_uses_dtn_python_and_quotes_shared_paths(self):
        with tempfile.TemporaryDirectory(prefix='space path ') as tmp:
            root=Path(tmp)
            manager=Manager.initialize(root/'config.json',root/'runtime',root/'ollama')
            with patch.object(manager, 'require_installation'):
                command=manager.dtn_command('test:latest',check=True)
            self.assertEqual(command[0], 'ssh')
            self.assertIn('dtn01.nersc.gov', command)
            remote=shlex.split(command[-1])
            self.assertEqual(remote[0], 'python3')
            self.assertIn(str(manager.binary),remote)
            self.assertIn('--check',remote)
            self.assertNotIn('salloc',command)
            manager.config['downloads']={'dtn_host':'login01'}
            with patch.object(manager,'require_installation'),self.assertRaises(ValueError):
                manager.dtn_command('test')

    def test_dtn_never_submits_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            manager=Manager.initialize(root/'config.json',root/'runtime',root/'ollama')
            with patch.object(manager,'require_installation'),patch('nersc_ollama_manager.core.run') as run:
                manager.download('test:latest')
            self.assertEqual(run.call_args.args[0][0], 'ssh')

    def test_check_stops_server_without_pull(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            binary=root/'ollama'; binary.touch(); binary.chmod(0o700)
            proc=MagicMock();proc.poll.return_value=None
            response=MagicMock();response.__enter__.return_value=response
            response.read.return_value=json.dumps({'version':'test'}).encode()
            opener=MagicMock();opener.open.return_value=response
            with patch('socket.gethostname',return_value='dtn01.nersc.gov'), \
                 patch('socket.socket') as socket, \
                 patch('subprocess.Popen',return_value=proc) as spawn, \
                 patch('urllib.request.build_opener',return_value=opener):
                socket.return_value.__enter__.return_value.getsockname.return_value=('127.0.0.1',23456)
                result=dtn_worker.main(['--binary',str(binary),'--models',str(root/'models'),
                    '--logs',str(root/'logs'),'--model','test','--check'])
            self.assertEqual(result,0)
            self.assertEqual(spawn.call_count,1)
            proc.terminate.assert_called_once()
            env=spawn.call_args.kwargs['env']
            self.assertEqual(env['CUDA_VISIBLE_DEVICES'],'-1')
            self.assertTrue(env['TMPDIR'].startswith(str(root)))

    def test_rejects_login_node(self):
        with patch('socket.gethostname',return_value='login01'),self.assertRaises(RuntimeError):
            dtn_worker.main(['--binary','/example/ollama','--models','/example/models',
                '--logs','/example/logs','--model','test'])

"""nersc-ollama-ssh2server: pick a live server interactively, then SSH to it."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from nersc_ollama_manager.core import Manager
from nersc_ollama_manager.ssh_launcher import main, parser


class SshLauncherParsingTests(unittest.TestCase):
    def test_defaults(self):
        args = parser().parse_args([])
        self.assertIsNone(args.server)
        self.assertFalse(args.remote)

    def test_server_and_remote_flags(self):
        args = parser().parse_args(['--server', 'gpu', '--remote', '--remote-host', 'saul.nersc.gov'])
        self.assertEqual(args.server, 'gpu')
        self.assertTrue(args.remote)
        self.assertEqual(args.remote_host, 'saul.nersc.gov')


class SshLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.config = root / 'config.json'
        Manager.initialize(self.config, root / 'runtime', root / 'ollama')
        self.record = {'id': 'gpu-1', 'name': 'gpu', 'available': True, 'profile': 'gpu',
                        'host': 'nid000001', 'remaining': '10:00'}

    def test_execs_ssh_for_the_picked_server(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'ssh_shell_command', return_value=['ssh', '-tt', 'nid000001']) as ssh_shell_command, \
             patch('nersc_ollama_manager.ssh_launcher.os.execvp') as execvp:
            main(['--config', str(self.config), '--server', 'gpu'])
        ssh_shell_command.assert_called_once_with(self.record)
        execvp.assert_called_once_with('ssh', ['ssh', '-tt', 'nid000001'])

    def test_no_live_servers_is_a_clear_error(self):
        with patch.object(Manager, 'list_servers', return_value=[]), \
             patch('nersc_ollama_manager.ssh_launcher.os.execvp') as execvp:
            code = main(['--config', str(self.config)])
        self.assertEqual(code, 1)
        execvp.assert_not_called()

    def test_unknown_requested_server_is_a_clear_error(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch('nersc_ollama_manager.ssh_launcher.os.execvp') as execvp:
            code = main(['--config', str(self.config), '--server', 'nope'])
        self.assertEqual(code, 1)
        execvp.assert_not_called()

    def test_missing_ssh_binary_is_reported_not_raised(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'ssh_shell_command', return_value=['ssh', '-tt', 'nid000001']), \
             patch('nersc_ollama_manager.ssh_launcher.os.execvp', side_effect=FileNotFoundError('no ssh')):
            code = main(['--config', str(self.config), '--server', 'gpu'])
        self.assertEqual(code, 1)


if __name__ == '__main__':
    unittest.main()

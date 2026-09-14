"""nersc-ollama-codex: pick server/model/context interactively, then launch Codex."""
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch

from nersc_ollama_manager.core import Manager
from nersc_ollama_manager.codex_launcher import main, parser


class CodexLauncherParsingTests(unittest.TestCase):
    def test_defaults(self):
        args = parser().parse_args([])
        self.assertIsNone(args.server)
        self.assertIsNone(args.model)
        self.assertIsNone(args.context)
        self.assertEqual(args.codex_args, [])

    def test_overrides_and_passthrough(self):
        args = parser().parse_args(['--server', 'gpu', '--model', 'test:latest', '--context', '131072',
                                     '--', '--full-auto'])
        self.assertEqual(args.server, 'gpu')
        self.assertEqual(args.model, 'test:latest')
        self.assertEqual(args.context, 131072)
        self.assertEqual(args.codex_args, ['--', '--full-auto'])


class CodexLauncherTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.config = root / 'config.json'
        Manager.initialize(self.config, root / 'runtime', root / 'ollama')
        self.record = {'id': 'gpu-1', 'name': 'gpu', 'available': True, 'profile': 'gpu',
                        'host': 'nid000001', 'remaining': '10:00'}

    def test_fully_specified_skips_all_prompts(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'codex_command', return_value=['codex', '-m', 'test:latest']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0) as call, \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask') as ask:
            code = main(['--config', str(self.config), '--server', 'gpu', '--model', 'test:latest',
                         '--context', '131072', '--', '--full-auto'])
        self.assertEqual(code, 0)
        ask.assert_not_called()
        codex_command.assert_called_once_with(self.record, 'test:latest', ['--full-auto'], context=131072)
        call.assert_called_once_with(['codex', '-m', 'test:latest'])

    def test_single_server_and_model_auto_picked_without_prompt(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'codex_command', return_value=['codex']), \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0), \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask') as ask:
            code = main(['--config', str(self.config), '--context', '131072', '--', '--full-auto'])
        self.assertEqual(code, 0)
        ask.assert_not_called()  # only one server, only one model, extra already given

    def test_context_discovered_from_model_info_when_omitted(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'ensure_tunnel', return_value=34567), \
             patch.object(Manager, 'codex_command', return_value=['codex']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0), \
             patch('nersc_ollama_manager.codex_launcher.request',
                   return_value={'model_info': {'general.architecture': 'qwen3', 'qwen3.context_length': 262144}}), \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask') as ask:
            main(['--config', str(self.config), '--', '--full-auto'])
        codex_command.assert_called_once_with(self.record, 'test:latest', ['--full-auto'], context=262144)
        ask.assert_not_called()

    def test_no_live_servers_is_a_clear_error(self):
        with patch.object(Manager, 'list_servers', return_value=[]):
            code = main(['--config', str(self.config)])
        self.assertEqual(code, 1)

    def test_unknown_requested_server_is_a_clear_error(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]):
            code = main(['--config', str(self.config), '--server', 'nope'])
        self.assertEqual(code, 1)

    def test_unknown_requested_model_is_a_clear_error(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]):
            code = main(['--config', str(self.config), '--server', 'gpu', '--model', 'nope'])
        self.assertEqual(code, 1)

    def test_approval_preset_prompted_when_no_codex_args_given(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'codex_command', return_value=['codex']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0), \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask', return_value=2):  # "Default" preset
            main(['--config', str(self.config), '--context', '131072'])
        codex_command.assert_called_once_with(self.record, 'test:latest', [], context=131072)

    def test_multiple_servers_prompts_and_honors_choice(self):
        second = {**self.record, 'id': 'gpu-2', 'name': 'gpu2'}
        with patch.object(Manager, 'list_servers', return_value=[self.record, second]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'codex_command', return_value=['codex']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0), \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask', return_value=2):  # pick the 2nd server
            main(['--config', str(self.config), '--context', '131072', '--', '--full-auto'])
        codex_command.assert_called_once_with(second, 'test:latest', ['--full-auto'], context=131072)


if __name__ == '__main__':
    unittest.main()

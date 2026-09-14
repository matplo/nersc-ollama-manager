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

    def test_record_is_refreshed_before_the_final_codex_command_call(self):
        # record['heartbeat'] is a point-in-time snapshot from pick_server();
        # validate_record() rejects one over 90s old, which real interactive
        # navigation (model/context/approval-mode prompts) can plausibly
        # exceed even though the server itself is fine the whole time -- hit
        # live. Re-fetch via select() right before use, rather than reusing
        # the original snapshot.
        refreshed = {**self.record, 'heartbeat': 'fresh'}
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'select', return_value=refreshed) as select, \
             patch.object(Manager, 'codex_command', return_value=['codex']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0), \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask') as ask:
            main(['--config', str(self.config), '--server', 'gpu', '--model', 'test:latest',
                  '--context', '131072', '--', '--full-auto'])
        select.assert_called_once_with('gpu-1')
        codex_command.assert_called_once_with(refreshed, 'test:latest', ['--full-auto'], context=131072)
        ask.assert_not_called()

    def test_refresh_failure_falls_back_to_the_record_already_in_hand(self):
        # Hit live immediately after the refresh above was first added: the
        # refresh is itself one more remote round-trip that can transiently
        # fail, and that must never crash a session where the server was
        # otherwise perfectly fine -- fall back rather than propagate.
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'select', side_effect=RuntimeError('Select exactly one live server ...')), \
             patch.object(Manager, 'codex_command', return_value=['codex']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0) as call:
            code = main(['--config', str(self.config), '--server', 'gpu', '--model', 'test:latest',
                         '--context', '131072', '--', '--full-auto'])
        self.assertEqual(code, 0)
        codex_command.assert_called_once_with(self.record, 'test:latest', ['--full-auto'], context=131072)
        call.assert_called_once_with(['codex'])

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
             patch('nersc_ollama_manager._prompts.IntPrompt.ask', return_value=1):  # "Codex's own defaults" (first/the default)
            main(['--config', str(self.config), '--context', '131072'])
        codex_command.assert_called_once_with(self.record, 'test:latest', [], context=131072)

    def test_approval_configuration_approve_for_me_skips_manual_prompts(self):
        # --approve-for-me and --sandbox are confirmed mutually exclusive by
        # Codex's own parser (a real error hit live, not a guess) -- choosing
        # --approve-for-me must never also produce -s/-a, so the manual
        # prompts are skipped entirely rather than risking that combination.
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'codex_command', return_value=['codex']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0), \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask', return_value=2) as ask, \
             patch('nersc_ollama_manager.codex_launcher.Confirm.ask', return_value=True):
            main(['--config', str(self.config), '--context', '131072'])
        codex_command.assert_called_once_with(self.record, 'test:latest', ['--approve-for-me'], context=131072)
        ask.assert_called_once()  # only the top-level "Approval mode" menu -- -s/-a never even asked

    def test_approval_configuration_builds_verified_manual_flags(self):
        # Real, --help-verified flags (-s/--sandbox, -a/--ask-for-approval) --
        # not the guessed --full-auto an earlier revision shipped, which
        # doesn't exist in current Codex CLI at all.
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'codex_command', return_value=['codex']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0), \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask', side_effect=[2, 4, 3]), \
             patch('nersc_ollama_manager.codex_launcher.Confirm.ask', return_value=False):
            main(['--config', str(self.config), '--context', '131072'])
        codex_command.assert_called_once_with(
            self.record, 'test:latest',
            ['--sandbox', 'danger-full-access', '--ask-for-approval', 'never'],
            context=131072)

    def test_approval_configuration_leaves_unset_choices_unflagged(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'codex_command', return_value=['codex']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0), \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask', side_effect=[2, 1, 1]), \
             patch('nersc_ollama_manager.codex_launcher.Confirm.ask', return_value=False):
            main(['--config', str(self.config), '--context', '131072'])
        codex_command.assert_called_once_with(self.record, 'test:latest', [], context=131072)

    def test_approval_custom_preset_prompts_for_raw_arguments(self):
        with patch.object(Manager, 'list_servers', return_value=[self.record]), \
             patch.object(Manager, 'models', return_value=[{'name': 'test:latest'}]), \
             patch.object(Manager, 'codex_command', return_value=['codex']) as codex_command, \
             patch('nersc_ollama_manager.codex_launcher.subprocess.call', return_value=0), \
             patch('nersc_ollama_manager._prompts.IntPrompt.ask', return_value=4), \
             patch('nersc_ollama_manager.codex_launcher.Prompt.ask', return_value='--foo --bar baz'):
            main(['--config', str(self.config), '--context', '131072'])
        codex_command.assert_called_once_with(self.record, 'test:latest', ['--foo', '--bar', 'baz'], context=131072)

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

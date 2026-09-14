"""--remote (SSH-to-login-node) mode, allocate --screen, and the pull() fix."""
import contextlib
import getpass
import io
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import unittest
from unittest.mock import patch, Mock

from nersc_ollama_manager.core import Manager, atomic_json
from nersc_ollama_manager.cli import main, parser, tui_main


class RemoteManagerTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.m = Manager.initialize(root / 'config.json', root / 'runtime', root / 'ollama')
        self.record = dict(schema_version=1, id='test-123-abcdef', name='test', uid=os.getuid(),
                            job_id='123', host='nid000001', port=23456, state='ready', heartbeat=time.time())

    # -- Correction 1: pull() no longer raises NameError -----------------

    def test_pull_runs_without_nameerror(self):
        with patch.object(self.m, 'require_installation'), \
             patch.object(self.m, 'ensure_tunnel', return_value=34567), \
             patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.pull(self.record, 'test:latest')
        args, kwargs = run_mock.call_args
        self.assertIn('pull', args[0])
        self.assertEqual(kwargs['env']['OLLAMA_HOST'], '127.0.0.1:34567')

    def test_pull_local_checks_node_and_skips_tunnel(self):
        with patch.object(self.m, 'require_installation'), \
             patch('socket.gethostname', return_value='nid000001'), \
             patch.object(self.m, 'validate_record') as validate, \
             patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.pull({**self.record, 'host': 'nid000001'}, 'test:latest', local=True)
        validate.assert_called_once()
        self.assertEqual(run_mock.call_args.kwargs['env']['OLLAMA_HOST'], '127.0.0.1:23456')

    # -- job() / stop_allocation() / tail_log() remote branching ----------

    def test_job_local_uses_bare_squeue(self):
        proc = Mock(stdout='123|u|RUNNING|nid000001|29:00\n')
        with patch('nersc_ollama_manager.core.run', return_value=proc) as run_mock:
            self.m.job('123')
        self.assertEqual(run_mock.call_args[0][0][0], 'squeue')

    def test_job_remote_wraps_squeue_over_ssh_with_identity(self):
        self.m.remote = 'saul.nersc.gov'
        self.m.config['remote_user'] = 'nersc_user'
        self.m.config['remote_identity'] = '/tmp/id_key'
        proc = Mock(stdout='123|nersc_user|RUNNING|nid000001|29:00\n')
        with patch('nersc_ollama_manager.core.run', return_value=proc) as run_mock:
            job = self.m.job('123')
        self.assertEqual(job['id'], '123')
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], 'ssh')
        self.assertEqual(args[args.index('-l') + 1], 'nersc_user')
        self.assertEqual(args[args.index('-i') + 1], '/tmp/id_key')
        self.assertEqual(args[-2], 'saul.nersc.gov')
        self.assertIn('squeue', args[-1])

    def test_stop_allocation_remote_wraps_scancel(self):
        self.m.remote = 'saul.nersc.gov'
        with patch.object(self.m, 'validate_record'), patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.stop_allocation(self.record, yes=True)
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], 'ssh')
        self.assertIn('scancel 123', args[-1])

    def test_tail_log_local_uses_bare_tail(self):
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.tail_log(self.record['id'], 50)
        self.assertEqual(run_mock.call_args[0][0][0], 'tail')

    def test_tail_log_remote_wraps_over_ssh(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.tail_log(self.record['id'], 50)
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], 'ssh')
        self.assertIn('tail -n 50', args[-1])

    # -- list_servers() remote branch reuses `status --json` --------------

    def test_list_servers_remote_parses_status_json(self):
        self.m.remote = 'saul.nersc.gov'
        payload = [{'id': 'x', 'available': True}]
        proc = Mock(stdout=json.dumps(payload))
        with patch('nersc_ollama_manager.core.run', return_value=proc) as run_mock:
            result = self.m.list_servers()
        self.assertEqual(result, payload)
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], 'ssh')
        self.assertIn('status --json', args[-1])
        self.assertIn('nersc-ollama', args[-1])

    def test_list_servers_remote_uses_remote_python_and_config_when_set(self):
        self.m.remote = 'saul.nersc.gov'
        self.m.config['remote_python'] = '/path/to/python'
        self.m.config['remote_config'] = '/nersc/home/config.json'
        proc = Mock(stdout='[]')
        with patch('nersc_ollama_manager.core.run', return_value=proc) as run_mock:
            self.m.list_servers()
        remote_cmd = run_mock.call_args[0][0][-1]
        self.assertIn('/path/to/python -m nersc_ollama_manager', remote_cmd)
        self.assertIn('--config /nersc/home/config.json status --json', remote_cmd)

    def test_list_servers_remote_bad_json_is_actionable(self):
        self.m.remote = 'saul.nersc.gov'
        proc = Mock(stdout='not json')
        with patch('nersc_ollama_manager.core.run', return_value=proc), \
             self.assertRaisesRegex(RuntimeError, 'Unexpected output'):
            self.m.list_servers()

    # -- ensure_tunnel() ProxyCommand hop -----------------------------------

    def test_ensure_tunnel_remote_adds_proxycommand_and_identity(self):
        self.m.remote = 'saul.nersc.gov'
        control, meta = Path(self.tmp.name) / 'sock', Path(self.tmp.name) / 'meta.json'
        with patch.object(self.m, 'validate_record'), \
             patch.object(self.m, 'tunnel_paths', return_value=(control, meta)), \
             patch('nersc_ollama_manager.core.free_port', return_value=45678), \
             patch('nersc_ollama_manager.core.run') as run_mock, \
             patch('nersc_ollama_manager.core.request', return_value={'version': 'test'}):
            port = self.m.ensure_tunnel(self.record)
        self.assertEqual(port, 45678)
        args = run_mock.call_args[0][0]
        self.assertIn('-M', args)
        proxy_opts = [a for a in args if isinstance(a, str) and a.startswith('ProxyCommand=')]
        self.assertEqual(len(proxy_opts), 1)
        self.assertIn('saul.nersc.gov', proxy_opts[0])
        self.assertIn('-W', proxy_opts[0])

    def test_ensure_tunnel_local_has_no_proxycommand(self):
        control, meta = Path(self.tmp.name) / 'sock', Path(self.tmp.name) / 'meta.json'
        with patch.object(self.m, 'validate_record'), \
             patch.object(self.m, 'tunnel_paths', return_value=(control, meta)), \
             patch('nersc_ollama_manager.core.free_port', return_value=45678), \
             patch('nersc_ollama_manager.core.run') as run_mock, \
             patch('nersc_ollama_manager.core.request', return_value={'version': 'test'}):
            self.m.ensure_tunnel(self.record)
        args = run_mock.call_args[0][0]
        self.assertFalse(any(isinstance(a, str) and a.startswith('ProxyCommand=') for a in args))

    # -- _remote_run() error translation ------------------------------------

    def test_remote_run_translates_ssh_auth_failure(self):
        self.m.remote = 'saul.nersc.gov'
        error = subprocess.CalledProcessError(255, ['ssh'], output='', stderr='Permission denied')
        with patch('nersc_ollama_manager.core.run', side_effect=error), \
             self.assertRaisesRegex(RuntimeError, 'sshproxy'):
            self.m.job('123')

    # -- vendored screen helpers (allocate --screen / peek) ------------------

    def test_start_screen_session_local_stuffs_joined_command(self):
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.start_screen_session('ollama-default', ['salloc', '--time', '01:00:00'])
        calls = run_mock.call_args_list
        self.assertEqual(calls[0][0][0], ['screen', '-dmS', 'ollama-default'])
        stuff_args = calls[1][0][0]
        self.assertEqual(stuff_args[:5], ['screen', '-S', 'ollama-default', '-p', '0'])
        self.assertEqual(stuff_args[-1], "salloc --time 01:00:00\n")

    def test_capture_screen_local_runs_hardcopy_and_trims(self):
        proc = Mock(stdout='line1\nline2\nline3\n')
        with patch('nersc_ollama_manager.core.run', return_value=proc) as run_mock:
            text = self.m.capture_screen('ollama-default', lines=2)
        self.assertEqual(text, 'line2\nline3')
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], 'sh')
        self.assertIn('hardcopy', args[-1])

    def test_screen_helpers_remote_wrap_over_ssh(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.send_to_screen('ollama-default', 'echo hi')
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], 'ssh')
        self.assertIn('stuff', args[-1])

    # -- allocate(screen=True) / --remote delegation -------------------------

    def test_allocate_screen_local_starts_detached_session(self):
        with patch.object(self.m, 'require_installation'), patch.object(self.m, 'start_screen_session') as start:
            self.m.allocate('default', 'cpu', yes=True, screen=True)
        session, argv = start.call_args[0]
        self.assertEqual(session, 'ollama-default')
        self.assertIn('salloc', argv)

    def test_allocate_remote_requires_screen(self):
        self.m.remote = 'saul.nersc.gov'
        with self.assertRaises(RuntimeError):
            self.m.allocate('default', 'cpu', yes=True, screen=False)

    def test_allocate_remote_screen_delegates_over_ssh(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.allocate('default', 'gpu', account='acct_g', yes=True, screen=True)
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], 'ssh')
        remote_cmd = args[-1]
        self.assertIn('allocate --screen', remote_cmd)
        self.assertIn('--name default', remote_cmd)
        self.assertIn('--account acct_g', remote_cmd)
        self.assertIn('--yes', remote_cmd)

    def test_allocate_remote_screen_dry_run_prints_without_running(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.allocate('default', 'cpu', yes=True, dry_run=True, screen=True)
        run_mock.assert_not_called()


class RemoteCLIParsingTests(unittest.TestCase):
    def test_remote_bare_does_not_swallow_subcommand(self):
        # nargs='?' on a single --remote flag would greedily consume the
        # following subcommand token; the two-flag split must not.
        args = parser().parse_args(['--remote', 'status'])
        self.assertTrue(args.remote)
        self.assertEqual(args.command, 'status')

    def test_remote_host_overrides_config_default(self):
        args = parser().parse_args(['--remote', '--remote-host', 'saul.nersc.gov', 'status'])
        self.assertEqual(args.remote_host, 'saul.nersc.gov')
        self.assertEqual(args.command, 'status')

    def test_allocate_screen_flag_parses(self):
        args = parser().parse_args(['allocate', '--screen'])
        self.assertTrue(args.screen)

    def test_peek_parses_name(self):
        args = parser().parse_args(['peek', 'default'])
        self.assertEqual(args.name, 'default')


class RemoteCLIRejectionTests(unittest.TestCase):
    def _run(self, argv):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = main(argv)
        return code, buf.getvalue()

    def test_remote_rejects_nersc_only_commands(self):
        # 'setup' is deliberately NOT in this list -- it now delegates
        # instead of being rejected, see RemoteSetupAndCleanupTests.
        for argv in (['--remote', 'serve'], ['--remote', 'download', 'test:latest']):
            code, output = self._run(argv)
            self.assertEqual(code, 1, output)
            self.assertIn('--remote', output)

    def test_remote_allocate_without_screen_is_rejected(self):
        code, output = self._run(['--remote', 'allocate'])
        self.assertEqual(code, 1, output)
        self.assertIn('--screen', output)

    def test_remote_host_without_remote_is_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            Manager.initialize(config, Path(tmp) / 'runtime', Path(tmp) / 'ollama')
            code, output = self._run(['--config', str(config), '--remote-host', 'saul.nersc.gov', 'status'])
        self.assertEqual(code, 1, output)
        self.assertIn('--remote-host requires --remote', output)


class FollowUpTests(unittest.TestCase):
    """job() error text, forget/forget --all, --remote setup, sessions."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.m = Manager.initialize(root / 'config.json', root / 'runtime', root / 'ollama')
        self.record = dict(schema_version=1, id='test-123-abcdef', name='test', uid=os.getuid(),
                            job_id='123', host='nid000001', port=23456, state='ready', heartbeat=time.time())

    # -- job(): a confirmed-purged job ID is "not running", not an error ---

    def test_job_local_purged_job_id_returns_none(self):
        error = subprocess.CalledProcessError(1, ['squeue'], output='',
                                               stderr='slurm_load_jobs error: Invalid job id specified\n')
        with patch('nersc_ollama_manager.core.run', side_effect=error):
            self.assertIsNone(self.m.job('123'))

    def test_job_local_other_failure_still_raises(self):
        error = subprocess.CalledProcessError(1, ['squeue'], output='', stderr='squeue: command not found\n')
        with patch('nersc_ollama_manager.core.run', side_effect=error), self.assertRaises(subprocess.CalledProcessError):
            self.m.job('123')

    def test_job_remote_purged_job_id_returns_none(self):
        self.m.remote = 'saul.nersc.gov'
        error = subprocess.CalledProcessError(1, ['ssh'], output='',
                                               stderr='slurm_load_jobs error: Invalid job id specified\n')
        with patch('nersc_ollama_manager.core.run', side_effect=error):
            self.assertIsNone(self.m.job('123'))

    def test_delete_expired_now_succeeds_once_job_is_confirmed_gone(self):
        path = self.m.records / (self.record['id'] + '.json')
        atomic_json(path, self.record)
        error = subprocess.CalledProcessError(1, ['squeue'], output='', stderr='Invalid job id specified\n')
        with patch('nersc_ollama_manager.core.run', side_effect=error):
            self.m.delete_expired(self.record['id'])
        self.assertFalse(path.exists())

    # -- forget / forget --all ----------------------------------------------

    def test_forget_all_local_removes_confirmed_gone_keeps_still_running(self):
        gone = {**self.record, 'id': 'a', 'job_id': '123'}
        still_running_but_stale = {**self.record, 'id': 'b', 'job_id': '456', 'heartbeat': time.time() - 100}
        atomic_json(self.m.records / 'a.json', gone)
        atomic_json(self.m.records / 'b.json', still_running_but_stale)

        def fake_job(job_id):
            if job_id == '123':
                return None
            return dict(id=job_id, user=getpass.getuser(), state='RUNNING', nodes='nid000001', remaining='10:00')

        with patch.object(self.m, 'job', side_effect=fake_job):
            self.m.forget(purge_all=True)
        self.assertFalse((self.m.records / 'a.json').exists())
        self.assertTrue((self.m.records / 'b.json').exists())  # still running -- delete_expired refuses

    def test_forget_remote_delegates_single_id(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run', return_value=Mock(stdout='Removed test-123.\n')) as run_mock:
            self.m.forget('test-123')
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], 'ssh')
        self.assertIn('forget test-123', args[-1])

    def test_forget_remote_delegates_all(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run', return_value=Mock(stdout='Removed 0 expired record(s): none.\n')) as run_mock:
            self.m.forget(purge_all=True)
        self.assertIn('forget --all', run_mock.call_args[0][0][-1])

    # -- --remote setup delegation -------------------------------------------

    def test_setup_remote_argv_with_all_options(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.setup_remote('/cfs/runtime', '/cfs/ollama', '0.34.0')
        args = run_mock.call_args[0][0]
        self.assertEqual(args[0], 'ssh')
        self.assertIn('setup --runtime /cfs/runtime --root /cfs/ollama --version 0.34.0', args[-1])
        self.assertNotIn('capture_output', run_mock.call_args.kwargs)  # streams live, not buffered

    def test_setup_remote_argv_no_options(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.setup_remote(None, None, None)
        remote_cmd = run_mock.call_args[0][0][-1]
        self.assertIn('nersc-ollama setup', remote_cmd)
        self.assertNotIn('--runtime', remote_cmd)
        self.assertNotIn('--root', remote_cmd)
        self.assertNotIn('--version', remote_cmd)

    def test_cli_remote_setup_delegates_instead_of_being_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            Manager.initialize(config, Path(tmp) / 'runtime', Path(tmp) / 'ollama')
            with patch.object(Manager, 'setup_remote') as setup_remote_mock:
                code = main(['--config', str(config), '--remote', '--remote-host', 'saul.nersc.gov',
                             'setup', '--runtime', '/cfs/r', '--root', '/cfs/o'])
        self.assertEqual(code, 0)
        setup_remote_mock.assert_called_once_with('/cfs/r', '/cfs/o', None, None)

    # -- sessions -------------------------------------------------------------

    def test_list_screen_sessions_parses_names(self):
        sample = ('There are screens on:\n'
                   '\t12345.ollama-default\t(Detached)\n'
                   '\t23456.ollama-gpu\t(Attached)\n'
                   '2 Sockets in /run/screen/S-user.\n')
        with patch('nersc_ollama_manager.core.run', return_value=Mock(stdout=sample, stderr='')):
            self.assertEqual(self.m.list_screen_sessions(), ['ollama-default', 'ollama-gpu'])

    def test_list_screen_sessions_no_sessions_is_empty(self):
        error = subprocess.CalledProcessError(1, ['screen', '-ls'], output='',
                                               stderr='No Sockets found in /run/screen/S-user.\n')
        with patch('nersc_ollama_manager.core.run', side_effect=error):
            self.assertEqual(self.m.list_screen_sessions(), [])

    def test_list_screen_sessions_other_failure_raises(self):
        error = subprocess.CalledProcessError(1, ['screen', '-ls'], output='', stderr='screen: command not found\n')
        with patch('nersc_ollama_manager.core.run', side_effect=error), self.assertRaises(subprocess.CalledProcessError):
            self.m.list_screen_sessions()

    def test_list_screen_sessions_remote_no_sessions_is_empty(self):
        self.m.remote = 'saul.nersc.gov'
        error = subprocess.CalledProcessError(1, ['ssh'], output='',
                                               stderr='No Sockets found in /run/screen/S-user.\n')
        with patch('nersc_ollama_manager.core.run', side_effect=error):
            self.assertEqual(self.m.list_screen_sessions(), [])

    # -- CLI parsing ------------------------------------------------------------

    def test_forget_and_sessions_parse(self):
        args = parser().parse_args(['forget', 'abc123'])
        self.assertEqual(args.id, 'abc123')
        self.assertFalse(args.all)
        args = parser().parse_args(['forget', '--all'])
        self.assertIsNone(args.id)
        self.assertTrue(args.all)
        self.assertEqual(parser().parse_args(['sessions']).command, 'sessions')

    def test_forget_requires_exactly_one_of_id_or_all(self):
        for argv in (['forget'], ['forget', 'abc', '--all']):
            with tempfile.TemporaryDirectory() as tmp:
                config = Path(tmp) / 'config.json'
                Manager.initialize(config, Path(tmp) / 'runtime', Path(tmp) / 'ollama')
                buf = io.StringIO()
                with contextlib.redirect_stdout(buf):
                    code = main(['--config', str(config), *argv])
            self.assertEqual(code, 1, buf.getvalue())


class GpuSpreadFlagTests(unittest.TestCase):
    """--gpu-spread/--no-gpu-spread on `allocate` (CLI counterpart to the TUI switch)."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.m = Manager.initialize(root / 'config.json', root / 'runtime', root / 'ollama')

    def test_allocation_command_spread_override_ignores_persisted_value(self):
        self.m.set_gpu_spread(True)
        off = self.m.allocation_command('gpu', 'gpu', 'example_g', spread_override=False)
        self.assertNotIn('--gpu-bind', off)
        self.m.set_gpu_spread(False)
        on = self.m.allocation_command('gpu', 'gpu', 'example_g', spread_override=True)
        self.assertIn('--gpu-bind', on)
        self.assertEqual(on[on.index('--gpu-bind') + 1], 'none')

    def test_gpu_spread_rejected_for_cpu_profile(self):
        with self.assertRaisesRegex(ValueError, 'only apply to --profile gpu'):
            self.m.allocate('cpu', 'cpu', yes=True, gpu_spread=True)

    def test_allocate_persists_gpu_spread_after_confirm(self):
        with patch.object(self.m, 'require_installation'), patch('nersc_ollama_manager.core.run'):
            self.m.allocate('gpu', 'gpu', 'example_g', yes=True, gpu_spread=True)
        self.assertTrue(Manager(self.m.config_path).gpu_spread())

    def test_allocate_dry_run_does_not_persist_but_previews_flag(self):
        buf = io.StringIO()
        with patch.object(self.m, 'require_installation'), contextlib.redirect_stdout(buf):
            self.m.allocate('gpu', 'gpu', 'example_g', dry_run=True, gpu_spread=True)
        self.assertIn('--gpu-bind', buf.getvalue())
        self.assertFalse(Manager(self.m.config_path).gpu_spread())  # unset default, not persisted

    def test_allocate_remote_screen_forwards_gpu_spread_flag(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.allocate('gpu', 'gpu', 'example_g', yes=True, screen=True, gpu_spread=False)
        remote_cmd = run_mock.call_args[0][0][-1]
        self.assertIn('--no-gpu-spread', remote_cmd)

    def test_cli_parses_three_states(self):
        self.assertIsNone(parser().parse_args(['allocate']).gpu_spread)
        self.assertTrue(parser().parse_args(['allocate', '--gpu-spread']).gpu_spread)
        self.assertFalse(parser().parse_args(['allocate', '--no-gpu-spread']).gpu_spread)


class TuiMainTests(unittest.TestCase):
    """nersc-ollama-tui always appends 'tui' -- passing a subcommand of your own
    (e.g. under --remote) must be rejected clearly, not with argparse's confusing
    "unrecognized arguments: tui", and --help/--version must still print once."""

    def test_rejects_subcommand_with_clear_message(self):
        buf = io.StringIO()
        with contextlib.redirect_stdout(buf):
            code = tui_main(['--remote', '--remote-host', 'saul.nersc.gov', 'sessions'])
        self.assertEqual(code, 1)
        message = buf.getvalue()
        self.assertIn('sessions', message)
        self.assertIn('nersc-ollama-tui always launches the TUI', message)
        self.assertNotIn('unrecognized arguments', message)

    def test_help_prints_exactly_once(self):
        buf = io.StringIO()
        with self.assertRaises(SystemExit), contextlib.redirect_stdout(buf):
            tui_main(['--help'])
        self.assertEqual(buf.getvalue().count('usage:'), 1)

    def test_plain_global_flags_still_launch_tui(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            Manager.initialize(config, Path(tmp) / 'runtime', Path(tmp) / 'ollama')
            with patch('nersc_ollama_manager.cli.main') as main_mock:
                tui_main(['--config', str(config)])
        main_mock.assert_called_once_with(['--config', str(config), 'tui'])


class OllamaBinaryOverrideTests(unittest.TestCase):
    """ollama_binary: point at an Ollama that already exists on NERSC some
    other way, instead of this tool's own pinned-release install."""

    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        root = Path(self.tmp.name)
        self.m = Manager.initialize(root / 'config.json', root / 'runtime', root / 'ollama')

    def test_default_binary_is_under_ollama_root(self):
        self.assertEqual(self.m.binary, self.m.root / 'current/bin/ollama')

    def test_set_ollama_binary_overrides_resolution(self):
        existing = Path(self.tmp.name) / 'shared' / 'ollama'
        existing.parent.mkdir()
        existing.touch()
        self.m.set_ollama_binary(str(existing))
        self.assertEqual(self.m.binary, existing)
        # Reopening the same config must resolve the override again.
        self.assertEqual(Manager(self.m.config_path).binary, existing)

    def test_set_ollama_binary_rejects_relative_path(self):
        with self.assertRaises(ValueError):
            self.m.set_ollama_binary('relative/ollama')

    def test_require_installation_message_reflects_override(self):
        self.m.set_ollama_binary('/does/not/exist/ollama')
        with self.assertRaisesRegex(RuntimeError, 'Configured ollama_binary does not exist'):
            self.m.require_installation()

    def test_install_refuses_when_binary_overridden(self):
        self.m.set_ollama_binary('/opt/shared/ollama')
        with self.assertRaisesRegex(RuntimeError, "ollama_binary is configured"):
            self.m.install('0.34.0')

    def test_cli_setup_sets_ollama_binary_on_new_config(self):
        with tempfile.TemporaryDirectory() as tmp:
            config = Path(tmp) / 'config.json'
            code = main(['--config', str(config), 'setup', '--runtime', str(Path(tmp) / 'runtime'),
                         '--root', str(Path(tmp) / 'ollama'), '--ollama-binary', '/opt/shared/ollama'])
            self.assertEqual(code, 0)
            self.assertEqual(Manager(config).config.get('ollama_binary'), '/opt/shared/ollama')

    def test_setup_remote_forwards_ollama_binary(self):
        self.m.remote = 'saul.nersc.gov'
        with patch('nersc_ollama_manager.core.run') as run_mock:
            self.m.setup_remote(None, None, None, '/opt/shared/ollama')
        self.assertIn('--ollama-binary /opt/shared/ollama', run_mock.call_args[0][0][-1])


if __name__ == '__main__':
    unittest.main()

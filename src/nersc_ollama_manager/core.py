"""Lifecycle operations. All subprocess commands use argument arrays."""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import shlex
import shutil
import signal
import socket
import subprocess
import tarfile
import tempfile
import time
import urllib.request
import uuid

def xdg_path(variable, fallback):
    value = os.environ.get(variable)
    return Path(value) if value and Path(value).is_absolute() else Path.home() / fallback


def config_location(explicit=None):
    selected = explicit or os.environ.get('NERSC_OLLAMA_CONFIG')
    return (Path(selected).expanduser().absolute(), True) if selected else (
        xdg_path('XDG_CONFIG_HOME', '.config') / 'nersc-ollama/config.json', False)


def default_runtime():
    return xdg_path('XDG_DATA_HOME', '.local/share') / 'nersc-ollama'



def run(args, **kwargs):
    if not kwargs.pop('stream_errors', False):
        return subprocess.run([str(a) for a in args], check=True, **kwargs)
    from collections import deque
    tail = deque(maxlen=20)
    command = [str(a) for a in args]
    with subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                          text=True, errors='replace', bufsize=1, **kwargs) as proc:
        try:
            for line in proc.stdout:
                print(line, end='', flush=True)
                tail.append(line[-2000:].rstrip())
            code = proc.wait()
        except BaseException:
            proc.terminate()
            try:
                proc.wait(timeout=10)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            raise
    if code:
        details = '\n'.join(tail).strip() or 'No diagnostic output was produced.'
        if 'QOSMaxWallDurationPerJobLimit' in details:
            details += (
                '\nHint: reduce the requested allocation time or choose a QOS/account '
                'that permits that walltime. Perlmutter interactive QOS currently '
                'allows up to 4 hours; debug QOS allows up to 30 minutes.'
            )
        raise RuntimeError(f'{command[0]} failed (exit {code}):\n{details}')
    return subprocess.CompletedProcess(command, code)


def atomic_json(path, value, create_only=False):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix='.write-')
    try:
        with os.fdopen(fd, 'w') as stream:
            json.dump(value, stream, indent=2)
            stream.write('\n')
        if create_only:
            try:
                os.link(tmp, path)
            except FileExistsError:
                pass
        else:
            os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            os.unlink(tmp)


@contextlib.contextmanager
def stream_log(path):
    """Tee the worker's saved Ollama log to its allocation terminal."""
    import threading
    done = threading.Event()
    def follow():
        with open(path, errors='replace') as source:
            while True:
                data = source.read(8192)
                if data:
                    print(data, end='', flush=True)
                elif done.is_set():
                    break
                else:
                    done.wait(.2)
    thread = threading.Thread(target=follow, daemon=True)
    thread.start()
    try:
        yield
    finally:
        done.set()
        thread.join(timeout=5)


def read_json(path):
    with open(path) as stream:
        return json.load(stream)


def request(port, route='/api/version', data=None, timeout=5):
    body = None if data is None else json.dumps(data).encode()
    req = urllib.request.Request(f'http://127.0.0.1:{int(port)}{route}', data=body,
                                 headers={'Content-Type': 'application/json'})
    # Never send loopback requests through a configured proxy.
    with urllib.request.build_opener(urllib.request.ProxyHandler({})).open(req, timeout=timeout) as response:
        return json.load(response)


def free_port():
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        return sock.getsockname()[1]


def confirm(message, approved=False):
    if approved:
        return
    if not os.isatty(0) or input(message + ' [y/N] ').lower() not in ('y', 'yes'):
        raise RuntimeError('Not approved; no scheduler action taken.')


def safe_name(value):
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.-]{0,127}', value):
        raise ValueError('Names must contain 1–128 letters, digits, dots, underscores, or hyphens.')
    return value


class Manager:
    def __init__(self, config=None):
        self.config_path, _ = config_location(config)
        self.config = read_json(self.config_path)
        if not isinstance(self.config, dict) or self.config.get('schema_version') != 1:
            raise ValueError('Unsupported configuration schema; expected schema_version 1.')
        for key in ('runtime', 'ollama_root'):
            if not isinstance(self.config.get(key), str) or not Path(self.config[key]).is_absolute():
                raise ValueError(f'Configuration {key} must be an absolute path.')
        if self.config.get('ollama_binary') is not None:
            if not isinstance(self.config['ollama_binary'], str) or not Path(self.config['ollama_binary']).is_absolute():
                raise ValueError('Configuration ollama_binary must be an absolute path.')
        profiles = self.config.get('profiles')
        if not isinstance(profiles, dict) or not all(isinstance(profiles.get(k), dict) for k in ('cpu', 'gpu')):
            raise ValueError('Configuration must define CPU and GPU profiles.')
        for profile in profiles.values():
            if not all(k in profile for k in ('constraint', 'qos', 'time', 'nodes', 'gpus', 'context')):
                raise ValueError('Incomplete resource profile.')
        self.runtime = Path(self.config['runtime']).absolute()
        self.root = Path(self.config['ollama_root']).absolute()
        self.records = self.runtime / 'servers'
        self.logs = self.runtime / 'logs'
        # ollama_binary opts out of this tool's own pinned-release install/symlink
        # management entirely, in favor of an Ollama that already exists on NERSC
        # some other way (a module, a shared install, ...).
        self.binary = (Path(self.config['ollama_binary']).absolute() if self.config.get('ollama_binary')
                        else self.root / 'current/bin/ollama')
        self.remote = None  # resolved login-node host when running with --remote

    @classmethod
    def initialize(cls, config, runtime=None, root=None):
        if Path(config).exists():
            return cls(config)
        runtime = Path(runtime or default_runtime()).expanduser().absolute()
        root = Path(root or runtime / 'ollama_root').expanduser().absolute()
        for path in (runtime, root, runtime / 'servers', runtime / 'logs'):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        if not Path(config).exists():
            atomic_json(config, {
                'schema_version': 1, 'runtime': str(runtime), 'ollama_root': str(root),
                'startup_timeout': 300,
                'profiles': {
                    'cpu': {'constraint': 'cpu', 'qos': 'interactive', 'time': '00:30:00',
                            'nodes': 1, 'gpus': 0, 'account': None, 'context': 65536},
                    'gpu': {'constraint': 'gpu', 'qos': 'interactive', 'time': '00:30:00',
                            'nodes': 1, 'gpus': 4, 'account': None, 'context': 65536},
                },
            }, create_only=True)
        return cls(config)

    @classmethod
    def open(cls, config=None):
        path, explicit = config_location(config)
        if not path.exists():
            if explicit:
                raise RuntimeError(f'Configuration does not exist: {path}. Run nersc-ollama --config {shlex.quote(str(path))} setup.')
            return cls.initialize(path)
        return cls(path)

    # --- --remote support: run client-side subcommands from outside NERSC ---
    # over SSH to a login node, using explicit sshproxy-key identity args
    # (rather than requiring a pre-existing ~/.ssh/config ProxyJump).
    #
    # Think in terms of two roles, not two machines: the NERSC server side
    # (Slurm, the compute node, Ollama itself -- always on NERSC) and the
    # Codex client side (wherever this process is actually running -- a
    # login node, some other node, or a laptop). self.remote/--remote is
    # just "does the client side need an SSH hop to reach the server side,
    # or is it already sitting there" -- not a statement about which
    # machine does the real work (the server side always does).

    def _remote_identity_args(self):
        import getpass
        user = self.config.get('remote_user') or getpass.getuser()
        identity = os.path.expanduser(self.config.get('remote_identity') or '~/.ssh/nersc')
        return ['-l', user, '-i', identity, '-o', 'IdentitiesOnly=yes']

    def _login_ssh_command(self, remote_argv):
        return ['ssh', '-o', 'ConnectTimeout=10', *self._remote_identity_args(),
                self.remote, shlex.join(str(a) for a in remote_argv)]

    def _compute_proxy_option(self):
        proxy = shlex.join(['ssh', *self._remote_identity_args(), '-W', '%h:%p', self.remote])
        return ['-o', f'ProxyCommand={proxy}']

    def _remote_cli_argv(self, *subcommand_args):
        """Build the argv for invoking this same package's CLI on the login
        node -- delegating a whole subcommand there, rather than
        reimplementing it over SSH (matches _list_servers_remote's use of
        `status --json`)."""
        python = self.config.get('remote_python')
        argv = [python, '-m', 'nersc_ollama_manager'] if python else ['nersc-ollama']
        if self.config.get('remote_config'):
            argv += ['--config', self.config['remote_config']]
        return argv + list(subcommand_args)

    def resolve_remote(self, remote, remote_host=None):
        """Shared --remote/--remote-host resolution, used by both nersc-ollama
        and nersc-ollama-codex so the two entry points behave identically."""
        if remote:
            self.remote = remote_host or self.config.get('remote_login_host')
            if not self.remote:
                raise RuntimeError('Pass --remote-host HOST, or set remote_login_host in the configuration.')
        elif remote_host:
            raise RuntimeError('--remote-host requires --remote.')

    def _remote_run(self, argv, **kwargs):
        """Run argv on the login node over SSH, translating common failure
        modes (auth, missing remote command) into actionable errors."""
        try:
            return run(self._login_ssh_command(argv), **kwargs)
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip() if isinstance(exc.stderr, str) else ''
            if exc.returncode == 255:
                hint = ('SSH could not reach the login node; if using a sshproxy key '
                        'it may have expired (~24h lifetime) — rerun sshproxy and try again.')
            elif 'not found' in stderr or 'No such file' in stderr:
                hint = ("The remote command was not found on PATH for a non-interactive SSH "
                        "session. Set 'remote_python' in your configuration to the interpreter "
                        "with nersc-ollama-manager installed.")
            else:
                hint = stderr or 'See stderr above for details.'
            raise RuntimeError(f'Remote command on {self.remote} failed: {hint}') from exc

    def gpu_spread(self):
        value = self.config['profiles']['gpu'].get('sched_spread')
        if value is None:
            return os.environ.get('OLLAMA_SCHED_SPREAD', '').lower() in ('1', 'true', 'yes', 'on')
        if not isinstance(value, bool):
            raise ValueError('GPU sched_spread must be true or false.')
        return value

    def set_gpu_spread(self, enabled):
        if not isinstance(enabled, bool):
            raise ValueError('GPU sched_spread must be true or false.')
        config = read_json(self.config_path)
        config['profiles']['gpu']['sched_spread'] = enabled
        atomic_json(self.config_path, config)
        self.config = config

    def startup_timeout(self, profile):
        value = self.config['profiles'][profile].get('startup_timeout', self.config.get('startup_timeout', 300))
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not 0 < value <= 86400:
            raise ValueError('startup_timeout must be a number of seconds greater than zero and at most 86400.')
        return value

    def delete_expired(self, identity):
        safe_name(identity)
        path = self.records / (identity + '.json')
        if path.is_symlink() or path.stat().st_uid != os.getuid():
            raise ValueError('Refusing to remove a discovery record not owned by this user.')
        record = read_json(path)
        if record.get('id') != identity or record.get('uid') != os.getuid():
            raise ValueError('Discovery record identity/owner mismatch.')
        # A stale heartbeat is not evidence that a job ended. Fail closed on scheduler errors.
        if self.job(record['job_id']) is not None:
            raise RuntimeError('Allocation is still present in Slurm; only expired allocation records can be removed.')
        path.unlink()

    def forget(self, identity=None, purge_all=False):
        """CLI-facing wrapper around delete_expired(): a single ID, or a
        best-effort sweep of every currently-unavailable record."""
        if self.remote:
            argv = self._remote_cli_argv('forget', *(['--all'] if purge_all else [identity]))
            result = self._remote_run(argv, capture_output=True, text=True, timeout=25)
            print(result.stdout, end='')
            return
        if purge_all:
            removed, kept = [], []
            for record in self.list_servers():
                if record['available']:
                    continue
                try:
                    self.delete_expired(record['id'])
                    removed.append(record['id'])
                except (OSError, ValueError, KeyError, RuntimeError) as exc:
                    kept.append((record['id'], str(exc)))
            print(f'Removed {len(removed)} expired record(s): {", ".join(removed) or "none"}.')
            for rid, error in kept:
                print(f'Kept {rid}: {error}')
            return
        self.delete_expired(identity)
        print(f'Removed {identity}.')

    def require_installation(self):
        if not self.binary.is_file():
            if self.config.get('ollama_binary'):
                raise RuntimeError(f'Configured ollama_binary does not exist or is not a file: {self.binary}')
            raise RuntimeError(f'Ollama is not installed. Run nersc-ollama --config {shlex.quote(str(self.config_path))} setup --version VERSION.')

    def set_storage(self, runtime, root):
        if self.binary.exists() or any(self.records.glob('*.json')):
            raise RuntimeError('Storage cannot be changed by onboarding after installation or server creation.')
        runtime, root = Path(runtime).expanduser(), Path(root).expanduser()
        if not runtime.is_absolute() or not root.is_absolute():
            raise ValueError('Choose absolute storage paths.')
        for path in (runtime, root, runtime / 'servers', runtime / 'logs'):
            path.mkdir(parents=True, exist_ok=True, mode=0o700)
        config = {**self.config, 'runtime': str(runtime), 'ollama_root': str(root)}
        atomic_json(self.config_path, config)
        self.__init__(self.config_path)

    def set_ollama_binary(self, path):
        """Point at an Ollama that already exists on NERSC some other way
        (a module, a shared install, ...) instead of this tool's own pinned
        release under ollama_root/current. No restriction on when this can
        be changed -- unlike set_storage(), it never moves or orphans files,
        just which binary self.binary resolves to."""
        binary = Path(path).expanduser()
        if not binary.is_absolute():
            raise ValueError('Choose an absolute ollama_binary path.')
        config = {**self.config, 'ollama_binary': str(binary)}
        atomic_json(self.config_path, config)
        self.__init__(self.config_path)

    def diagnostics(self):
        return {'config': str(self.config_path), 'python': os.sys.executable,
                'runtime': str(self.runtime), 'ollama_root': str(self.root),
                'ollama_binary': str(self.binary), 'ollama_binary_override': self.config.get('ollama_binary'),
                'ollama_installed': self.binary.is_file(),
                'commands': {name: shutil.which(name) for name in
                             ('codex', 'salloc', 'srun', 'squeue', 'scancel', 'ssh', 'curl', 'zstd')}}

    def setup_remote(self, runtime, root, version, ollama_binary=None):
        """Delegate `setup` to the login node over SSH, so the NERSC side can
        be provisioned from a client machine without a manual SSH login
        first. This Manager's own local config is used only to resolve the
        remote_* SSH settings -- its runtime/ollama_root are never touched."""
        argv = self._remote_cli_argv('setup')
        if runtime:
            argv += ['--runtime', runtime]
        if root:
            argv += ['--root', root]
        if ollama_binary:
            argv += ['--ollama-binary', ollama_binary]
        if version:
            argv += ['--version', version]
        # No capture: --version triggers a real download that can take a
        # while, and buffering it would make this look hung until it's done.
        run(self._login_ssh_command(argv))

    def install(self, version):
        if self.config.get('ollama_binary'):
            raise RuntimeError(f"ollama_binary is configured ({self.config['ollama_binary']}); this tool won't "
                                "manage a separate pinned install alongside it. Remove ollama_binary from your "
                                "configuration first if you want a self-managed install instead.")
        if not re.fullmatch(r'\d+\.\d+\.\d+(?:[-.][A-Za-z0-9.]+)?', version):
            raise ValueError('Supply an explicit release version, e.g. 0.18.0.')
        arch = {'x86_64': 'amd64', 'aarch64': 'arm64'}.get(platform.machine())
        if not arch or platform.system() != 'Linux':
            raise RuntimeError('Binary installation supports Linux amd64/arm64.')
        if not shutil.which('zstd'):
            raise RuntimeError('zstd is required to unpack Ollama.')
        releases = self.root / 'releases'
        releases.mkdir(exist_ok=True)
        dest = releases / version
        if dest.exists():
            raise RuntimeError(f'{dest} already exists; select a new version or use the current installation.')
        url = f'https://github.com/ollama/ollama/releases/download/v{version}/ollama-linux-{arch}.tar.zst'
        # Stage on the destination filesystem; no scratch storage.
        with tempfile.TemporaryDirectory(prefix='.install-', dir=releases) as tmp:
            stage = Path(tmp)
            archive = stage / 'bundle.tar.zst'
            run(['curl', '--fail', '--location', '--retry', '2', url, '--output', archive])
            with archive.open('rb') as stream:
                digest = hashlib.file_digest(stream, 'sha256').hexdigest()
            unpack = stage / 'unpacked'
            unpack.mkdir()
            with subprocess.Popen(['zstd', '-dc', str(archive)], stdout=subprocess.PIPE) as proc:
                with tarfile.open(fileobj=proc.stdout, mode='r|') as tar:
                    tar.extractall(unpack, filter='data')
                if proc.wait() != 0:
                    raise RuntimeError('Archive decompression failed.')
            if not (unpack / 'bin/ollama').is_file():
                raise RuntimeError('Archive does not contain bin/ollama.')
            atomic_json(unpack / 'installation.json', {'version': version, 'url': url, 'sha256': digest})
            os.rename(unpack, dest)
        link = self.root / ('.current-' + uuid.uuid4().hex)
        link.symlink_to(dest)
        os.replace(link, self.root / 'current')
        (self.root / 'models').mkdir(exist_ok=True, mode=0o700)
        return dest

    def profile(self, name, account=None, walltime=None):
        p = dict(self.config['profiles'][name])
        if account:
            p['account'] = account
        if walltime:
            p['time'] = walltime
        if not re.fullmatch(r'\d{2,3}:\d{2}:\d{2}', p['time']):
            raise ValueError('Walltime must be HH:MM:SS.')
        if p['gpus'] and not p.get('account'):
            raise ValueError('Set the GPU account in config or pass --account (NERSC GPU accounts end in _g).')
        if int(p['nodes']) != 1 or int(p['gpus']) < 0:
            raise ValueError('Version 1 supports one node per server and a nonnegative GPU count.')
        return p

    def allocation_command(self, name, profile, account=None, walltime=None, spread_override=None):
        safe_name(name)
        p = self.profile(profile, account, walltime)
        args = ['salloc', '--nodes', '1', '--qos', p['qos'], '--time', p['time'],
                '--constraint', p['constraint'], '--job-name', 'ollama-' + name]
        if p.get('account'):
            args += ['--account', p['account']]
        if p['gpus']:
            args += ['--gpus-per-node', str(p['gpus'])]
        args += ['srun', '--nodes', '1', '--ntasks', '1', '--unbuffered']
        if p['gpus']:
            args += ['--gpus', str(p['gpus'])]
            # spread_override previews a not-yet-persisted --gpu-spread/--no-gpu-spread
            # request (e.g. under --dry-run) without reading it back from config.
            spread = self.gpu_spread() if spread_override is None else spread_override
            if spread:
                args += ['--gpu-bind', 'none']
        args += [os.sys.executable, '-m', 'nersc_ollama_manager', '--config', str(self.config_path),
                 'serve', '--name', name, '--profile', profile]
        return args

    # --- detached `screen` session on the login node, for `allocate --screen` ---
    # A dropped SSH connection sends SIGHUP to salloc's foreground process
    # tree, tearing down the allocation (see salloc(1)). Running it inside a
    # detached screen session on the login node means SIGHUP never reaches
    # it. Command shapes (start/stuff/hardcopy) follow matplo/scrn-mgr's
    # screen_backend.py, reimplemented directly here (rather than depending
    # on that package) so the sshproxy identity key applies automatically,
    # with no separate ~/.ssh/config entry required.

    def _screen_run(self, argv, **kwargs):
        kwargs.setdefault('capture_output', True)
        kwargs.setdefault('text', True)
        kwargs.setdefault('timeout', 15)
        runner = self._remote_run if self.remote else run
        return runner(argv, **kwargs)

    def start_screen_session(self, session, argv):
        safe_name(session)
        self._screen_run(['screen', '-dmS', session])
        self.send_to_screen(session, shlex.join(str(a) for a in argv))

    def send_to_screen(self, session, text):
        if not text.endswith('\n'):
            text += '\n'
        self._screen_run(['screen', '-S', session, '-p', '0', '-X', 'stuff', text])

    def capture_screen(self, session, lines=200):
        """One-shot snapshot of the session's currently visible pane -- not
        full scrollback, and not followed/tailed. Useful as a human-facing
        peek ("is salloc still queueing?"); never load-bearing -- status and
        logs rely on the atomic discovery record and tail_log() instead."""
        remote_cmd = (f"t=$(mktemp); screen -S {shlex.quote(session)} -p 0 -X hardcopy \"$t\" "
                      f'&& cat "$t" 2>/dev/null; rm -f "$t"')
        result = self._screen_run(['sh', '-c', remote_cmd])
        text = result.stdout
        return '\n'.join(text.splitlines()[-lines:]) if lines else text

    def list_screen_sessions(self):
        """Parse `screen -ls` (local, or via SSH under --remote) for
        ollama-* sessions. No registry -- discovery records already track
        real servers; this only fills the narrow gap where one doesn't exist
        yet (a --screen allocation still queueing) or you forgot a --name."""
        try:
            result = self._screen_run(['screen', '-ls'])
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            # `screen -ls` conventionally exits nonzero when nothing is
            # running ("No Sockets found...") -- that's an empty list, not
            # a failure to look.
            text = getattr(exc, 'stdout', None) or getattr(exc, 'stderr', None) or str(exc)
            if isinstance(text, str) and 'no sockets found' in text.lower():
                return []
            raise
        text = (result.stdout or '') + (getattr(result, 'stderr', '') or '')
        return sorted(set(re.findall(r'\d+\.(ollama-[\w.-]+)\s', text)))

    def _allocate_remote_screen(self, name, profile, account, walltime, yes, dry_run, gpu_spread):
        """Delegate to `nersc-ollama allocate --screen` run *on the login
        node* over SSH, rather than screen-wrapping a locally-built salloc
        command: allocation_command() bakes in this Manager's own
        config_path/ollama_root, which are meaningless off the NERSC side."""
        safe_name(name)
        remote_cmd = self._remote_cli_argv('allocate', '--screen', '--name', name, '--profile', profile)
        if account:
            remote_cmd += ['--account', account]
        if walltime:
            remote_cmd += ['--time', walltime]
        if gpu_spread is not None:
            remote_cmd += ['--gpu-spread' if gpu_spread else '--no-gpu-spread']
        remote_cmd += ['--yes']  # confirmation already happens below, on this side
        print(shlex.join(remote_cmd), flush=True)
        if dry_run:
            return
        confirm(f'Request this Slurm allocation on {self.remote}, detached in a screen session there?', yes)
        self._remote_run(remote_cmd, capture_output=True, text=True, timeout=25)

    def allocate(self, name, profile, account=None, walltime=None, yes=False, dry_run=False, screen=False,
                 gpu_spread=None):
        if gpu_spread is not None and profile != 'gpu':
            raise ValueError('--gpu-spread/--no-gpu-spread only apply to --profile gpu.')
        if self.remote:
            if not screen:
                raise RuntimeError('--remote allocate requires --screen (a foreground salloc '
                                    'cannot run over one SSH round-trip).')
            return self._allocate_remote_screen(name, profile, account, walltime, yes, dry_run, gpu_spread)
        if os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Already inside an allocation; use serve rather than nesting salloc.')
        self.require_installation()
        args = self.allocation_command(name, profile, account, walltime, spread_override=gpu_spread)
        print(shlex.join(args), flush=True)
        if dry_run:
            return
        confirm('Request these Slurm resources?', yes)
        if gpu_spread is not None:
            # Persisted only once we're actually proceeding, matching the TUI
            # switch's "applies to newly started GPU servers" semantics.
            self.set_gpu_spread(gpu_spread)
            print(f'GPU spreading {"enabled" if gpu_spread else "disabled"} for future GPU allocations.', flush=True)
        if screen:
            session = 'ollama-' + name
            self.start_screen_session(session, args)
            print(f"Started detached screen session '{session}' on the login node. "
                  f"Use `nersc-ollama peek {name}` or `logs {name}` to check progress; "
                  "it survives this connection dropping.", flush=True)
            return
        return run(args, stream_errors=True)

    def job(self, job_id):
        if not re.fullmatch(r'\d+', str(job_id)):
            raise ValueError('Invalid job ID.')
        squeue = ['squeue', '--noheader', '--jobs', str(job_id), '--format', '%i|%u|%T|%N|%L']
        runner = self._remote_run if self.remote else run
        try:
            result = runner(squeue, capture_output=True, text=True, timeout=25 if self.remote else 15)
        except (subprocess.CalledProcessError, RuntimeError) as exc:
            # squeue exits nonzero once a job ages out of Slurm's accounting
            # entirely ("Invalid job id specified") -- that means the job is
            # gone, not that the query failed. Anything else (squeue missing,
            # permission denied, timeout, SSH auth failure in --remote mode)
            # still raises, so callers like delete_expired() stay fail-closed
            # on genuine scheduler errors.
            stderr = getattr(exc, 'stderr', None) or str(exc)
            if isinstance(stderr, str) and 'invalid job id' in stderr.lower():
                return None
            raise
        for line in result.stdout.splitlines():
            fields = line.strip().split('|')
            if len(fields) == 5 and fields[0] == str(job_id):
                return dict(zip(('id', 'user', 'state', 'nodes', 'remaining'), fields))
        return None

    def validate_record(self, record):
        import getpass
        # record['uid']/job['user'] are NERSC-side identity, meaningless compared
        # against this process's own os.getuid()/getpass.getuser() when self.remote
        # is set (this process is running on an entirely different machine). The
        # remote side already ran this exact check against its own identity while
        # building the record via _list_servers_remote()'s `status --json` call;
        # here, re-validate the *username* against the configured remote_user (the
        # intended NERSC identity) instead, and skip the numeric uid check, which
        # has no meaningful client-side equivalent to compare against.
        if record.get('schema_version') != 1 or (not self.remote and record.get('uid') != os.getuid()):
            raise ValueError('Unrecognized discovery record or wrong owner.')
        safe_name(record['id'])
        safe_name(record['name'])
        if not re.fullmatch(r'nid\d+', record['host']):
            raise ValueError('Expected a Perlmutter nid compute hostname.')
        if not isinstance(record['port'], int) or not 1024 <= record['port'] <= 65535:
            raise ValueError('Invalid server port.')
        job = self.job(record['job_id'])
        expected_user = (self.config.get('remote_user') or getpass.getuser()) if self.remote else getpass.getuser()
        if not job or job['user'] != expected_user or job['state'] != 'RUNNING':
            raise RuntimeError('Allocation is no longer running or is not owned by this user.')
        # One-node allocations render a single hostname, not a compressed hostlist.
        if job['nodes'].split('.')[0] != record['host']:
            raise RuntimeError('Discovery hostname does not match the allocation.')
        if record.get('state') != 'ready':
            raise RuntimeError('Server is not ready.')
        if time.time() - record.get('heartbeat', 0) > 90:
            raise RuntimeError('Server heartbeat has expired.')
        return job

    def dismiss_session(self, identity):
        """Persist a dashboard-only preference; do not contact Slurm."""
        safe_name(identity)
        atomic_json(self.runtime / 'dismissed-sessions' / (identity + '.json'), {'id': identity})

    def list_servers(self, *, dashboard=False):
        if self.remote:
            return self._list_servers_remote()
        result = []
        for path in sorted(self.records.glob('*.json')):
            if dashboard and (self.runtime / 'dismissed-sessions' / path.name).exists():
                continue
            try:
                if path.stat().st_uid != os.getuid() or path.stat().st_mode & 0o022:
                    raise ValueError('Discovery file is not privately owned/writable.')
                record = read_json(path)
                job = self.validate_record(record)
                result.append({**record, 'available': True, 'remaining': job['remaining']})
            except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
                result.append({'id': path.stem, 'name': path.stem, 'available': False, 'error': str(exc)})
        return result

    def _list_servers_remote(self):
        """Discovery over SSH: invoke `status --json` on the login node and
        parse its stdout, rather than reimplementing discovery remotely. The
        remote side has already run full validate_record() on every entry."""
        remote_cmd = self._remote_cli_argv('status', '--json')
        result = self._remote_run(remote_cmd, capture_output=True, text=True, timeout=25)
        try:
            records = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f'Unexpected output from remote status --json: {result.stdout[:200]!r}') from exc
        if not records and not (self.config.get('remote_config') and self.config.get('remote_python')):
            # A non-interactive `ssh host command` often doesn't see whatever
            # env var/module load selects your config interactively, so this
            # silently-empty result is more often a config mismatch than an
            # honest "no servers" -- print to stderr, not stdout, so it never
            # pollutes `status --json`'s machine-readable output.
            print(f'Note: {self.remote} reported zero servers. If you expected some, this SSH session may be '
                  "resolving a different config than your interactive shell does. Run `nersc-ollama doctor` "
                  "on the login node and set matching remote_config/remote_python in your local configuration.",
                  file=os.sys.stderr)
        return records

    def select_for_disconnect(self, identity):
        if not identity:
            return self.select()
        if self.remote:
            matches = [r for r in self.list_servers() if identity in (r['id'], r['name'])]
            if len(matches) != 1:
                raise RuntimeError('Specify one exact server ID for disconnect.')
            record = matches[0]
            safe_name(record['id'])
            # tunnel_paths()'s id-derived digest and a (possibly placeholder)
            # host string are all disconnect actually needs to locate/close
            # this client's own local control socket -- unlike ensure_tunnel(),
            # closing an expired server's tunnel deliberately doesn't require
            # the allocation to still be running, so list_servers()'s reduced
            # shape for an unavailable record (no host/port) is fine here.
            return {'id': record['id'], 'name': record.get('name', record['id']),
                    'host': record.get('host', record['id'])}
        matches = []
        for path in self.records.glob('*.json'):
            try:
                record = read_json(path)
                if identity in (record['id'], record['name']) and record['uid'] == os.getuid():
                    safe_name(record['id'])
                    if not re.fullmatch(r'nid\d+', record['host']):
                        continue
                    matches.append(record)
            except (OSError, ValueError, KeyError):
                continue
        if len(matches) != 1:
            raise RuntimeError('Specify one exact server ID for disconnect.')
        return matches[0]

    def select(self, identity=None):
        records = [r for r in self.list_servers() if r['available'] and
                   (identity is None or identity in (r['id'], r['name']))]
        if len(records) != 1:
            raise RuntimeError('Select exactly one live server with --server ID or NAME; use status to list servers.')
        return records[0]

    def dtn_command(self, model, check=False):
        self.check_model(model)
        self.require_installation()
        settings = self.config.get('downloads', {})
        host = settings.get('dtn_host', 'dtn01.nersc.gov')
        if not re.fullmatch(r'dtn0[1-4](?:\.nersc\.gov)?', host):
            raise ValueError('Choose an interactive NERSC DTN: dtn01 through dtn04.nersc.gov.')
        worker = Path(__file__).with_name('dtn_worker.py').absolute()
        remote = [settings.get('dtn_python', 'python3'), str(worker),
                  '--binary', str(self.binary), '--models', str(self.root / 'models'),
                  '--logs', str(self.logs), '--model', model]
        if check:
            remote.append('--check')
        return ['ssh', '-tt', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
                '-o', 'ServerAliveCountMax=3', host, shlex.join(remote)]

    def download(self, model, account=None, walltime=None, yes=False, dry_run=False, backend='dtn', check=False):
        self.check_model(model)
        self.require_installation()
        if backend == 'dtn':
            args = self.dtn_command(model, check)
            print('DTN transfer: no Slurm allocation; account and walltime do not apply.', flush=True)
            print(shlex.join(args), flush=True)
            if not dry_run:
                run(args)
            return
        if backend != 'cpu' or check:
            raise ValueError('Use backend dtn or cpu; --check is supported only for dtn.')
        if os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Run download from a login terminal, outside an existing allocation.')
        args = self.allocation_command('download', 'cpu', account, walltime)
        args += ['--download-model', model]
        print(shlex.join(args), flush=True)
        if dry_run:
            return
        confirm('Request this CPU allocation for the download? It ends when the download finishes.', yes)
        run(args, stream_errors=True)

    def serve(self, name, profile, download_model=None):
        import getpass
        safe_name(name)
        if download_model:
            self.check_model(download_model)
            if profile != 'cpu':
                raise ValueError('Download-only workers require the CPU profile.')
        job_id = os.environ.get('SLURM_JOB_ID', '')
        host = socket.gethostname().split('.')[0]
        job = self.job(job_id)
        if not job or job['state'] != 'RUNNING' or job['nodes'].split('.')[0] != host or job['user'] != getpass.getuser():
            raise RuntimeError('serve must run on the compute node inside your running allocation.')
        if not re.fullmatch(r'nid\d+', host):
            raise RuntimeError('Refusing to start inference on a login node.')
        if profile == 'gpu' and not os.environ.get('CUDA_VISIBLE_DEVICES'):
            raise RuntimeError('No Slurm GPU visibility; launch srun with --gpus.')
        startup_timeout = self.startup_timeout(profile)
        self.logs.mkdir(exist_ok=True, mode=0o700)
        identity = f'{name}-{job_id}-{uuid.uuid4().hex[:8]}'
        port = free_port()
        record = {'schema_version': 1, 'id': identity, 'name': name, 'uid': os.getuid(),
                  'job_id': job_id, 'host': host, 'port': port, 'profile': profile,
                  'context_length': self.config['profiles'][profile]['context'],
                  'created': time.time(), 'heartbeat': time.time(), 'state': 'starting'}
        path = self.records / (identity + '.json')
        env = os.environ.copy()
        env.update(OLLAMA_MODELS=str(self.root / 'models'), OLLAMA_HOST=f'127.0.0.1:{port}',
                   OLLAMA_CONTEXT_LENGTH=str(self.config['profiles'][profile]['context']))
        env['OLLAMA_SCHED_SPREAD'] = str(profile == 'gpu' and self.gpu_spread()).lower()
        if profile == 'cpu':
            env['CUDA_VISIBLE_DEVICES'] = '-1'
        stopped = False
        def stop(signum, frame):
            nonlocal stopped
            stopped = True
        old = {sig: signal.signal(sig, stop) for sig in (signal.SIGTERM, signal.SIGINT)}
        log_path = self.logs / (identity + '.log')
        print(f'Ollama log: {log_path} (startup timeout: {startup_timeout:g}s)', flush=True)
        try:
            with open(log_path, 'a') as log, stream_log(log_path):
                with subprocess.Popen([str(self.binary), 'serve'], env=env, stdout=log, stderr=subprocess.STDOUT) as proc:
                    record['pid'] = proc.pid
                    atomic_json(path, record)
                    try:
                        deadline = time.monotonic() + startup_timeout
                        while not stopped:
                            if proc.poll() is not None:
                                raise RuntimeError(f'Ollama exited ({proc.returncode}); inspect logs for {identity}.')
                            try:
                                request(port)
                                break
                            except (OSError, ValueError):
                                if time.monotonic() >= deadline:
                                    raise RuntimeError(f'Ollama readiness timed out after {startup_timeout:g}s; log: {log_path}')
                                time.sleep(.5)
                        if not stopped:
                            record['state'] = 'downloading' if download_model else 'ready'
                            print(f'Server {identity} ready on {host}:{port}', flush=True)
                        if download_model and not stopped:
                            atomic_json(path, record)
                            with subprocess.Popen([str(self.binary), 'pull', download_model], env=env) as pull:
                                try:
                                    while pull.poll() is None and not stopped:
                                        if proc.poll() is not None:
                                            raise RuntimeError('Ollama exited during download; inspect server logs.')
                                        time.sleep(.5)
                                    if not stopped and pull.returncode:
                                        raise RuntimeError(f'Model download failed (exit {pull.returncode}).')
                                finally:
                                    if pull.poll() is None:
                                        pull.terminate()
                                        try:
                                            pull.wait(timeout=10)
                                        except subprocess.TimeoutExpired:
                                            pull.kill()
                                            pull.wait()
                            if stopped:
                                raise RuntimeError('Download interrupted; no model data was deleted.')
                            print('Download complete; releasing CPU allocation.', flush=True)
                            return
                        while not stopped:
                            if proc.poll() is not None:
                                raise RuntimeError(f'Ollama exited ({proc.returncode}); inspect server logs.')
                            record['heartbeat'] = time.time()
                            atomic_json(path, record)
                            time.sleep(2)
                    finally:
                        if proc.poll() is None:
                            proc.terminate()
                            try:
                                proc.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                proc.kill()
                                proc.wait()
        finally:
            record.update(state='stopped', heartbeat=time.time())
            atomic_json(path, record)
            for sig, handler in old.items():
                signal.signal(sig, handler)

    def tunnel_paths(self, record):
        # /tmp is host-local; shared CFS is unsuitable for SSH control sockets.
        local = Path(tempfile.gettempdir()) / f'nersc-ollama-{os.getuid()}'
        local.mkdir(mode=0o700, exist_ok=True)
        if local.is_symlink() or local.stat().st_uid != os.getuid() or local.stat().st_mode & 0o077:
            raise RuntimeError('Tunnel directory must be private and owned by this user.')
        digest = hashlib.sha256((socket.gethostname() + record['id']).encode()).hexdigest()[:20]
        return local / (digest + '.sock'), local / (digest + '.json')

    def ensure_tunnel(self, record):
        self.validate_record(record)
        control, meta = self.tunnel_paths(record)
        with self.tunnel_lock(meta):
            if meta.exists():
                state = read_json(meta)
                check = subprocess.run(['ssh', '-S', str(control), '-O', 'check', record['host']],
                                       capture_output=True, timeout=10)
                if check.returncode == 0:
                    if state['server_id'] != record['id'] or state['remote_port'] != record['port']:
                        raise RuntimeError('Existing tunnel identity mismatch.')
                    try:
                        request(state['port'])
                        return state['port']
                    except (OSError, ValueError):
                        self._disconnect(record, control, meta)
            # A dead master can leave its socket behind. Remove only our private socket.
            control.unlink(missing_ok=True)
            port = free_port()
            args = ['ssh', '-M', '-S', str(control), '-fNT']
            if self.remote:
                # An additional hop through the login node: the tunnel still
                # opens directly to the compute node, just via a ProxyCommand
                # instead of assuming it's already reachable.
                args += self._remote_identity_args() + self._compute_proxy_option()
                # The login node's own host key (inside the ProxyCommand above)
                # is still verified normally. But an external client has no
                # legitimate way to have pre-trusted this specific compute
                # node's key -- nid* host keys aren't publicly distributed, and
                # a fresh client's known_hosts has never seen one. The compute
                # node is only ever reached via that already-authenticated
                # login-node hop, so trust it on first use here instead,
                # scoped to a private known_hosts file (not ~/.ssh/known_hosts,
                # since nid* hostnames get reused across different physical
                # nodes over time) so a genuine later key change still fails
                # loudly rather than being silently re-accepted every time.
                args += ['-o', 'StrictHostKeyChecking=accept-new',
                         '-o', f'UserKnownHostsFile={control.parent / "known_hosts"}']
            args += ['-o', 'BatchMode=yes', '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
                     '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                     '-L', f'127.0.0.1:{port}:127.0.0.1:{record["port"]}', record['host']]
            run(args, timeout=30)
            atomic_json(meta, {'server_id': record['id'], 'remote_port': record['port'], 'port': port})
            try:
                request(port)
            except Exception:
                self._disconnect(record, control, meta)
                raise
            return port

    def ssh_shell_command(self, record):
        """Argv for a plain interactive shell on a live server's compute node
        (for nersc-ollama-ssh2server). Not ssh_session.py's codex-local-enabling
        setup: that writes its rc file under self.runtime, which is only shared
        with the compute node in classic/on-NERSC mode -- under --remote,
        self.runtime is the client's own local directory, invisible from NERSC
        entirely. Reuses ensure_tunnel()'s identity/ProxyCommand/host-key
        handling instead, just without the -L port forward."""
        self.validate_record(record)
        control, _ = self.tunnel_paths(record)
        args = ['ssh', '-tt']
        if self.remote:
            args += self._remote_identity_args() + self._compute_proxy_option()
            args += ['-o', 'StrictHostKeyChecking=accept-new',
                     '-o', f'UserKnownHostsFile={control.parent / "known_hosts"}']
        args += ['-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15', '-o', 'ServerAliveCountMax=3',
                 record['host']]
        return args

    @contextlib.contextmanager
    def tunnel_lock(self, meta):
        import fcntl
        with open(str(meta) + '.lock', 'a') as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            yield

    def _disconnect(self, record, control, meta):
        subprocess.run(['ssh', '-S', str(control), '-O', 'exit', record['host']],
                       capture_output=True, timeout=10)
        meta.unlink(missing_ok=True)

    def disconnect(self, record):
        control, meta = self.tunnel_paths(record)
        with self.tunnel_lock(meta):
            if meta.exists():
                self._disconnect(record, control, meta)

    def downloaded_models(self):
        """Read completed local manifests and stat their blobs; never load model data."""
        storage = self.root / 'models'
        manifests = storage / 'manifests'
        result = []
        for path in sorted(manifests.glob('*/*/*/*')):
            if not path.is_file() or path.is_symlink():
                continue
            try:
                registry, namespace, model, tag = path.relative_to(manifests).parts
                manifest = read_json(path)
                descriptors = [manifest['config']] + manifest['layers']
                size = 0
                for descriptor in descriptors:
                    digest, length = descriptor['digest'], descriptor['size']
                    if not re.fullmatch(r'sha256:[0-9a-f]{64}', digest) or not isinstance(length, int) or length < 0:
                        raise ValueError('Invalid model manifest descriptor.')
                    blob = storage / 'blobs' / digest.replace(':', '-')
                    if not blob.is_file() or blob.stat().st_size != length:
                        raise ValueError('Download is incomplete.')
                    size += length
                prefix = '' if registry == 'registry.ollama.ai' else registry + '/'
                if namespace != 'library' or prefix:
                    prefix += namespace + '/'
                result.append({'name': prefix + model + ':' + tag, 'size': size})
            except (OSError, ValueError, KeyError, TypeError):
                # Concurrent/incomplete or malformed downloads are not installed models.
                continue
        return result

    def models(self, record):
        return request(self.ensure_tunnel(record), '/api/tags').get('models', [])

    def pull(self, record, model, *, local=False):
        self.require_installation()
        self.check_model(model)
        if local:
            if record['host'] != socket.gethostname().split('.')[0]:
                raise RuntimeError('Local server is not on this node.')
            self.validate_record(record)
        port = record['port'] if local else self.ensure_tunnel(record)
        # CLI streams progress and handles long downloads without an HTTP timeout.
        env = os.environ.copy()
        env['OLLAMA_HOST'] = f'127.0.0.1:{port}'
        run([self.binary, 'pull', model], env=env)

    @staticmethod
    def check_model(model):
        if not model or model.startswith('-') or not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./:-]*', model):
            raise ValueError('Invalid model tag.')
        if 'cloud' in model.lower():
            raise ValueError('Cloud models are outside this local-inference workflow.')

    def codex_catalog(self, record, model, info, *, context_override=None):
        """Supply explicit local-model metadata rather than Codex's hosted fallback.

        context_override lets a client request more than the server's own
        profile default (record['context_length'], fixed at `serve()` time) --
        Ollama's context length is only a default, not a hard ceiling: given a
        larger num_ctx per request, it reloads the model to fit, GPU memory
        permitting. Still capped by the model's own advertised maximum below;
        never persisted or sent to the server ahead of time."""
        profile = self.config['profiles'][record.get('profile', 'cpu')]
        context = context_override if context_override is not None else record.get('context_length', profile['context'])
        model_info = info.get('model_info', {})
        architecture = model_info.get('general.architecture', '')
        maximum = model_info.get(architecture + '.context_length')
        if isinstance(maximum, (int, float)) and maximum > 0:
            context = min(context, int(maximum))
        if isinstance(context, bool) or not isinstance(context, int) or context <= 0:
            raise ValueError('Model context length must be a positive integer.')
        modalities = ['text']
        if 'vision' in info.get('capabilities', []):
            modalities.append('image')
        entry = {
            'slug': model, 'display_name': model, 'context_window': context,
            'shell_type': 'default', 'visibility': 'list', 'supported_in_api': True,
            'priority': 0, 'truncation_policy': {'mode': 'bytes', 'limit': 10000},
            'input_modalities': modalities, 'base_instructions': '',
            'support_verbosity': True, 'default_verbosity': 'low',
            'supports_parallel_tool_calls': False, 'supports_reasoning_summaries': False,
            'supported_reasoning_levels': [], 'experimental_supported_tools': [],
        }
        catalog = {'models': [entry]}
        digest = hashlib.sha256(json.dumps(catalog, sort_keys=True).encode()).hexdigest()
        path = self.runtime / 'codex-models' / (digest + '.json')
        atomic_json(path, catalog, create_only=True)
        return path, context

    def codex_launch_info(self, record, model, *, local=False, context=None):
        """Resolve the endpoint and generated metadata used for a Codex launch."""
        self.check_model(model)
        if local:
            if record['host'] != socket.gethostname().split('.')[0]:
                raise RuntimeError('Local server is not on this node.')
            self.validate_record(record)
        port = record['port'] if local else self.ensure_tunnel(record)
        names = {m['name'] for m in request(port, '/api/tags').get('models', [])}
        canonical = model if ':' in model.rsplit('/', 1)[-1] else model + ':latest'
        if model not in names and canonical not in names:
            raise RuntimeError(f'{model} is not installed; explicitly run models pull first.')
        info = request(port, '/api/show', {'model': model})
        if 'tools' not in info.get('capabilities', []):
            raise RuntimeError('This model does not advertise tool support required for coding agents.')
        catalog, context = self.codex_catalog(record, model, info, context_override=context)
        return {'server': record.get('id', '—'), 'host': record.get('host', '—'), 'port': port,
                'model': model, 'context': context, 'catalog': catalog}

    def codex_command(self, record, model, extra=(), *, local=False, launch=None, context=None):
        codex = shutil.which('codex')
        if not codex:
            raise RuntimeError('codex is not on PATH; launch from an environment containing Codex.')
        launch = launch or self.codex_launch_info(record, model, local=local, context=context)
        port, catalog, context = launch['port'], launch['catalog'], launch['context']
        # Custom provider allows any selected local port without editing user config.
        args = [codex, '-m', model, '-c', 'model_provider="nersc_ollama"',
                '-c', 'model_providers.nersc_ollama.name="NERSC Ollama"',
                '-c', f'model_providers.nersc_ollama.base_url="http://127.0.0.1:{port}/v1"',
                '-c', 'model_providers.nersc_ollama.wire_api="responses"',
                '-c', 'model_providers.nersc_ollama.requires_openai_auth=false',
                '-c', 'model_catalog_json=' + json.dumps(str(catalog)),
                '-c', f'model_context_window={context}',
                '-c', f'model_auto_compact_token_limit={int(context * .9)}']
        return args + list(extra)

    def stop_allocation(self, record, yes=False):
        self.validate_record(record)
        confirm(f'Cancel your Slurm job {record["job_id"]} ({record["name"]})?', yes)
        runner = self._remote_run if self.remote else run
        runner(['scancel', record['job_id']])

    def tail_log(self, identity, lines):
        safe_name(identity)
        path = str(self.logs / (identity + '.log'))
        cmd = ['tail', '-n', str(lines), path]
        runner = self._remote_run if self.remote else run
        runner(cmd)

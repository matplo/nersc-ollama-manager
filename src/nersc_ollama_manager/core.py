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
        self.binary = self.root / 'current/bin/ollama'

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

    def require_installation(self):
        if not self.binary.is_file():
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

    def diagnostics(self):
        return {'config': str(self.config_path), 'python': os.sys.executable,
                'runtime': str(self.runtime), 'ollama_root': str(self.root),
                'ollama_installed': self.binary.is_file(),
                'commands': {name: shutil.which(name) for name in
                             ('codex', 'salloc', 'srun', 'squeue', 'scancel', 'ssh', 'curl', 'zstd')}}

    def install(self, version):
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

    def allocation_command(self, name, profile, account=None, walltime=None):
        safe_name(name)
        p = self.profile(profile, account, walltime)
        args = ['salloc', '--nodes', '1', '--qos', p['qos'], '--time', p['time'],
                '--constraint', p['constraint'], '--job-name', 'ollama-' + name]
        if p.get('account'):
            args += ['--account', p['account']]
        if p['gpus']:
            args += ['--gpus', str(p['gpus'])]
        args += ['srun', '--nodes', '1', '--ntasks', '1', '--unbuffered']
        if p['gpus']:
            args += ['--gpus', str(p['gpus'])]
        args += [os.sys.executable, '-m', 'nersc_ollama_manager', '--config', str(self.config_path),
                 'serve', '--name', name, '--profile', profile]
        return args

    def allocate(self, name, profile, account=None, walltime=None, yes=False, dry_run=False):
        if os.environ.get('SLURM_JOB_ID'):
            raise RuntimeError('Already inside an allocation; use serve rather than nesting salloc.')
        self.require_installation()
        args = self.allocation_command(name, profile, account, walltime)
        print(shlex.join(args), flush=True)
        if dry_run:
            return
        confirm('Request these Slurm resources?', yes)
        return run(args, stream_errors=True)

    def job(self, job_id):
        if not re.fullmatch(r'\d+', str(job_id)):
            raise ValueError('Invalid job ID.')
        result = run(['squeue', '--noheader', '--jobs', str(job_id), '--format', '%i|%u|%T|%N|%L'],
                     capture_output=True, text=True, timeout=15)
        for line in result.stdout.splitlines():
            fields = line.strip().split('|')
            if len(fields) == 5 and fields[0] == str(job_id):
                return dict(zip(('id', 'user', 'state', 'nodes', 'remaining'), fields))
        return None

    def validate_record(self, record):
        import getpass
        if record.get('schema_version') != 1 or record.get('uid') != os.getuid():
            raise ValueError('Unrecognized discovery record or wrong owner.')
        safe_name(record['id'])
        safe_name(record['name'])
        if not re.fullmatch(r'nid\d+', record['host']):
            raise ValueError('Expected a Perlmutter nid compute hostname.')
        if not isinstance(record['port'], int) or not 1024 <= record['port'] <= 65535:
            raise ValueError('Invalid server port.')
        job = self.job(record['job_id'])
        if not job or job['user'] != getpass.getuser() or job['state'] != 'RUNNING':
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

    def select_for_disconnect(self, identity):
        if not identity:
            return self.select()
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
            args = ['ssh', '-M', '-S', str(control), '-fNT', '-o', 'BatchMode=yes',
                    '-o', 'ExitOnForwardFailure=yes', '-o', 'ConnectTimeout=10',
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

    def pull(self, record, model):
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

    def codex_catalog(self, record, model, info):
        """Supply explicit local-model metadata rather than Codex's hosted fallback."""
        profile = self.config['profiles'][record.get('profile', 'cpu')]
        context = record.get('context_length', profile['context'])
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

    def codex_command(self, record, model, extra=(), *, local=False):
        codex = shutil.which('codex')
        if not codex:
            raise RuntimeError('codex is not on PATH; launch from an environment containing Codex.')
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
        catalog, context = self.codex_catalog(record, model, info)
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
        run(['scancel', record['job_id']])

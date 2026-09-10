"""Standalone standard-library worker; runs with the DTN's Python (3.6+).

Only downloads models. Never publishes discovery or offers inference commands.
"""
import argparse
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import tempfile
import time
import urllib.request


def stop_process(proc):
    if proc is not None and proc.poll() is None:
        proc.terminate()
        try:
            proc.wait(timeout=10)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--binary', required=True)
    parser.add_argument('--models', required=True)
    parser.add_argument('--logs', required=True)
    parser.add_argument('--model', required=True)
    parser.add_argument('--check', action='store_true')
    args = parser.parse_args(argv)
    if not re.fullmatch(r'dtn0[1-4](?:\.nersc\.gov)?', socket.gethostname()):
        raise RuntimeError('Download worker must run on a NERSC interactive DTN.')
    if not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_./:-]*', args.model) or 'cloud' in args.model.lower():
        raise ValueError('A local Ollama model tag is required.')
    binary = Path(args.binary)
    if not binary.is_file() or not os.access(str(binary), os.X_OK):
        raise RuntimeError('Ollama binary is not executable on the DTN.')
    for path in (args.models, args.logs):
        if not Path(path).is_absolute():
            raise ValueError('Shared storage paths must be absolute.')
        Path(path).mkdir(parents=True, exist_ok=True, mode=0o700)
    # DTN /tmp must not hold transfer data. Keep all temporary I/O on shared storage.
    scratch = Path(args.models).parent / 'transfer-tmp'
    scratch.mkdir(exist_ok=True, mode=0o700)
    with socket.socket() as sock:
        sock.bind(('127.0.0.1', 0))
        port = sock.getsockname()[1]
    env = os.environ.copy()
    env.update(OLLAMA_HOST='127.0.0.1:{}'.format(port), OLLAMA_MODELS=args.models,
               CUDA_VISIBLE_DEVICES='-1', ROCR_VISIBLE_DEVICES='-1', GOMAXPROCS='2')
    interrupted = [False]
    def stop(signum, frame):
        interrupted[0] = True
    signals = (signal.SIGINT, signal.SIGTERM, signal.SIGHUP)
    old = {sig: signal.signal(sig, stop) for sig in signals}
    server = pull = None
    try:
        with tempfile.TemporaryDirectory(prefix='pull-', dir=str(scratch)) as tmp:
            env['TMPDIR'] = tmp
            fd, logfile = tempfile.mkstemp(prefix='dtn-download-', suffix='.log', dir=args.logs)
            print('DTN download log: {}'.format(logfile), flush=True)
            with os.fdopen(fd, 'w') as log:
                server = subprocess.Popen([str(binary), 'serve'], env=env, stdout=log, stderr=subprocess.STDOUT)
                opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
                deadline = time.monotonic() + 60
                while not interrupted[0]:
                    if server.poll() is not None:
                        raise RuntimeError('Ollama failed to start; inspect {}'.format(logfile))
                    try:
                        with opener.open('http://127.0.0.1:{}/api/version'.format(port), timeout=2) as response:
                            version = json.load(response)['version']
                        break
                    except (OSError, ValueError):
                        if time.monotonic() >= deadline:
                            raise RuntimeError('DTN Ollama readiness timed out; inspect {}'.format(logfile))
                        time.sleep(.25)
                if interrupted[0]:
                    return 130
                print('Ollama {} ready for transfer on {}'.format(version, socket.gethostname()), flush=True)
                if args.check:
                    print('DTN preflight passed; no model downloaded.', flush=True)
                    return 0
                pull = subprocess.Popen([str(binary), 'pull', args.model], env=env)
                while pull.poll() is None and not interrupted[0]:
                    if server.poll() is not None:
                        raise RuntimeError('Ollama stopped during download; inspect {}'.format(logfile))
                    time.sleep(.5)
                if interrupted[0]:
                    return 130
                if pull.returncode:
                    raise RuntimeError('Model pull failed with exit {}'.format(pull.returncode))
                print('Download complete. Model files are ready for compute-node use.', flush=True)
                return 0
    finally:
        stop_process(pull)
        stop_process(server)
        for sig, handler in old.items():
            signal.signal(sig, handler)


if __name__ == '__main__':
    try:
        raise SystemExit(main())
    except (OSError, RuntimeError, ValueError) as exc:
        print('Error: {}'.format(exc), flush=True)
        raise SystemExit(1)

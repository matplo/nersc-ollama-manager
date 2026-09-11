"""Prepare a private, uniquely named Bash setup file for a node session."""
import os
import json
import shlex
import sys
import tempfile
from pathlib import Path


def session_rc(manager, record, model=None):
    config = shlex.quote(str(manager.config_path))
    identity = shlex.quote(record['id'])
    host = shlex.quote(record['host'])
    model_value = shlex.quote(model if isinstance(model, str) else '')
    python = shlex.quote(sys.executable)
    binding = shlex.quote(json.dumps({'uid': os.getuid(), 'host': record['host'],
                                      'server': record['id'], 'config': str(manager.config_path)}))
    return '\n'.join([
        '# Generated SSH session setup. Source only on the selected compute node.',
        'if [[ -f ~/.bashrc ]]; then source ~/.bashrc; fi',
        f'if [[ "$(hostname -s)" != {host} ]]; then',
        "  printf '%s\\n' 'Wrong node for this setup file.' >&2; return 1",
        'fi',
        f'export NERSC_OLLAMA_CONFIG={config}',
        f'export NERSC_OLLAMA_SERVER={identity}',
        f'export NERSC_OLLAMA_MODEL={model_value}',
        # /tmp is explicitly node-local; do not inherit a shared TMPDIR.
        'NERSC_CODEX_SESSION_DIR=$(mktemp -d /tmp/nersc-codex.XXXXXXXXXX) || return 1',
        'mkdir -m 700 "$NERSC_CODEX_SESSION_DIR/sqlite" "$NERSC_CODEX_SESSION_DIR/tmp" || return 1',
        "(umask 077; printf '%s\\n' " + binding + ' > "$NERSC_CODEX_SESSION_DIR/session.json") || return 1',
        'export NERSC_CODEX_SESSION_DIR',
        'function codex-local() {',
        f'  NERSC_OLLAMA_CONFIG={config} NERSC_OLLAMA_SERVER={identity} '
        'CODEX_HOME="$NERSC_CODEX_SESSION_DIR" '
        'CODEX_SQLITE_HOME="$NERSC_CODEX_SESSION_DIR/sqlite" '
        'TMPDIR="$NERSC_CODEX_SESSION_DIR/tmp" '
        f'{python} -m nersc_ollama_manager.local "$@"',
        '}',
        "printf 'codex-local ready; private node-local state: %s\\n' \"$NERSC_CODEX_SESSION_DIR\"",
        "printf '%s\\n' 'History is local to this shell session and may be lost when the allocation ends.'",
        '',
    ])


def ssh_command(manager, record, model=None):
    directory = Path(manager.runtime) / 'ssh-sessions'
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    fd, filename = tempfile.mkstemp(prefix='session-', suffix='.bash', dir=directory)
    with os.fdopen(fd, 'w') as stream:
        stream.write(session_rc(manager, record, model))
    remote = shlex.join(['bash', '--rcfile', filename, '-i'])
    return ['ssh', '-tt', '-o', 'ConnectTimeout=10', '-o', 'ServerAliveInterval=15',
            '-o', 'ServerAliveCountMax=3', record['host'], remote]

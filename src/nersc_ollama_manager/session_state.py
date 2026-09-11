"""Validate exported SSH-session state before starting a Codex subprocess."""
import json
import os
from pathlib import Path
import socket
import stat


def private_path(path, directory=False):
    info = path.lstat()
    expected = stat.S_ISDIR if directory else stat.S_ISREG
    if not expected(info.st_mode) or info.st_uid != os.getuid() or info.st_mode & 0o077:
        raise RuntimeError('Session state must be privately owned: ' + str(path))


def apply_session_state():
    value = os.environ.get('NERSC_CODEX_SESSION_DIR')
    if value is None:
        if os.environ.get('NERSC_OLLAMA_SERVER'):
            raise RuntimeError('Missing SSH session state. Reconnect through the TUI SSH action.')
        return
    path = Path(value)
    if not value or path.parent != Path('/tmp') or not path.name.startswith('nersc-codex.'):
        raise RuntimeError('Invalid node-local session directory. Reconnect through the TUI.')
    try:
        private_path(path, directory=True)
        private_path(path / 'sqlite', directory=True)
        private_path(path / 'tmp', directory=True)
        marker = path / 'session.json'
        private_path(marker)
        if marker.stat().st_size > 16384:
            raise RuntimeError('Invalid session binding size.')
        binding = json.loads(marker.read_text())
        expected = {'uid': os.getuid(), 'host': socket.gethostname().split('.')[0],
                    'server': os.environ.get('NERSC_OLLAMA_SERVER'),
                    'config': os.environ.get('NERSC_OLLAMA_CONFIG')}
        if binding != expected:
            raise RuntimeError('Session state belongs to a different node, server, or configuration.')
    except (OSError, ValueError) as exc:
        raise RuntimeError('Cannot validate SSH session state; reconnect through the TUI.') from exc
    os.environ.update(CODEX_HOME=str(path), CODEX_SQLITE_HOME=str(path / 'sqlite'),
                      TMPDIR=str(path / 'tmp'))

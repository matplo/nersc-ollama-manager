"""nersc-ollama-ssh2server: pick a live server interactively, then SSH to its
compute node with a plain interactive shell.
"""
import argparse
import os
import sys

from rich.console import Console

from .core import Manager
from ._prompts import pick_server
from . import __version__


def parser():
    p = argparse.ArgumentParser(description='Pick a live server interactively, then SSH to its compute node')
    p.add_argument('--config', help='Configuration path (overrides NERSC_OLLAMA_CONFIG and the per-user default)')
    p.add_argument('--remote', action='store_true',
                    help='Run against a NERSC login node over SSH (same as nersc-ollama --remote)')
    p.add_argument('--remote-host', help='Override remote_login_host from config for this invocation')
    p.add_argument('--server', help='Skip the server prompt: an exact server ID or name')
    p.add_argument('--version', action='version', version=__version__)
    return p


def main(argv=None):
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)
    console = Console()
    try:
        manager = Manager.open(args.config)
        manager.resolve_remote(args.remote, args.remote_host)
        record = pick_server(manager, console, args.server)
        command = manager.ssh_shell_command(record)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, RuntimeError) as exc:
        console.print(f'Error: {exc}', style='red', markup=False)
        return 1
    console.print(f"Connecting to {record['host']}...")
    try:
        os.execvp(command[0], command)  # replaces this process; returns the shell's own exit status
    except OSError as exc:
        console.print(f'Error: {exc}', style='red', markup=False)
        return 1


if __name__ == '__main__':
    sys.exit(main())

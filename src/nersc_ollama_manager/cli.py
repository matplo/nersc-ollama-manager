"""Command-line entry point."""
import argparse
import json
import os
from pathlib import Path
import subprocess
import sys

from rich.console import Console
from rich.table import Table

from .core import Manager, config_location, run, safe_name
from . import __version__


def parser():
    p = argparse.ArgumentParser(description='Ollama in personal NERSC interactive allocations')
    p.add_argument('--config', help='Configuration path (overrides NERSC_OLLAMA_CONFIG and the per-user default)')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('setup', help='Initialize configuration; optionally install a pinned Ollama release')
    setup.add_argument('--runtime', help='Runtime storage for a new configuration')
    setup.add_argument('--root')
    setup.add_argument('--version', help='Exact Ollama release to download; omit for configuration only')
    allocate = sub.add_parser('allocate', help='Request and hold an interactive allocation')
    allocate.add_argument('--profile', choices=['cpu', 'gpu'], default='cpu')
    allocate.add_argument('--name', default='default')
    allocate.add_argument('--account')
    allocate.add_argument('--time')
    allocate.add_argument('--yes', action='store_true', help='Explicitly approve the displayed resource request')
    allocate.add_argument('--dry-run', action='store_true')
    serve = sub.add_parser('serve', help='Internal worker, run via srun on a compute node')
    serve.add_argument('--download-model', help=argparse.SUPPRESS)
    serve.add_argument('--name', default='default')
    serve.add_argument('--profile', choices=['cpu', 'gpu'], default='cpu')
    status = sub.add_parser('status')
    status.add_argument('--json', action='store_true')
    logs = sub.add_parser('logs')
    logs.add_argument('id', help='Server ID shown by status, including stopped servers')
    logs.add_argument('--lines', type=int, default=50)
    for name in ['tunnel', 'disconnect', 'stop', 'codex']:
        cmd = sub.add_parser(name)
        cmd.add_argument('--server')
        if name == 'codex':
            cmd.add_argument('--model', required=True)
            cmd.add_argument('codex_args', nargs=argparse.REMAINDER, help='Arguments after -- go to Codex')
        if name == 'stop':
            cmd.add_argument('--yes', action='store_true')
    models = sub.add_parser('models')
    models.add_argument('--server')
    modsub = models.add_subparsers(dest='model_action', required=True)
    modsub.add_parser('list')
    pull = modsub.add_parser('pull')
    pull.add_argument('model')
    download = sub.add_parser('download', help='Download on a DTN (default) or in a temporary CPU allocation')
    download.add_argument('model')
    download.add_argument('--backend', choices=['dtn', 'cpu'], default='dtn')
    download.add_argument('--check', action='store_true', help='Check DTN server startup without downloading a model')
    download.add_argument('--account')
    download.add_argument('--time')
    download.add_argument('--yes', action='store_true')
    download.add_argument('--dry-run', action='store_true')
    sub.add_parser('tui')
    sub.add_parser('doctor', help='Show environment, paths, and external command availability')
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    console = Console()
    try:
        if args.command == 'setup':
            path, _ = config_location(args.config)
            manager = Manager.initialize(path, args.runtime, args.root)
            if path.exists() and (args.runtime or args.root):
                wanted_runtime = Path(args.runtime).expanduser().absolute() if args.runtime else manager.runtime
                wanted_root = Path(args.root).expanduser().absolute() if args.root else manager.root
                if (wanted_runtime, wanted_root) != (manager.runtime, manager.root):
                    manager.set_storage(wanted_runtime, wanted_root)
            if args.version:
                print(manager.install(args.version))
            print(f'Configuration: {manager.config_path}')
            return 0
        manager = Manager.open(args.config)
        if args.command == 'doctor':
            print(json.dumps(manager.diagnostics(), indent=2))
        elif args.command == 'allocate':
            manager.allocate(args.name, args.profile, args.account, args.time, args.yes, args.dry_run)
        elif args.command == 'download':
            manager.download(args.model, args.account, args.time, args.yes, args.dry_run, args.backend, args.check)
        elif args.command == 'serve':
            manager.serve(args.name, args.profile, args.download_model)
        elif args.command == 'status':
            records = manager.list_servers()
            if args.json:
                print(json.dumps(records, indent=2))
            else:
                table = Table('Server', 'Profile', 'Node', 'Remaining', 'State')
                for r in records:
                    table.add_row(r['id'], r.get('profile', '—'), r.get('host', '—'), r.get('remaining', '—'),
                                  'ready' if r['available'] else r.get('error', 'unavailable'))
                console.print(table)
        elif args.command == 'logs':
            safe_name(args.id)
            if args.lines < 1:
                raise ValueError('--lines must be positive.')
            run(['tail', '-n', str(args.lines), str(manager.logs / (args.id + '.log'))])
        elif args.command == 'tui':
            try:
                from .tui import Dashboard
            except ImportError as exc:
                raise RuntimeError('Install the TUI extra: python -m pip install "nersc-ollama-manager[tui]"') from exc
            Dashboard(manager).run()
        elif args.command == 'models' and args.model_action == 'list' and not args.server:
            for model in manager.downloaded_models():
                print(f'{model["name"]}\t{model["size"]:,} bytes')
        else:
            record = (manager.select_for_disconnect(args.server) if args.command == 'disconnect'
                      else manager.select(args.server))
            if args.command == 'tunnel':
                print(f'http://127.0.0.1:{manager.ensure_tunnel(record)}')
            elif args.command == 'disconnect':
                manager.disconnect(record)
            elif args.command == 'stop':
                manager.stop_allocation(record, args.yes)
            elif args.command == 'models':
                if args.model_action == 'pull':
                    manager.pull(record, args.model)
                else:
                    for model in manager.models(record):
                        print(f'{model["name"]}\t{model.get("size", 0):,} bytes')
            elif args.command == 'codex':
                extra = args.codex_args
                if extra[:1] == ['--']:
                    extra = extra[1:]
                return subprocess.call(manager.codex_command(record, args.model, extra))
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f'Error: {exc}', style='red', markup=False)
        return 1


def tui_main(argv=None):
    """Console shortcut accepting the same global options as nersc-ollama."""
    return main([*(sys.argv[1:] if argv is None else argv), 'tui'])


if __name__ == '__main__':
    sys.exit(main())

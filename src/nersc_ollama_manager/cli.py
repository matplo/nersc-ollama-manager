"""Command-line entry point."""
import argparse
import contextlib
import io
import json
import os
from pathlib import Path
import subprocess
import sys

from rich.console import Console
from rich.table import Table

from .core import Manager, config_location
from . import __version__


def parser():
    p = argparse.ArgumentParser(description='Ollama in personal NERSC interactive allocations')
    p.add_argument('--config', help='Configuration path (overrides NERSC_OLLAMA_CONFIG and the per-user default)')
    p.add_argument('--remote', action='store_true',
                    help='Run against a NERSC login node over SSH instead of assuming local NERSC execution. '
                         'Uses remote_login_host from config unless --remote-host overrides it. '
                         '(Two flags, not one optional-valued flag, so a bare --remote never swallows '
                         'the required subcommand that follows it.)')
    p.add_argument('--remote-host', help='Override remote_login_host from config for this invocation; requires --remote.')
    p.add_argument('--version', action='version', version=__version__)
    sub = p.add_subparsers(dest='command', required=True)
    setup = sub.add_parser('setup', help='Initialize configuration; optionally install a pinned Ollama release')
    setup.add_argument('--runtime', help='Runtime storage for a new configuration')
    setup.add_argument('--root')
    setup.add_argument('--ollama-binary', help='Use an existing Ollama binary (e.g. a module or shared install) '
                                                'instead of a pinned release managed by this tool')
    setup.add_argument('--version', help='Exact Ollama release to download; omit for configuration only')
    allocate = sub.add_parser('allocate', help='Request and hold an interactive allocation')
    allocate.add_argument('--profile', choices=['cpu', 'gpu'], default='cpu')
    allocate.add_argument('--name', default='default')
    allocate.add_argument('--account')
    allocate.add_argument('--time')
    allocate.add_argument('--yes', action='store_true', help='Explicitly approve the displayed resource request')
    allocate.add_argument('--dry-run', action='store_true')
    allocate.add_argument('--screen', action='store_true',
                           help='Run detached in a screen session on the login node, immune to a dropped SSH connection')
    allocate.add_argument('--gpu-spread', dest='gpu_spread', action='store_true', default=None,
                           help='Persist "spread across GPUs" for future --profile gpu allocations '
                                '(same setting as the TUI switch); running jobs are unchanged')
    allocate.add_argument('--no-gpu-spread', dest='gpu_spread', action='store_false',
                           help='Persist GPU spreading off for future --profile gpu allocations')
    peek = sub.add_parser('peek', help='Snapshot a screen-detached allocation\'s visible output')
    peek.add_argument('name', help='The --name given to allocate --screen')
    sub.add_parser('sessions', help='List ollama-* screen sessions on the login node')
    forget = sub.add_parser('forget', help="Remove an expired discovery record, once Slurm confirms it's gone")
    forget.add_argument('id', nargs='?', help='Server ID shown by status')
    forget.add_argument('--all', action='store_true', help='Remove every currently-unavailable record instead of one ID')
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
            cmd.add_argument('--context', type=int,
                              help='Override the advertised context window (tokens), still capped by the '
                                   "model's own maximum. Does not change the running server's own allocation -- "
                                   'Ollama reloads the model to fit on demand, GPU memory permitting.')
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


NERSC_ONLY_COMMANDS = {'serve', 'download'}


def main(argv=None):
    args = parser().parse_args(argv)
    console = Console()
    try:
        if args.remote and args.command in NERSC_ONLY_COMMANDS:
            raise RuntimeError(f'--remote cannot be combined with {args.command!r}; run it directly on NERSC.')
        if args.remote and args.command == 'allocate' and not args.screen:
            raise RuntimeError('--remote allocate requires --screen '
                                '(a foreground salloc cannot run over one SSH round-trip).')
        if args.command == 'setup' and not args.remote:
            path, _ = config_location(args.config)
            manager = Manager.initialize(path, args.runtime, args.root)
            if path.exists() and (args.runtime or args.root):
                wanted_runtime = Path(args.runtime).expanduser().absolute() if args.runtime else manager.runtime
                wanted_root = Path(args.root).expanduser().absolute() if args.root else manager.root
                if (wanted_runtime, wanted_root) != (manager.runtime, manager.root):
                    manager.set_storage(wanted_runtime, wanted_root)
            if args.ollama_binary:
                manager.set_ollama_binary(args.ollama_binary)
            if args.version:
                print(manager.install(args.version))
            print(f'Configuration: {manager.config_path}')
            return 0
        manager = Manager.open(args.config)
        if args.remote:
            manager.remote = args.remote_host or manager.config.get('remote_login_host')
            if not manager.remote:
                raise RuntimeError('Pass --remote-host HOST, or set remote_login_host in the configuration.')
        elif args.remote_host:
            raise RuntimeError('--remote-host requires --remote.')
        if args.command == 'setup':  # only reachable here when args.remote is set
            manager.setup_remote(args.runtime, args.root, args.version, args.ollama_binary)
        elif args.command == 'doctor':
            print(json.dumps(manager.diagnostics(), indent=2))
        elif args.command == 'allocate':
            manager.allocate(args.name, args.profile, args.account, args.time, args.yes, args.dry_run, args.screen,
                              args.gpu_spread)
        elif args.command == 'peek':
            print(manager.capture_screen('ollama-' + args.name))
        elif args.command == 'sessions':
            names = manager.list_screen_sessions()
            print('\n'.join(names) if names else 'No ollama-* screen sessions found.')
        elif args.command == 'forget':
            if bool(args.id) == bool(args.all):
                raise ValueError('Give exactly one of an ID or --all.')
            manager.forget(args.id, args.all)
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
            if args.lines < 1:
                raise ValueError('--lines must be positive.')
            manager.tail_log(args.id, args.lines)
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
                return subprocess.call(manager.codex_command(record, args.model, extra, context=args.context))
        return 0
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f'Error: {exc}', style='red', markup=False)
        return 1


def tui_main(argv=None):
    """Console shortcut for `nersc-ollama tui`, accepting the same global options
    (--config, --remote, --remote-host). It always appends 'tui' itself -- passing
    a subcommand of your own (e.g. `nersc-ollama-tui --remote ... sessions`) would
    silently conflict with that, so detect it first and point at the fix instead
    of argparse's confusing "unrecognized arguments: tui"."""
    args = sys.argv[1:] if argv is None else list(argv)
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        try:
            given = parser().parse_args(args)
        except SystemExit:
            given = None  # no subcommand given yet (or --help/--version) -- the expected shortcut usage
    if given is not None:
        Console().print(f"Error: nersc-ollama-tui always launches the TUI; drop {given.command!r} and "
                         f"run nersc-ollama directly instead: nersc-ollama {' '.join(args)}",
                         style='red', markup=False)
        return 1
    return main([*args, 'tui'])


if __name__ == '__main__':
    sys.exit(main())

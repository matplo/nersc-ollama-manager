"""nersc-ollama-codex: pick a live server/model/context interactively, then
launch Codex against it.

A thin wizard on top of the existing status/models/codex primitives -- no new
discovery or launch logic here, just fewer keystrokes for the common case of
picking from what's already there. Every step is skippable via a flag, so it
degrades to a fully non-interactive one-liner for scripted use.
"""
import argparse
import subprocess
import sys

from rich.console import Console

from .core import Manager, request
from ._prompts import pick, pick_server
from . import __version__

APPROVAL_PRESETS = [
    ('Full auto (sandboxed, auto-approve)', ['--full-auto']),
    ('Default (Codex asks before acting)', []),
    ('Bypass approvals and sandbox entirely', ['--dangerously-bypass-approvals-and-sandbox']),
]


def parser():
    p = argparse.ArgumentParser(description='Pick a live server/model/context interactively, then launch Codex')
    p.add_argument('--config', help='Configuration path (overrides NERSC_OLLAMA_CONFIG and the per-user default)')
    p.add_argument('--remote', action='store_true',
                    help='Run against a NERSC login node over SSH (same as nersc-ollama --remote)')
    p.add_argument('--remote-host', help='Override remote_login_host from config for this invocation')
    p.add_argument('--server', help='Skip the server prompt: an exact server ID or name')
    p.add_argument('--model', help='Skip the model prompt: an exact, already-installed model tag')
    p.add_argument('--context', type=int, help="Skip the context prompt: an exact token count (still capped by the model's own maximum)")
    p.add_argument('--version', action='version', version=__version__)
    p.add_argument('codex_args', nargs=argparse.REMAINDER,
                    help='Arguments after -- go to Codex, skipping the approval-mode prompt')
    return p


def main(argv=None):
    args = parser().parse_args(sys.argv[1:] if argv is None else argv)
    console = Console()
    try:
        manager = Manager.open(args.config)
        manager.resolve_remote(args.remote, args.remote_host)

        record = pick_server(manager, console, args.server)

        names = [m['name'] for m in manager.models(record)]
        if not names:
            raise RuntimeError(f"No models installed on {record['id']}; run "
                                f"`nersc-ollama models --server {record['id']} pull TAG` first.")
        if args.model:
            if args.model not in names:
                raise RuntimeError(f'{args.model!r} is not installed on {record["id"]}. Installed: {", ".join(names)}')
            model = args.model
        else:
            preferred = manager.config.get('codex_model')
            default_index = next((i for i, n in enumerate(names) if n == preferred), None)
            if default_index is None:
                default_index = next((i for i, n in enumerate(names) if 'qwen3.8' in n), 0)
            model = pick(console, names, 'Model', lambda n: n, default_index=default_index)

        context = args.context
        if context is None:
            info = request(manager.ensure_tunnel(record), '/api/show', {'model': model})
            model_info = info.get('model_info', {})
            maximum = model_info.get(model_info.get('general.architecture', '') + '.context_length')
            if isinstance(maximum, (int, float)) and maximum > 0:
                context = int(maximum)
                console.print(f"Context: {context:,} tokens (model's own maximum)")

        extra = args.codex_args
        if extra[:1] == ['--']:
            extra = extra[1:]
        if not extra:
            _, extra = pick(console, APPROVAL_PRESETS, 'Approval mode', lambda preset: preset[0])

        command = manager.codex_command(record, model, extra, context=context)
        console.print('Launching: ' + ' '.join(command))
        return subprocess.call(command)
    except KeyboardInterrupt:
        return 130
    except (OSError, ValueError, KeyError, RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f'Error: {exc}', style='red', markup=False)
        return 1


if __name__ == '__main__':
    sys.exit(main())

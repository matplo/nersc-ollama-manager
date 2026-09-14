"""nersc-ollama-codex: pick a live server/model/context interactively, then
launch Codex against it.

A thin wizard on top of the existing status/models/codex primitives -- no new
discovery or launch logic here, just fewer keystrokes for the common case of
picking from what's already there. Every step is skippable via a flag, so it
degrades to a fully non-interactive one-liner for scripted use.
"""
import argparse
import shlex
import subprocess
import sys

from rich.console import Console
from rich.prompt import Confirm, Prompt

from .core import Manager, request
from ._prompts import pick, pick_server
from . import __version__

# Real, verified values straight from `codex --help` -- an earlier revision
# of this file guessed at a bundled `--full-auto` flag that turned out not to
# exist in current Codex CLI at all. No attempt here to guess which
# combination of these "honestly works" together; each is independent and
# defaults to unset (Codex's own config decides) so nothing is ever pinned
# unless explicitly chosen.
UNSET = "(unset -- use Codex's own default)"
SANDBOX_MODES = [UNSET, 'read-only', 'workspace-write', 'danger-full-access']
APPROVAL_POLICIES = [UNSET, 'on-request', 'never']


def _configure_approval(console):
    # --approve-for-me and --sandbox are mutually exclusive -- confirmed by
    # Codex's own parser ("the argument '--sandbox <SANDBOX_MODE>' cannot be
    # used with '--approve-for-me'"), not assumed. --approve-for-me's own
    # --help text ("...using the workspace-write sandbox") suggests it's a
    # self-contained shorthand rather than a modifier meant to stack with
    # manual -s/-a choices, so asking it first and returning immediately
    # avoids ever reproducing that confirmed-broken combination.
    if Confirm.ask('Auto-approve for me (--approve-for-me)? Uses its own sandbox internally; '
                    "skips the manual -s/-a choices below if yes.", default=False):
        return ['--approve-for-me']
    extra = []
    sandbox = pick(console, SANDBOX_MODES, 'Sandbox (-s)', lambda s: s, default_index=0)
    if sandbox != UNSET:
        extra += ['--sandbox', sandbox]
    policy = pick(console, APPROVAL_POLICIES, 'Ask for approval (-a)', lambda s: s, default_index=0)
    if policy != UNSET:
        extra += ['--ask-for-approval', policy]
    return extra


def _custom_args(console):
    return shlex.split(Prompt.ask('Codex arguments'))


# (label, resolver(console) -> extra args). 'Codex's own defaults' is first/
# the default on purpose: no flags at all, so it can never be wrong.
APPROVAL_CHOICES = [
    ("Codex's own defaults (no extra flags)", lambda console: []),
    ('Configure approval (-s / -a / --approve-for-me, independently)', _configure_approval),
    ('Bypass approvals and sandbox entirely (dangerous)', lambda console: ['--dangerously-bypass-approvals-and-sandbox']),
    ('Custom (type your own Codex arguments)', _custom_args),
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
            _, resolve = pick(console, APPROVAL_CHOICES, 'Approval mode', lambda choice: choice[0])
            extra = resolve(console)

        # Best-effort refresh: the approval-mode menu (and any Confirm/Prompt
        # follow-ups) is exactly the kind of open-ended wait that can push the
        # original snapshot's heartbeat past validate_record()'s 90s window,
        # well after the server itself is still fine -- hit live. But a fresh
        # select() is itself one more remote round-trip that can transiently
        # hiccup (hit live too, immediately after this was first added), so a
        # failure here must never be fatal on its own: fall back to the
        # record already in hand rather than a refresh attempt becoming a new
        # single point of failure. If it's genuinely gone, codex_command()'s
        # own validate_record() call below still reports that clearly.
        try:
            record = manager.select(record['id'])
        except (RuntimeError, KeyError, ValueError, OSError):
            pass
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

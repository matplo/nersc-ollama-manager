"""Shared interactive-picker helpers for nersc-ollama-codex / nersc-ollama-ssh2server."""
from rich.prompt import IntPrompt


def pick(console, items, label, format_row, default_index=0):
    """Number `items` and return the chosen one. Auto-picks (announcing it)
    when there's only one; Enter accepts default_index otherwise."""
    if len(items) == 1:
        console.print(f'{label}: {format_row(items[0])} (only one available)')
        return items[0]
    for i, item in enumerate(items, 1):
        suffix = ' (default)' if i - 1 == default_index else ''
        console.print(f'  {i}. {format_row(item)}{suffix}')
    choice = IntPrompt.ask(label, default=default_index + 1,
                            choices=[str(i) for i in range(1, len(items) + 1)], show_choices=False)
    return items[choice - 1]


def pick_server(manager, console, requested=None):
    """Resolve --server (exact ID/name match) or prompt among live servers."""
    records = [r for r in manager.list_servers() if r['available']]
    if not records:
        raise RuntimeError('No live servers; run allocate first (see `nersc-ollama status`).')
    if requested:
        record = next((r for r in records if requested in (r['id'], r['name'])), None)
        if record is None:
            raise RuntimeError(f'{requested!r} is not a live server; run `nersc-ollama status` to list them.')
        return record
    return pick(console, records, 'Server', lambda r:
                f"{r['id']} ({r.get('profile', '—')}, {r.get('host', '—')}, {r.get('remaining', '—')} left)")

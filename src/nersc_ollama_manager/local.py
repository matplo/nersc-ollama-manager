"""Resolve this node's Ollama endpoint and preserve Codex arguments."""
import os
import shutil
import socket
import sys
from .core import Manager, request


def model_argument(args):
    model = None
    for i, arg in enumerate(args):
        if arg == '--':
            break
        if arg in ('-m', '--model'):
            if i + 1 == len(args):
                raise ValueError('Missing model after ' + arg)
            model = args[i + 1]
        elif arg.startswith('--model='):
            model = arg.split('=', 1)[1]
        elif arg.startswith('-m') and len(arg) > 2:
            model = arg[2:]
    return model


def command(args):
    codex = shutil.which('codex')
    if not codex:
        raise RuntimeError('codex is not on PATH.')
    if any(a in ('--help', '-h', '--version', '-V') for a in args):
        return [codex, *args]
    manager = Manager.open(None)
    host = socket.gethostname().split('.')[0]
    candidates = []
    from .core import read_json
    for path in manager.records.glob('*.json'):
        try:
            record = read_json(path)
            if record.get('host') != host or record.get('uid') != os.getuid():
                continue
            identity = os.environ.get('NERSC_OLLAMA_SERVER')
            if identity and record.get('id') != identity:
                continue
            manager.validate_record(record)
            candidates.append(record)
        except (OSError, ValueError, KeyError, RuntimeError):
            continue
    if len(candidates) != 1:
        raise RuntimeError('Expected one live local server; found %d. Use NERSC_OLLAMA_SERVER to select its exact ID.' % len(candidates))
    record = candidates[0]
    model = model_argument(args) or os.environ.get('NERSC_OLLAMA_MODEL') or manager.config.get('codex_model')
    if not model:
        models = request(record['port'], '/api/tags').get('models', [])
        if len(models) != 1:
            raise RuntimeError('Choose a model with -m MODEL or NERSC_OLLAMA_MODEL. Available: ' + ', '.join(m['name'] for m in models))
        model = models[0]['name']
    result = manager.codex_command(record, model, args, local=True)
    if model_argument(args):
        result = result[:1] + result[3:]  # Keep the caller's model flag exactly once.
    return result


def main():
    try:
        args = command(sys.argv[1:])
        os.execv(args[0], args)
    except KeyboardInterrupt:
        return 130
    except Exception as exc:
        print('codex-local: ' + str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    sys.exit(main())

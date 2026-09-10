"""Fail before upload if a tag or the runtime version disagrees with metadata."""
import ast
import os
from pathlib import Path
import tomllib

root = Path(__file__).resolve().parents[1]
project = tomllib.loads((root / 'pyproject.toml').read_text())['project']
version = project['version']
module = ast.parse((root / 'src/nersc_ollama_manager/__init__.py').read_text())
versions = [ast.literal_eval(node.value) for node in module.body
            if isinstance(node, ast.Assign)
            and any(isinstance(t, ast.Name) and t.id == '__version__' for t in node.targets)]
if versions != [version]:
    raise SystemExit('pyproject.toml and __version__ must agree')
tag = os.environ.get('RELEASE_TAG', '')
if tag and tag not in (version, 'v' + version):
    raise SystemExit(f'Tag {tag!r} does not match package version {version!r}')
print(f'Release version verified: {version}')

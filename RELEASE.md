# Publishing releases

The package ships three entry points: `nersc-ollama`, `nersc-ollama-tui`, and
`codex-local`. Install the dashboard with `pip install 'nersc-ollama-manager[tui]'`.

## One-time setup

1. Create the GitHub repository and commit this source tree, including
   `.github/workflows/release.yml`. Choose and add a license before distributing.
   Do not commit local environments, model data, generated runtime/config files,
   or historical build artifacts.
2. In GitHub repository Settings → Environments, create `pypi`. Allow release
   tags (for example `v*`) as deployment refs. Leave required reviewers unset if
   you want automatic publication on every valid tag. Restrict who can create
   release tags using repository rulesets.
3. On PyPI, configure a **pending trusted publisher** if this project has never
   been published, or add a trusted publisher to the existing project:

   | Field | Value |
   | --- | --- |
   | PyPI project | `nersc-ollama-manager` |
   | GitHub owner | `matplo` |
   | Repository | `nersc-ollama-manager` |
   | Workflow filename | `release.yml` |
   | Environment | `pypi` |

   Project-name availability and permission to publish must be checked on PyPI.
   No PyPI API token or GitHub secret is required. The publish job alone receives
   `id-token: write`. It downloads previously tested build artifacts and does not
   check out or execute project code.

   See [PyPI pending publishers](https://docs.pypi.org/trusted-publishers/creating-a-project-through-oidc/)
   and [using a trusted publisher](https://docs.pypi.org/trusted-publishers/using-a-publisher/).

## Each release

1. Update `version` in `pyproject.toml` and `__version__` in
   `src/nersc_ollama_manager/__init__.py` to the same unused PyPI version.
2. Commit the change and push the branch. Branch and pull-request builds test
   Python 3.11, 3.13 and 3.14, build the wheel and source distribution, check
   metadata with Twine, and smoke-test the wheel's commands.
3. Tag that commit using the exact version, optionally prefixed with `v`:

   ```bash
   git tag -a v0.2.8 -m 'Release 0.2.8'
   git push origin v0.2.8
   ```

   This example requires setting both version fields to `0.2.8` first.
   **Pushing the tag triggers publication to real PyPI.** Every tag runs the
   workflow, but mismatched tags fail validation and cannot publish. Tests and
   build checks must pass. GitHub Releases are optional; creating one is not
   the trigger. Manual workflow runs only test/build, even when run on a tag.

PyPI does not let you replace an uploaded release file. For changed contents,
bump the version and push a new tag. Authentication/setup failures can be retried
using the failed workflow after correcting the publisher settings. The workflow
intentionally does not silently skip existing files.

Download the `python-distributions` workflow artifact to inspect the exact wheel
and sdist. Publishing never runs a Slurm job, downloads a model, or connects to
NERSC; all tests use temporary state and mocked cluster operations.

## Local validation

```bash
python -m pip install '.[tui]' build twine
python -m unittest discover -s tests -v
python scripts/check_release.py
python -m build --outdir wheelhouse
python -m twine check --strict wheelhouse/*
```

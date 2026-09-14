# NERSC Ollama Manager

For step-by-step CLI and TUI workflows, see [QUICK_START.md](QUICK_START.md).

Run personal Ollama servers in Perlmutter interactive allocations. Includes a CLI,
an optional Textual dashboard, SSH tunnel reuse, model selection, and Codex launch.
Requires Python 3.10+, a Python environment accessible on compute nodes, Slurm,
SSH access to allocated compute nodes, and a separately installed Codex executable.
This package targets any Perlmutter user, not arbitrary Slurm clusters.

## Install into your environment

From a source checkout:

```sh
python -m pip install '.[tui]'
nersc-ollama tui
```

A built wheel can likewise be installed with `python -m pip install /path/to/wheel`.
The distribution name is `nersc-ollama-manager`; the command is `nersc-ollama`.
PyPI publication is pending. No environment manager is required. Activate your
preferred environment before launching; the package never activates another one.

## First launch and configuration

The first normal command creates a private per-user JSON configuration. Help and
`--version` create nothing. Resolution order is:

1. `--config PATH`
2. `NERSC_OLLAMA_CONFIG`
3. `$XDG_CONFIG_HOME/nersc-ollama/config.json`
4. `~/.config/nersc-ollama/config.json`

A missing explicitly selected file is an error; create it with `setup`.
Relative XDG environment values are ignored. Runtime defaults to
`$XDG_DATA_HOME/nersc-ollama` or `~/.local/share/nersc-ollama`; Ollama storage
is its `ollama_root` subdirectory. First-run configuration starts no downloads,
servers, or allocations. Review available storage before downloading large models.

The TUI offers storage fields and an explicit **Install Ollama** action when Ollama
is missing. Existing installations open the dashboard directly. CLI equivalent:

```sh
nersc-ollama setup --runtime /path/to/personal/runtime --root /path/to/personal/ollama
nersc-ollama setup --version 0.34.0
nersc-ollama doctor
```

Storage changes through setup/onboarding are allowed only before installation or
server creation. Existing installations are never automatically moved. Releases
are safely extracted under the root and activated with a `current` symlink.
Recorded SHA-256 values provide provenance, not independent signature validation.
Models are downloaded separately into `ollama_root/models`.

Already have an Ollama on NERSC some other way (a module, a shared install) and
don't want this tool managing its own pinned copy? Point at it instead of
installing:

```sh
nersc-ollama setup --ollama-binary /path/to/existing/ollama
```

(the TUI has the equivalent field in onboarding, next to **Install Ollama**). This
can be changed any time, unlike `--runtime`/`--root` — it never moves or orphans
files, just which binary this tool runs. `setup --version` (and the TUI's
**Install Ollama**) refuse while `ollama_binary` is set, to avoid managing two
Ollamas at once; remove it from your configuration first to switch back to a
self-managed install.

## Allocate and connect

Keep an allocation terminal open:

```sh
nersc-ollama allocate --profile cpu --name cpu
# GPU alternative: supply your actual charging account
nersc-ollama allocate --profile gpu --account YOUR_ACCOUNT_g --name gpu
```

In another login-node terminal using the same configuration:

```sh
nersc-ollama status
nersc-ollama models --server gpu pull YOUR_MODEL_TAG
nersc-ollama codex --server gpu --model YOUR_MODEL_TAG
```

CPU/GPU profiles default to one node, interactive QOS, 30 minutes, and a 65,536-token
context. The GPU profile requests four GPUs and requires an account. CPU can use
the Slurm default account. Edit profiles in your local JSON configuration.
`allocate --dry-run` prints the command; allocation and `stop` require confirmation
or explicit `--yes` authorization. No scheduler action occurs during installation.

The compute worker uses the Python interpreter running the manager, which must be
accessible on compute nodes with the package installed. Codex resolves from the
caller's PATH and inherits the environment and working directory. Arguments after
`--` pass through to Codex. Its own environment/security settings still apply.
`doctor` shows the interpreter, resolved commands, config and storage paths.

The allocation terminal owns the server. Ctrl-C or terminal closure ends this
foreground workflow. Codex exit leaves it running. In the TUI, Allocate suspends
the dashboard; use another terminal to connect. There is no automatic renewal.

## Status and cleanup

* `status --json`: scheduler/discovery state, unique IDs, remaining time.
* `tunnel --server ID`: verify or create a loopback SSH forward and print its URL.
* `disconnect --server ID`: close the owned tunnel, including after job expiry.
* `stop --server ID`: confirm before cancelling the allocation.
* `logs ID`: show recent logs, including for stopped servers.
* `models list`: list completed downloads in shared storage, without a server.
* `models --server ID list`: query the selected running server.
* `models --server ID pull TAG`: explicitly download through the selected server.
* `forget ID` / `forget --all`: remove an expired discovery record once Slurm confirms
  the allocation has ended (unlike the TUI's **Dismiss session**, this actually deletes
  the record, not just hides it from that dashboard).
* `sessions`: list `ollama-*` `screen` sessions on the login node (see `allocate --screen`
  below) — mainly useful while one is still queueing, before it has a discovery record.

Discovery uses private atomic records with heartbeats and scheduler ownership
checks. SSH control sockets live in a private host-local temporary directory.
SSH authentication and host-key trust must already be configured; this package
does not read credentials or modify SSH settings. Models require outbound network
access for pulls. Cloud tags are rejected. Codex models must advertise tool support;
actual tool reliability and memory requirements depend on model and context size.
Multiple servers are supported without load balancing; choose an ID if ambiguous.

## From outside NERSC

Think of this as two roles, not two locations: the **NERSC server side** (Slurm,
the allocated compute node, Ollama itself — always on NERSC) and the **Codex
client side** (wherever you actually type `nersc-ollama` and run Codex — a NERSC
login node, some other node, or a laptop; `--remote` is what tells the client side
it isn't already sitting on NERSC and needs an SSH hop to get there). "Local" vs.
"remote" describes that hop, not which machine matters more — the server side is
what does the work either way.

`status`, `tunnel`, `logs`, `models`, `codex`, `disconnect`, `stop`, `forget`, and
`sessions` can run from any client machine with SSH access to a NERSC login node,
so Codex itself runs on the client side — editing your own files — while only
Ollama inference happens on the NERSC server side. `allocate` (without `--screen`)
and `serve` still need to run directly on NERSC.

`setup` can too, but doesn't have to: `--remote setup` provisions the NERSC-side
configuration directly from the client, no manual SSH login first —
`nersc-ollama --remote --remote-host saul.nersc.gov setup --runtime ... --root ...`
runs `setup` on the login node over SSH, exactly as if you'd typed it there yourself
(omit `--runtime`/`--root`/`--version` to let the remote side pick its own defaults).
The client's own local config, needed either way, comes from its own local
`nersc-ollama setup --runtime ... --root ...`. Ollama itself only needs to be installed
locally for `models pull`; the generated Codex model catalog is written locally
regardless. No `~/.ssh/config` editing is required — every SSH call is built explicitly
from the fields below plus a valid sshproxy key.

```sh
nersc-ollama --remote status
nersc-ollama --remote tunnel --server gpu
nersc-ollama --remote codex --server gpu --model YOUR_MODEL_TAG
```

Codex itself is a plain local process once launched — to keep it running if you close
your terminal, wrap it in your own `screen`/`tmux` session; no special support is
needed: `screen -S mine nersc-ollama --remote codex --server gpu --model YOUR_MODEL_TAG`.

`--remote` alone uses `remote_login_host` from config; `--remote --remote-host HOST`
overrides it for one invocation. (Two separate flags, not one optional-valued flag,
so a bare `--remote` never gets confused with the required subcommand after it.)

Optional configuration (existing configs need no migration):

```json
{
  "remote_login_host": "saul.nersc.gov",
  "remote_user": "your_nersc_username",
  "remote_identity": "~/.ssh/nersc",
  "remote_python": "/path/to/venv/bin/python",
  "remote_config": null
}
```

Prefer a specific login node over the round-robin `perlmutter.nersc.gov` alias to avoid
a different host key on every connection. `remote_user` may differ from the local
account name. `remote_identity` defaults to `~/.ssh/nersc` (the sshproxy-issued key,
~24h lifetime — an auth failure over `--remote` usually means it needs renewing, not a
code bug). Set `remote_python` to the interpreter on the NERSC side that has
`nersc-ollama-manager` installed: a non-interactive `ssh host command` often does not
get interactive-shell PATH/henv activation, so without it a bare `nersc-ollama` is
tried and fails with a clear error naming this field. `remote_config` overrides
`--config` on the remote side only; omit it to let the remote side resolve its own
default.

## Surviving a dropped connection

A dropped SSH connection sends SIGHUP to `salloc`'s foreground process tree, which
releases the allocation (see `salloc(1)`) — with or without `--remote`. `allocate
--screen` avoids this by running detached in a `screen` session on the login node
instead, immune to the terminal that started it going away:

```sh
nersc-ollama allocate --profile gpu --account YOUR_ACCOUNT_g --screen
nersc-ollama --remote --remote-host saul.nersc.gov allocate --profile gpu --account YOUR_ACCOUNT_g --screen
```

`--remote allocate` is otherwise rejected (a foreground `salloc` cannot run over one SSH
round-trip); `--screen` is the one exception, since it starts the session and returns
immediately rather than blocking. `status`, `logs ID`, and `--remote status` remain the
primary way to check on a screen-detached allocation. `nersc-ollama peek NAME` (session
name is `ollama-<name>`) additionally snapshots the session's currently visible output —
a one-shot, human-facing "is salloc still queueing?" check, not a substitute for `logs`.
`nersc-ollama sessions` lists every `ollama-*` screen session currently on the login
node — there's no registry, just what `screen -ls` reports right now — mainly useful
before a discovery record exists yet, or to rediscover a `--name` you forgot. There's
deliberately no "kill session" action: killing one kills the `salloc ...` process inside
it, releasing the allocation much like Ctrl-C would rather than a safe, no-consequence
cleanup — use `stop` (`scancel`) to end an allocation deliberately instead.

## Compatibility and releases

Version 0.2 ignores legacy `python` and `henv` fields without rewriting existing
configuration. Existing paths, models, discovery records and external personal
wrappers remain usable. The package no longer generates wrappers or accepts
`setup --henv`; launch directly from your environment. Existing workers need no
restart for this upgrade.

```sh
PYTHONPATH=src python -m unittest discover -s tests -v
python -m pip wheel --no-deps --no-build-isolation --wheel-dir dist .
```

Tests mock scheduler/SSH operations and run the dashboard headlessly. Live
inference validation requires a separately approved allocation and selected model.
Publication and license selection are separate release steps.

References: [NERSC interactive jobs](https://docs.nersc.gov/jobs/interactive/),
[Ollama Linux installation](https://docs.ollama.com/linux),
[Ollama with Codex](https://docs.ollama.com/integrations/codex).


## Allocation time and download-only mode

The TUI allocation row includes an editable HH:MM:SS duration. Switching CPU/GPU
profiles reloads that profile's default; edits apply to the next request without
rewriting configuration or extending an existing job. Slurm enforces QOS limits.

Enter a model tag, select **Download: DTN**, and choose **Download only**. This is
the default download backend: it uses SSH to an interactive NERSC data transfer
node, starts a temporary loopback Ollama server solely for the pull, and stops it
on completion, failure, or interruption. No Slurm allocation is requested; account
and allocation-time fields do not apply. It publishes no inference discovery record
or tunnel. Model data remains in the configured shared directory for GPU use later.

```sh
nersc-ollama download MODEL_TAG
nersc-ollama download MODEL_TAG --check  # DTN startup check; no model downloaded
nersc-ollama download MODEL_TAG --dry-run
```

The package source, Ollama binary, model storage and log paths must be accessible
on the DTN. The worker is a standalone standard-library script using the DTN's
Python 3.6+; it does not activate or import the login node's Python environment.
SSH authentication and host-key trust use the user's normal SSH configuration.
The remote PTY propagates Ctrl-C; disconnection stops the temporary worker.
Re-run the same download to let Ollama reuse existing download data.

Optional configuration (existing configs need no migration):

```json
"downloads": {
  "dtn_host": "dtn01.nersc.gov",
  "dtn_python": "python3"
}
```

Choose dtn01 through dtn04.nersc.gov. Transfer temporary files live alongside the
model directory, never in DTN /tmp. Logs are stored under the runtime logs directory.
No inference is performed on DTNs. See [NERSC DTN restrictions](https://docs.nersc.gov/systems/dtn/#restrictions).

For the earlier workflow, select **Download: CPU job** or run:

```sh
nersc-ollama download MODEL_TAG --backend cpu --time 01:00:00
```

CPU mode requests a temporary CPU allocation and releases it after the download.
The TUI uses its displayed duration and, if the CPU profile is selected, its account
field; otherwise it uses the configured CPU/default account. The exact Slurm request
is displayed for confirmation. Neither backend automatically falls back to another.


## Downloaded models in the dashboard

The model selector reads completed manifests from the configured shared model
storage. It populates on startup, refreshes periodically and after downloads, and
can be refreshed with **Models**. No allocation, Ollama server, SSH tunnel, or
inference is needed for this list. The manager checks referenced blob sizes without
reading or hashing large model files; partial or malformed manifests are omitted.
The selected model is preserved across refreshes while it remains available.

To launch Codex, select a running server row and press Enter, select a downloaded
model, and click **Codex**. The launcher then verifies the server and its model
availability and establishes the tunnel. A visible downloaded model alone does not
mean an allocation is running. With no allocation, model browsing and DTN downloads
still work. Keep the allocation terminal open and use a second terminal for clients.

## Startup timeout, live logs, and expired records

Compute-node Ollama startup now allows **300 seconds** by default, including for
existing configurations. Set top-level `"startup_timeout": 600` in `config.json`
to allow ten minutes. A profile's `startup_timeout` overrides that global value.
Values are seconds, positive and at most 86400. This controls server readiness;
it does not extend the Slurm allocation or change Ollama's model-load timeout.

After Slurm grants the allocation, Ollama logs stream to the allocation terminal
and are also saved under the runtime logs directory. Startup errors include the
log path. In a second dashboard, select the server record and click **Logs** to
view recent saved output.

Select a row and click **Dismiss session** to hide it persistently from the
dashboard. Jobs, records, logs and models are retained; no Slurm query is needed.
To actually remove an expired record's file, use the CLI instead: `nersc-ollama
forget ID` (or `forget --all` to sweep every currently-unavailable record at once);
both refuse to touch anything Slurm still reports as running.

## Codex model metadata

The launcher generates a private model catalog under `runtime/codex-models` for
the exact selected model tag. It supplies that catalog through invocation-specific
Codex configuration, along with the context limit and a 90% compaction threshold.
Text/image capability comes from Ollama's model information. The context limit
uses the server record's configured context (or the profile for older records),
capped by the model's advertised maximum when available. New workers record their
context at startup, so later configuration edits do not change that recorded value.

Catalogs are named by content hash to avoid conflicts between simultaneous
sessions. Global Codex settings are not rewritten. Restart Codex through the
manager to pick up the generated catalog; an existing Codex session is unaffected.
This addresses the missing-model-metadata warning, not GPU discovery or cold-load
latency. No model weights are downloaded or inference requests made to generate it.

In the TUI session table, use the arrow keys to highlight a session and press
**d** (or click **Dismiss session**) to hide it persistently. Dismissal makes no
Slurm request and leaves jobs, discovery records, logs, and models intact.
Hidden sessions are skipped by dashboard refreshes, but remain available to CLI
commands. Preferences are stored as individual files in
`<runtime>/dismissed-sessions/`; removing a session's preference file restores it.
The shortcut applies only while the table has focus.

To open a shell on a job node, highlight a live session row, then
click **SSH** next to **Codex**. Type `exit` (or press Ctrl-D) to return to the
dashboard. SSH uses your existing authentication and does not extend the
allocation. **Codex** remains available as a separate action.

Server actions use the highlighted row; Enter is optional. For **Codex**, also
choose a model from **Downloaded model**. An empty model selection shows a
validation message. **SSH** does not require a model.

### Codex directly on the allocated node

Install the package in your environment to put `codex-local` on PATH. On the job
node, run `codex-local -m MODEL`, `codex-local exec "your task"`, or
`codex-local resume --last`. Arguments are forwarded unchanged to Codex.
The resolver uses the local loopback server and supplies the same model metadata
as the dashboard. It never starts an allocation, tunnel, or model download.
Set `NERSC_OLLAMA_CONFIG` for a non-default manager config. The model defaults to
`NERSC_OLLAMA_MODEL`, the config's `codex_model`, or the sole installed model.
With multiple local servers, set `NERSC_OLLAMA_SERVER` to the exact server ID.
The caller's working directory and environment are preserved.

The TUI's **SSH** button writes a unique private setup file under
`<runtime>/ssh-sessions/session-*.bash` and runs `bash --rcfile FILE -i` on the
selected node. It loads `.bashrc`, verifies the node, and defines `codex-local`
with the selected configuration, server and model. All arguments are forwarded.

Each shell gets a private `/tmp/nersc-codex.*` directory on that node. The
function sets `CODEX_HOME`, `CODEX_SQLITE_HOME`, and `TMPDIR` only for the launched
command, avoiding shared-home SQLite and temporary-file locking problems.
The existing `~/.codex` database, configuration, credentials and plugins are not
copied, changed or loaded as the user Codex home. Project configuration still
applies. This is an isolated local-model client; user-home customizations are
not automatically inherited.

Repeated `codex-local` calls in the same shell share history. Separate SSH shells
have independent state; node-local history may disappear when the allocation
ends. Setup files are retained for inspection, contain no credentials, and check
the selected node before use. No shell startup file is modified.

`nersc-ollama-tui` is an installed shortcut for `nersc-ollama tui`.
It also accepts global options, for example `nersc-ollama-tui --config /path/to/config.json`.

Release automation and PyPI setup are documented in [RELEASE.md](RELEASE.md).

The SSH setup exports its session directory. The Python `codex-local` entry point
validates the directory ownership, permissions, node, server and configuration
before applying local Codex storage. This also works in child scripts and after
`henv -x` activation; the interactive function is not required. If a bound session
is missing or invalid, the launcher stops instead of falling back to shared home.
After upgrading, open a new TUI SSH shell to receive the exported binding.
Shell scripts should begin with `#!/usr/bin/env bash`.

Select the **GPU** profile and toggle **Spread across GPUs** to save the Ollama
placement preference, or from the CLI, add `--gpu-spread`/`--no-gpu-spread` to
`allocate --profile gpu` (only valid with the GPU profile; persisted the same way
as the TUI switch, right after you confirm the Slurm request — a `--dry-run` never
persists it, only previews the resulting command). New GPU workers set
`OLLAMA_SCHED_SPREAD` from `profiles.gpu.sched_spread` in the manager configuration;
no shell export is needed. When enabled, the Slurm step also uses `--gpu-bind none`
so the single Ollama process can see the full GPU allocation. Turning it off permits
Ollama to use one GPU when the model fits.
Existing servers are unchanged. Without a saved preference, the environment
variable remains the fallback. Spreading does not guarantee faster generation.

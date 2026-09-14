# Quick start: CLI and TUI

Activate the Python environment where you installed `nersc-ollama-manager`.
From a source checkout, install the dashboard with:

```sh
python -m pip install '.[tui]'
```

All examples use your default configuration. For an existing custom configuration,
set `NERSC_OLLAMA_CONFIG=/absolute/path/to/config.json`, or put `--config PATH`
before the subcommand. The first normal launch creates a per-user configuration;
no download or job is started automatically. See the [README](README.md) for XDG
locations and configuration options.

## TUI workflow

Run this from a login-node terminal:

```sh
nersc-ollama tui
```

1. **First-time setup:** review the runtime and model storage paths. Enter an exact
   Ollama release version and click **Install Ollama**. Large models need substantial
   storage; choose your personal shared directory before installing.
2. **Download a model:** enter its exact Ollama tag, select **Download: DTN**, and
   click **Download only**. SSH may request authentication. This uses a temporary
   server on a transfer node and requires no CPU/GPU allocation. On success it stops
   the temporary server and returns to the dashboard. Ctrl-C interrupts the transfer
   and returns to the dashboard; existing downloaded data is retained.
3. **Browse downloads:** the **Downloaded model (shared storage)** selector lists
   completed local downloads, even with no allocation. Click **Models** to refresh;
   the list also refreshes automatically. A download still in progress is not listed.
4. **Allocate a server:** select CPU or GPU, enter the account if needed, and edit
   the **HH:MM:SS** field beside it. For example, `01:00:00` requests an hour.
   Switching profiles reloads that profile's default time. Click **Allocate** and
   approve the displayed Slurm request. The TUI suspends while this terminal holds
   the allocation. Wait for the server-ready message and keep that terminal open.
5. **Launch Codex from a second terminal:** change to the project directory and
   open another TUI using the same configuration. Select the running server row
   and press **Enter**, choose a downloaded model, and click **Codex**. The manager
   checks the server, establishes/reuses the tunnel, and launches Codex in that
   terminal's working directory and environment.

Codex runs on the login node; inference runs on the allocated compute node.
Quitting Codex leaves the allocation running. The allocation ends at its walltime,
or when you interrupt its owning terminal. **Stop job** asks before cancelling it.
An edited time field affects the next request, not an existing allocation.

**Pull model** differs from **Download only**: Pull uses the selected live server.
For downloads without an allocation, choose Download only with the DTN backend.
The optional **Download: CPU job** backend requests CPU time and uses the duration
field. DTN mode ignores the allocation account and duration.

## CLI workflow

Initialize and install Ollama once:

```sh
nersc-ollama setup --runtime /absolute/path/to/runtime --root /absolute/path/to/ollama_root
nersc-ollama setup --version 0.34.0
nersc-ollama doctor
```

Download an exact model tag and view completed downloads on the login node:

```sh
nersc-ollama download MODEL_TAG
nersc-ollama models list
```

Optional checks and CPU alternative:

```sh
nersc-ollama download MODEL_TAG --check
nersc-ollama download MODEL_TAG --dry-run
nersc-ollama download MODEL_TAG --backend cpu --time 01:00:00
```

`--check` starts/stops the DTN download server without downloading the model.
`--dry-run` prints the command without executing it.

In an allocation terminal:

```sh
nersc-ollama allocate --profile gpu --account YOUR_ACCOUNT_g --name gpu --time 01:00:00
# Or, for CPU inference:
nersc-ollama allocate --profile cpu --name cpu --time 01:00:00
```

In a second login-node terminal:

```sh
cd /path/to/your/project
nersc-ollama status
nersc-ollama models --server gpu list
nersc-ollama codex --server gpu --model MODEL_TAG
```

Use the exact tag shown by `models list`. If multiple live servers share a name,
use the full server ID printed by `status`. Extra Codex arguments go after `--`.

## End-to-end: start on NERSC, drive Codex from your own machine

This ties `allocate --screen` (survives you logging out) and `--remote` (the
client-side commands) into one concrete walkthrough: start a durable GPU server on
the **NERSC server side**, then leave it running and launch Codex from the
**client side** — your own machine here, though it could just as well be a NERSC
login node or any other machine with SSH access; `--remote` just means "this
client isn't already sitting on NERSC, hop there over SSH."

### 1. On NERSC: start the server

One-time setup, from a NERSC login-node terminal — use a CFS path, not your home
directory: it holds discovery records the compute node writes and the login node
reads back (so both need to see it), plus the Ollama binary and every downloaded
model (multi-GB to tens of GB each; home quotas are too small):

```sh
nersc-ollama setup --runtime /global/cfs/cdirs/YOUR_PROJECT/YOUR_USER/nersc-ollama
nersc-ollama setup --version 0.34.0
nersc-ollama download MODEL_TAG
```

`--root` (Ollama's own binary + models) defaults to `<runtime>/ollama_root` and
is omitted above for that reason — pass it explicitly only if you want the large
model storage on a *different* filesystem/quota than `runtime`'s small bookkeeping
(e.g. `$SCRATCH` instead of a smaller CFS project allocation).

Already have an Ollama on NERSC some other way (a module, a shared install)?
Skip `setup --version` entirely and point at it instead:
`nersc-ollama setup --ollama-binary /path/to/existing/ollama` — see the
[README](README.md#first-launch-and-configuration) for details.

Request the allocation **detached in a screen session**, so it survives you logging
out or your connection dropping. Add `--gpu-spread` here too if you want Ollama
spread across all requested GPUs rather than packed onto as few as fit the model —
it's persisted for future GPU allocations (like the TUI's **Spread across GPUs**
switch), not just this one:

```sh
nersc-ollama allocate --profile gpu --account YOUR_ACCOUNT_g --name gpu --time 04:00:00 --screen --gpu-spread
```

This prints the `salloc ...` command, asks you to confirm, starts a `screen`
session named `ollama-gpu`, types that command into it, and returns immediately —
unlike a plain `allocate`, it does not hold your terminal open. Check on it:

```sh
nersc-ollama sessions   # confirm the ollama-gpu session exists
nersc-ollama peek gpu   # one-shot look at its output, e.g. "still queueing"
nersc-ollama status     # once ready, shows it available with remaining time
```

You can now log out of NERSC entirely — the job keeps running.

### 2. From your own machine: launch Codex against it

One-time setup, on your own machine (not NERSC) — a bare config, no Ollama
install needed unless you also want local `models pull`:

```sh
nersc-ollama setup --runtime /local/path/nersc-ollama
```

Back on NERSC, run `nersc-ollama doctor` **interactively** and note its `"python"`
and `"config"` fields — you need both, not just the login host, because a
non-interactive SSH session (what `--remote` runs) won't see whatever env var or
module load your interactive shell uses to find either one. Add all of this to
the client config (or pass `--remote-host` each time instead of `remote_login_host` —
see the [README](README.md#from-outside-nersc) for every field):

```json
{
  "remote_login_host": "saul.nersc.gov",
  "remote_user": "YOUR_NERSC_USERNAME",
  "remote_identity": "~/.ssh/nersc",
  "remote_python": "PASTE_doctor's_python_FIELD_HERE",
  "remote_config": "PASTE_doctor's_config_FIELD_HERE"
}
```

Skipping `remote_python`/`remote_config` is the most common way this goes wrong:
`--remote status` silently succeeds with zero servers (no error) instead of
matching what `status` shows when run directly on NERSC.

Then, from your project directory on your own machine:

```sh
nersc-ollama --remote status                              # see the gpu server from step 1
nersc-ollama --remote codex --server gpu --model MODEL_TAG
```

Codex now runs **locally**, editing files in your current directory, while only
Ollama inference is tunneled to the compute node started in step 1. Quitting Codex
leaves the NERSC allocation running; end it deliberately when you're done with
`nersc-ollama stop --server gpu` directly on NERSC, or `nersc-ollama --remote stop
--server gpu` from your own machine. Wrap the `codex` invocation in your own
`screen`/`tmux` session if you want it to survive your own terminal closing too —
it's a plain local process, no special support needed.

## Common questions

- **Downloaded models are visible, but Codex won't start:** select a running server;
  downloads persist after jobs expire. Codex also requires a tool-capable model.
- **The model selector is empty:** click Models, verify the download ended with
  `success`, and check `nersc-ollama doctor` for the configured model storage path.
  All terminals must use the same configuration. Incomplete downloads are omitted.
- **The allocation disappeared:** the default request is 30 minutes, including
  startup and any server-side downloads. Set a longer supported duration before
  requesting another job. DTN downloads do not consume this allocation time.
- **SSH fails:** configure normal NERSC SSH authentication and host-key trust.
  The manager does not edit SSH configuration. DTN logs are in the runtime logs
  directory; server logs are available through `nersc-ollama logs SERVER_ID`.

See the [README](README.md) for full configuration and lifecycle details.

## Startup logs and cleaning the server list

Ollama startup logs appear automatically in the allocation terminal after Slurm
grants the job. The default startup deadline is five minutes. To increase it,
add `"startup_timeout": 600` at the top level of your JSON configuration (ten
minutes), or inside an individual CPU/GPU profile. Restart the worker through a
new allocation for the setting to take effect; existing jobs are not modified.

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

The SSH setup exports its session directory. The Python `codex-local` entry point
validates the directory ownership, permissions, node, server and configuration
before applying local Codex storage. This also works in child scripts and after
`henv -x` activation; the interactive function is not required. If a bound session
is missing or invalid, the launcher stops instead of falling back to shared home.
After upgrading, open a new TUI SSH shell to receive the exported binding.
Shell scripts should begin with `#!/usr/bin/env bash`.

Select the **GPU** profile and toggle **Spread across GPUs** to save the Ollama
placement preference, or from the CLI, add `--gpu-spread`/`--no-gpu-spread` to
`allocate --profile gpu` (see step 1 of the [end-to-end walkthrough](#1-on-nersc-start-the-server)
above). New GPU workers set `OLLAMA_SCHED_SPREAD` from `profiles.gpu.sched_spread`
in the manager configuration; no shell export is needed. When enabled, the Slurm
step also uses `--gpu-bind none` so the single Ollama process can see the full
GPU allocation. Turning it off permits Ollama to use one GPU when the model fits.
Existing servers are unchanged. Without a saved preference, the environment
variable remains the fallback. Spreading does not guarantee faster generation.

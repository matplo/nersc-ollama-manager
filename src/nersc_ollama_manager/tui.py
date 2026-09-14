"""Optional Textual dashboard; shares all operations with the CLI."""
import asyncio
from contextlib import contextmanager
import re
import subprocess
import time

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import Button, DataTable, Footer, Header, Input, RichLog, Select, Static, Switch


class Dashboard(App):
    TITLE = 'NERSC Ollama Manager'
    CSS = '''
    #onboarding { height: auto; }
    DataTable { height: 10; }
    Horizontal { height: 3; }
    Select { width: 1fr; }
    Input { width: 1fr; }
    Button { min-width: 12; }
    RichLog { height: 1fr; }
    #spread_label { width: auto; padding: 1 1 0 1; }
    '''
    BINDINGS = [
        ('q', 'quit', 'Quit'), ('r', 'refresh', 'Refresh'), ('d', 'delete_expired', 'Dismiss session'),
        Binding('1', 'select_row(0)', 'Select 1', show=False),
        Binding('2', 'select_row(1)', 'Select 2', show=False),
        Binding('3', 'select_row(2)', 'Select 3', show=False),
        Binding('4', 'select_row(3)', 'Select 4', show=False),
        Binding('5', 'select_row(4)', 'Select 5', show=False),
        Binding('6', 'select_row(5)', 'Select 6', show=False),
        Binding('7', 'select_row(6)', 'Select 7', show=False),
        Binding('8', 'select_row(7)', 'Select 8', show=False),
        Binding('9', 'select_row(8)', 'Select 9', show=False),
        Binding('0', 'select_row(9)', 'Select 10', show=False),
    ]

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.selected = None
        self.busy = False
        self.records = []
        self.last_launch = None

    @contextmanager
    def terminal_session(self, title=None, resume_hint=None):
        """Restore Textual before propagating errors from foreground commands.

        Textual 8.2's suspend context does not resume if its yield raises.
        """
        failure = None
        with self.suspend():
            print('\033[2J\033[H', end='')
            if title:
                print(f'== {title} ==', flush=True)
                if resume_hint:
                    print(resume_hint, flush=True)
                print('Return here by exiting the foreground command.', flush=True)
            try:
                yield
            except BaseException as exc:
                failure = exc
            finally:
                if title:
                    print(f'== {title} ended; returning to NERSC Ollama Manager ==', flush=True)
        if failure is not None:
            raise failure

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id='onboarding'):
            yield Static('First-time setup: choose two storage paths below, then install a pinned Ollama '
                         'release. Nothing is downloaded, no server starts, and no Slurm job runs yet.')
            yield Input(value=str(self.manager.runtime), placeholder='Runtime directory (discovery records, logs)',
                        id='runtime_path',
                        tooltip="Manager bookkeeping only: discovery records, per-server logs, generated "
                                "Codex model catalogs. Written by the compute node's serve process and read "
                                "from the login node -- use a path both can see (e.g. under $CFS), not $HOME.")
            yield Input(value=str(self.manager.root), placeholder='Ollama storage directory (binary + models)',
                        id='root_path',
                        tooltip='Holds the installed Ollama binary release and every downloaded model '
                                '(multi-GB to tens of GB each). Use a large shared path (e.g. under $CFS '
                                'or $SCRATCH) -- NERSC home directories are too small and are not the '
                                'point of this field.')
            yield Input(value=self.manager.config.get('ollama_binary') or '',
                        placeholder='Existing Ollama binary (optional; e.g. a NERSC module)',
                        id='ollama_binary_path',
                        tooltip='Use an Ollama that already exists on NERSC some other way instead of the '
                                'pinned install below -- an absolute path to an existing ollama executable. '
                                'Leave blank to manage a pinned release here instead.')
            with Horizontal():
                yield Button('Save storage', id='save_storage')
                yield Input(placeholder='Ollama release version, e.g. 0.34.0', id='release',
                            tooltip='An exact release tag from github.com/ollama/ollama/releases. Downloads '
                                    'and verifies that one release only; still starts no server. Not used '
                                    'if an existing Ollama binary is set above.')
                yield Button('Install Ollama', id='install', variant='primary')
        yield Static('Browse downloads without an allocation. For Codex, highlight a live server row and choose a model. Allocate holds this terminal.')
        yield DataTable(id='servers', cursor_type='row')
        yield Static('No server selected.', id='health')
        with Horizontal():
            yield Select([('CPU', 'cpu'), ('GPU', 'gpu')], value='cpu', allow_blank=False, id='profile')
            yield Input(placeholder='Slurm account (GPU requires account)', id='account')
            yield Input(value=self.manager.config['profiles']['cpu']['time'], placeholder='HH:MM:SS', id='walltime', tooltip='Allocation time (HH:MM:SS), including downloads and inference')
            yield Button('Allocate', id='allocate')
            yield Button('Stop job', id='stop', variant='warning')
            yield Button('Dismiss session', id='delete_expired', tooltip='Hide the selected session from this dashboard; keep the job, record, logs and models')
        with Horizontal():
            yield Static('Spread across GPUs (saved for new GPU servers)', id='spread_label')
            yield Switch(value=self.manager.gpu_spread(), id='gpu_spread', disabled=True)
        with Horizontal():
            yield Select([], prompt='Downloaded model (shared storage)', id='models')
            yield Button('Models', id='load_models')
            yield Button('Codex', id='codex', variant='primary')
            yield Button('Resume Codex', id='resume_codex')
            yield Button('SSH', id='ssh', tooltip='Open an interactive shell on the selected job node; exit to return')
        with Horizontal():
            yield Input(placeholder='Exact model tag to download', id='model_tag')
            yield Button('Pull model', id='pull')
            yield Select([('Download: DTN', 'dtn'), ('Download: CPU job', 'cpu')], value='dtn', allow_blank=False, id='download_backend')
            yield Button('Download only', id='download', tooltip='Pull via the selected download backend; DTN requires no allocation')
            yield Button('Tunnel', id='tunnel')
            yield Button('Disconnect', id='disconnect')
            yield Button('Logs', id='logs')
        yield RichLog(id='output', wrap=True)
        yield Footer()

    async def on_mount(self):
        self.query_one('#onboarding').display = not self.manager.binary.is_file()
        self.query_one('#servers', DataTable).add_columns('#', 'ID', 'Profile', 'Node', 'Remaining', 'State')
        await self.action_refresh()
        self.set_interval(15, self.action_refresh)

    def on_select_changed(self, event: Select.Changed):
        if event.select.id == 'profile' and isinstance(event.value, str):
            self.query_one('#walltime', Input).value = self.manager.config['profiles'][event.value]['time']
            self.query_one('#gpu_spread', Switch).disabled = event.value != 'gpu'

    async def on_switch_changed(self, event: Switch.Changed):
        if event.switch.id != 'gpu_spread':
            return
        try:
            await asyncio.to_thread(self.manager.set_gpu_spread, event.value)
            self.query_one(RichLog).write('GPU spreading saved: ' + ('on' if event.value else 'off') + '. Applies to newly started GPU servers; running jobs are unchanged.')
        except Exception as exc:
            with self.prevent(Switch.Changed):
                event.switch.value = self.manager.gpu_spread()
            self.query_one(RichLog).write(f'Cannot save GPU spreading: {exc}')

    async def action_refresh(self):
        if self.busy:
            return
        try:
            records = await asyncio.to_thread(self.manager.list_servers, dashboard=True)
            self.records = records
            table = self.query_one('#servers', DataTable)
            previous = self.highlighted_session()
            table.clear()
            for index, r in enumerate(records):
                shortcut = str(index + 1) if index < 9 else '0' if index == 9 else ''
                table.add_row(shortcut, r['id'], r.get('profile', '—'), r.get('host', '—'), r.get('remaining', '—'),
                              'ready' if r['available'] else r.get('error', 'unavailable'), key=r['id'])
            for index, record in enumerate(records):
                if record['id'] == previous:
                    table.move_cursor(row=index)
                    break
            self.selected = self.highlighted_session()
            self.update_health()
        except Exception as exc:
            self.query_one(RichLog).write(str(exc))
        await self.refresh_models()

    async def refresh_models(self):
        try:
            models = await asyncio.to_thread(self.manager.downloaded_models)
            select = self.query_one('#models', Select)
            previous = select.value
            select.set_options([(m['name'], m['name']) for m in models])
            if previous in {m['name'] for m in models}:
                select.value = previous
            return len(models)
        except Exception as exc:
            self.query_one(RichLog).write(f'Cannot read downloaded models: {exc}')
            return 0

    def highlighted_session(self):
        table = self.query_one('#servers', DataTable)
        if not table.row_count:
            return None
        return str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)

    def highlighted_record(self):
        selected = self.highlighted_session()
        for record in self.records:
            if record.get('id') == selected:
                return record
        return None

    def update_health(self):
        record = self.highlighted_record()
        if not record:
            self.query_one('#health', Static).update('No server selected.')
            return
        heartbeat = record.get('heartbeat')
        age = '—'
        if isinstance(heartbeat, (int, float)):
            age = f'{max(0, int(time.time() - heartbeat))}s'
        context = record.get('context_length', '—')
        try:
            log = self.manager.logs / (record['id'] + '.log')
        except Exception:
            log = '—'
        parts = [
            f'Selected {record["id"]}',
            f'profile {record.get("profile", "—")}',
            f'node {record.get("host", "—")}',
            f'remaining {record.get("remaining", "—")}',
            f'heartbeat {age}',
            f'context {context}',
            f'log {log}',
        ]
        if self.last_launch and self.last_launch.get('server') == record['id']:
            parts.extend([
                f'model {self.last_launch["model"]}',
                f'tunnel http://127.0.0.1:{self.last_launch["port"]}',
                f'catalog {self.last_launch["catalog"]}',
            ])
        self.query_one('#health', Static).update(' | '.join(str(part) for part in parts))

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        self.selected = str(event.row_key.value)
        self.update_health()
        self.query_one(RichLog).write(f'Selected {self.selected}')

    def action_select_row(self, index):
        if self.busy:
            return
        table = self.query_one('#servers', DataTable)
        if 0 <= int(index) < table.row_count:
            table.move_cursor(row=int(index))
            self.selected = self.highlighted_session()
            self.update_health()
            self.query_one(RichLog).write(f'Selected {self.selected}')

    async def action_delete_expired(self):
        """Dismiss the highlighted row, without requiring Enter first."""
        table = self.query_one('#servers', DataTable)
        if self.busy or not table.has_focus:
            return
        if not table.row_count:
            return
        self.selected = str(table.coordinate_to_cell_key(table.cursor_coordinate).row_key.value)
        await self.on_button_pressed(Button.Pressed(self.query_one('#delete_expired', Button)))

    async def on_button_pressed(self, event: Button.Pressed):
        if self.busy:
            return
        self.busy = True
        output = self.query_one(RichLog)
        try:
            action = event.button.id
            if action in ('save_storage', 'install'):
                runtime = self.query_one('#runtime_path', Input).value
                root = self.query_one('#root_path', Input).value
                await asyncio.to_thread(self.manager.set_storage, runtime, root)
                ollama_binary = self.query_one('#ollama_binary_path', Input).value.strip()
                if ollama_binary:
                    await asyncio.to_thread(self.manager.set_ollama_binary, ollama_binary)
                if action == 'install':
                    version = self.query_one('#release', Input).value.strip()
                    if not version:
                        raise RuntimeError('Enter an explicit Ollama release version.')
                    with self.terminal_session('Install Ollama'):
                        self.manager.install(version)
                    self.query_one('#onboarding').display = False
                    output.write('Ollama installed. Select a profile to request an allocation.')
                else:
                    output.write(f'Storage saved; using existing Ollama binary: {ollama_binary}' if ollama_binary
                                 else 'Storage saved. No download or allocation started.')
                    if self.manager.binary.is_file():
                        self.query_one('#onboarding').display = False
                return
            if action in ('allocate', 'download'):
                profile = self.query_one('#profile', Select).value
                account = self.query_one('#account', Input).value or None
                walltime = self.query_one('#walltime', Input).value.strip()
                backend = self.query_one('#download_backend', Select).value
                if (action == 'allocate' or backend == 'cpu') and (not re.fullmatch(r'\d{2,3}:[0-5]\d:[0-5]\d', walltime) or not any(int(part) for part in walltime.split(':'))):
                    raise ValueError('Allocation time must be a positive HH:MM:SS duration (for example 01:00:00).')
                with self.terminal_session('Download model' if action == 'download' else 'Allocate server'):
                    if action == 'download':
                        model = self.query_one('#model_tag', Input).value.strip()
                        cpu_account = account if profile == 'cpu' else None
                        self.manager.download(model, account=cpu_account, walltime=walltime, backend=backend)
                    else:
                        self.manager.allocate('default', profile, account=account, walltime=walltime)
                return
            if action == 'load_models':
                count = await self.refresh_models()
                output.write(f'{count} completed model(s) in shared storage. A live server is needed only to launch Codex.')
                return
            self.selected = self.highlighted_session()
            if not self.selected:
                raise RuntimeError('Select a live server row to launch Codex or use server actions. Downloaded models remain available without an allocation.')
            if action == 'delete_expired':
                await asyncio.to_thread(self.manager.dismiss_session, self.selected)
                output.write(f'Dismissed {self.selected} from the dashboard.')
                self.selected = None
                return
            if action == 'logs':
                from .core import safe_name
                safe_name(self.selected)
                from collections import deque
                with open(self.manager.logs / (self.selected + '.log')) as stream:
                    output.write(''.join(deque(stream, maxlen=50)))
                return
            record = await asyncio.to_thread(
                self.manager.select_for_disconnect if action == 'disconnect' else self.manager.select, self.selected)
            if action == 'tunnel':
                port = await asyncio.to_thread(self.manager.ensure_tunnel, record)
                output.write(f'Tunnel ready: http://127.0.0.1:{port}')
            elif action == 'disconnect':
                await asyncio.to_thread(self.manager.disconnect, record)
                output.write('Tunnel disconnected.')
            elif action == 'stop':
                with self.terminal_session('Stop allocation'):
                    self.manager.stop_allocation(record)
            elif action == 'pull':
                model = self.query_one('#model_tag', Input).value
                with self.terminal_session('Pull model'):
                    self.manager.pull(record, model)
            elif action == 'ssh':
                from .ssh_session import ssh_command
                model = self.query_one('#models', Select).value
                command = ssh_command(self.manager, record, model)
                with self.terminal_session(f'SSH {record["id"]}', 'Type exit or press Ctrl-D to return to the dashboard.'):
                    code = subprocess.call(command)
                output.write(f'SSH session ended (exit {code}); returned to dashboard.')
            elif action in ('codex', 'resume_codex'):
                model = self.query_one('#models', Select).value
                if not isinstance(model, str) or not model:
                    raise RuntimeError('Load and select an installed model first.')
                launch = await asyncio.to_thread(self.manager.codex_launch_info, record, model)
                self.last_launch = launch
                self.update_health()
                output.write(
                    'Codex launch\n'
                    f'  server: {launch["server"]}\n'
                    f'  host: {launch["host"]}\n'
                    f'  tunnel: http://127.0.0.1:{launch["port"]}\n'
                    '  provider: nersc_ollama\n'
                    f'  model: {launch["model"]}\n'
                    f'  context: {launch["context"]}\n'
                    f'  catalog: {launch["catalog"]}'
                )
                extra = ['resume', '--last'] if action == 'resume_codex' else []
                resume = f'Resume later with: nersc-ollama codex --server {record["id"]} --model {model} -- resume --last'
                command = await asyncio.to_thread(self.manager.codex_command, record, model, extra, launch=launch)
                with self.terminal_session('Codex', resume):
                    code = subprocess.call(command)
                output.write(f'Codex ended (exit {code}); {resume}')
        except KeyboardInterrupt:
            output.write('Operation interrupted; returning to dashboard.')
        except Exception as exc:
            output.write(f'Error: {exc}')
        finally:
            self.busy = False
            if event.button.id == 'delete_expired' and self.selected is None:
                table = self.query_one('#servers', DataTable)
                for key in list(table.rows):
                    if (self.manager.runtime / 'dismissed-sessions' / (str(key.value) + '.json')).exists():
                        table.remove_row(key)
            if event.button.id in ('download', 'pull', 'save_storage'):
                await self.refresh_models()

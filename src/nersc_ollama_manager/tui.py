"""Optional Textual dashboard; shares all operations with the CLI."""
import asyncio
from contextlib import contextmanager
import re
import subprocess

from textual.app import App, ComposeResult
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
    BINDINGS = [('q', 'quit', 'Quit'), ('r', 'refresh', 'Refresh'), ('d', 'delete_expired', 'Dismiss session')]

    def __init__(self, manager):
        super().__init__()
        self.manager = manager
        self.selected = None
        self.busy = False

    @contextmanager
    def terminal_session(self):
        """Restore Textual before propagating errors from foreground commands.

        Textual 8.2's suspend context does not resume if its yield raises.
        """
        failure = None
        with self.suspend():
            try:
                yield
            except BaseException as exc:
                failure = exc
        if failure is not None:
            raise failure

    def compose(self) -> ComposeResult:
        yield Header()
        with Vertical(id='onboarding'):
            yield Static('First-time setup: review storage, then explicitly install an Ollama release. No models or jobs are started.')
            yield Input(value=str(self.manager.runtime), placeholder='Absolute runtime directory', id='runtime_path')
            yield Input(value=str(self.manager.root), placeholder='Absolute Ollama storage directory', id='root_path')
            with Horizontal():
                yield Button('Save storage', id='save_storage')
                yield Input(placeholder='Ollama release version, e.g. 0.34.0', id='release')
                yield Button('Install Ollama', id='install', variant='primary')
        yield Static('Browse downloads without an allocation. For Codex, highlight a live server row and choose a model. Allocate holds this terminal.')
        yield DataTable(id='servers', cursor_type='row')
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
        self.query_one('#servers', DataTable).add_columns('ID', 'Profile', 'Node', 'Remaining', 'State')
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
            table = self.query_one('#servers', DataTable)
            previous = self.highlighted_session()
            table.clear()
            for r in records:
                table.add_row(r['id'], r.get('profile', '—'), r.get('host', '—'), r.get('remaining', '—'),
                              'ready' if r['available'] else r.get('error', 'unavailable'), key=r['id'])
            for index, record in enumerate(records):
                if record['id'] == previous:
                    table.move_cursor(row=index)
                    break
            self.selected = self.highlighted_session()
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

    def on_data_table_row_selected(self, event: DataTable.RowSelected):
        self.selected = str(event.row_key.value)
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
                if action == 'install':
                    version = self.query_one('#release', Input).value.strip()
                    if not version:
                        raise RuntimeError('Enter an explicit Ollama release version.')
                    with self.terminal_session():
                        self.manager.install(version)
                    self.query_one('#onboarding').display = False
                    output.write('Ollama installed. Select a profile to request an allocation.')
                else:
                    output.write('Storage saved. No download or allocation started.')
                return
            if action in ('allocate', 'download'):
                profile = self.query_one('#profile', Select).value
                account = self.query_one('#account', Input).value or None
                walltime = self.query_one('#walltime', Input).value.strip()
                backend = self.query_one('#download_backend', Select).value
                if (action == 'allocate' or backend == 'cpu') and (not re.fullmatch(r'\d{2,3}:[0-5]\d:[0-5]\d', walltime) or not any(int(part) for part in walltime.split(':'))):
                    raise ValueError('Allocation time must be a positive HH:MM:SS duration (for example 01:00:00).')
                with self.terminal_session():
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
                with self.terminal_session():
                    self.manager.stop_allocation(record)
            elif action == 'pull':
                model = self.query_one('#model_tag', Input).value
                with self.terminal_session():
                    self.manager.pull(record, model)
            elif action == 'ssh':
                from .ssh_session import ssh_command
                model = self.query_one('#models', Select).value
                command = ssh_command(self.manager, record, model)
                with self.terminal_session():
                    code = subprocess.call(command)
                output.write(f'SSH session ended (exit {code}); returned to dashboard.')
            elif action == 'codex':
                model = self.query_one('#models', Select).value
                if not isinstance(model, str) or not model:
                    raise RuntimeError('Load and select an installed model first.')
                command = await asyncio.to_thread(self.manager.codex_command, record, model)
                with self.terminal_session():
                    subprocess.call(command)
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

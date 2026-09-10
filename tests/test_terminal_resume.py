from contextlib import contextmanager
import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from nersc_ollama_manager.core import Manager
from nersc_ollama_manager.tui import Dashboard


class TerminalRestoreTests(unittest.TestCase):
    def test_errors_propagate_only_after_resume(self):
        app=Dashboard(None)
        for error in (RuntimeError('declined'), KeyboardInterrupt(), OSError('command failed')):
            events=[]
            @contextmanager
            def suspend():
                events.append('suspended')
                yield
                events.append('resumed')
            with patch.object(app,'suspend',suspend),self.assertRaises(type(error)):
                with app.terminal_session():
                    raise error
            self.assertEqual(events,['suspended','resumed'])


class DeclineTests(unittest.IsolatedAsyncioTestCase):
    async def test_decline_restores_dashboard_and_submits_nothing(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            manager=Manager.initialize(base/'config.json',base/'runtime',base/'ollama')
            app=Dashboard(manager)
            events=[]
            @contextmanager
            def suspend():
                events.append('suspended')
                yield
                events.append('resumed')
            with patch.object(app,'suspend',suspend),patch.object(manager,'require_installation'), \
                 patch('nersc_ollama_manager.core.os.isatty',return_value=True), \
                 patch('builtins.input',return_value='N'), \
                 patch('nersc_ollama_manager.core.run') as run, \
                 patch.dict('os.environ',{'SLURM_JOB_ID':''}):
                async with app.run_test(size=(160,50)) as pilot:
                    await pilot.click('#allocate')
                    await pilot.pause()
                    self.assertEqual(events,['suspended','resumed'])
                    self.assertFalse(app.busy)
                    run.assert_not_called()
                    await pilot.click('#load_models')
                    await pilot.pause()
                    self.assertFalse(app.busy)

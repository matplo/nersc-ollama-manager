import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
from nersc_ollama_manager.core import Manager

class SpreadTests(unittest.IsolatedAsyncioTestCase):
    async def test_switch_persists_without_job(self):
        from nersc_ollama_manager.tui import Dashboard
        from textual.widgets import Switch,Select
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp);m=Manager.initialize(p/'config.json',p/'runtime',p/'root')
            with patch.dict('os.environ',{'OLLAMA_SCHED_SPREAD':'false'}),patch.object(m,'allocate') as allocate:
                app=Dashboard(m)
                async with app.run_test(size=(160,55)) as pilot:
                    switch=app.query_one('#gpu_spread',Switch)
                    self.assertTrue(switch.disabled)
                    app.query_one('#profile',Select).value='gpu'
                    await pilot.pause()
                    self.assertFalse(switch.disabled)
                    await pilot.click('#gpu_spread');await pilot.pause()
                    self.assertTrue(Manager(m.config_path).gpu_spread())
                    await pilot.click('#gpu_spread');await pilot.pause()
                    with patch.dict('os.environ',{'OLLAMA_SCHED_SPREAD':'true'}):
                        self.assertFalse(Manager(m.config_path).gpu_spread())
                allocate.assert_not_called()

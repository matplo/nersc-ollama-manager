import hashlib
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nersc_ollama_manager.core import Manager, atomic_json


def add_model(manager, model='qwen3.8', tag='27b', namespace='library'):
    blob=b'small test fixture'
    digest=hashlib.sha256(blob).hexdigest()
    directory=manager.root/'models/blobs'
    directory.mkdir(parents=True,exist_ok=True)
    (directory/('sha256-'+digest)).write_bytes(blob)
    path=manager.root/'models/manifests/registry.ollama.ai'/namespace/model/tag
    descriptor={'digest':'sha256:'+digest,'size':len(blob)}
    atomic_json(path,{'config':descriptor,'layers':[descriptor]})
    return path


class OfflineModelsTests(unittest.TestCase):
    def test_local_manifest_and_incomplete_download(self):
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            manager=Manager.initialize(base/'config.json',base/'runtime',base/'ollama')
            self.assertEqual(manager.downloaded_models(),[])
            path=add_model(manager)
            add_model(manager,'custom','latest','example')
            with patch('nersc_ollama_manager.core.run',side_effect=AssertionError('No subprocess expected')):
                self.assertEqual({m['name'] for m in manager.downloaded_models()}, {'qwen3.8:27b','example/custom:latest'})
            path.write_text('{')
            self.assertEqual(len(manager.downloaded_models()),1)
            for blob in (manager.root/'models/blobs').iterdir():
                blob.write_bytes(b'incomplete')
            self.assertEqual(manager.downloaded_models(),[])


class OfflineDashboardTests(unittest.IsolatedAsyncioTestCase):
    async def test_model_visible_without_server_and_selection_survives_refresh(self):
        from nersc_ollama_manager.tui import Dashboard
        from textual.widgets import Select
        with tempfile.TemporaryDirectory() as tmp:
            base=Path(tmp)
            manager=Manager.initialize(base/'config.json',base/'runtime',base/'ollama')
            add_model(manager)
            app=Dashboard(manager)
            with patch.object(manager,'select',side_effect=AssertionError('No server needed')),patch.object(manager,'ensure_tunnel',side_effect=AssertionError('No tunnel needed')):
                async with app.run_test(size=(160,50)) as pilot:
                    select=app.query_one('#models',Select)
                    select.value='qwen3.8:27b'
                    self.assertIsNone(app.selected)
                    await pilot.click('#load_models')
                    await pilot.pause()
                    self.assertEqual(select.value,'qwen3.8:27b')
                    await app.action_refresh()
                    self.assertEqual(select.value,'qwen3.8:27b')

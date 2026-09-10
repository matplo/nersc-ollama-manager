import json
from pathlib import Path
import tempfile
import unittest
from nersc_ollama_manager.core import Manager


class CatalogTests(unittest.TestCase):
    def test_exact_tag_context_limit_and_modalities(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);m=Manager.initialize(root/'config.json',root/'runtime',root/'ollama')
            path, context=m.codex_catalog({'profile':'gpu','context_length':32768},'example:27b',
                {'capabilities':['tools','vision'],'model_info':{'general.architecture':'example','example.context_length':262144}})
            entry=json.loads(path.read_text())['models'][0]
            self.assertEqual(context,32768)
            self.assertEqual(entry['slug'],'example:27b')
            self.assertEqual(entry['input_modalities'],['text','image'])
            self.assertEqual(path.stat().st_mode & 0o777,0o600)
            second, _=m.codex_catalog({'profile':'gpu','context_length':32768},'example:27b',
                {'capabilities':['tools','vision'],'model_info':{'general.architecture':'example','example.context_length':262144}})
            self.assertEqual(path,second)
            third, length=m.codex_catalog({},'small:latest',{'model_info':{'general.architecture':'small','small.context_length':8192}})
            self.assertEqual(length,8192)
            self.assertNotEqual(path,third)
            self.assertEqual(json.loads(third.read_text())['models'][0]['input_modalities'],['text'])

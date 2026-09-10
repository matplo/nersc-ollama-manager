import contextlib
import io
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from nersc_ollama_manager.core import Manager, atomic_json, stream_log


class StartupCleanupTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory();self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        self.m=Manager.initialize(self.root/'config.json',self.root/'runtime',self.root/'ollama')

    def test_timeout_defaults_and_overrides(self):
        self.assertEqual(self.m.startup_timeout('gpu'),300)
        self.m.config.pop('startup_timeout')
        self.assertEqual(self.m.startup_timeout('gpu'),300)
        self.m.config['startup_timeout']=600
        self.assertEqual(self.m.startup_timeout('gpu'),600)
        self.m.config['profiles']['gpu']['startup_timeout']=900
        self.assertEqual(self.m.startup_timeout('gpu'),900)
        self.assertEqual(self.m.startup_timeout('cpu'),600)
        for invalid in (0,-1,True,'600',float('nan')):
            self.m.config['profiles']['gpu']['startup_timeout']=invalid
            with self.assertRaises(ValueError):self.m.startup_timeout('gpu')

    def test_delete_only_after_scheduler_confirms_expiry(self):
        path=self.m.records/'test-123.json'
        atomic_json(path,{'id':'test-123','uid':os.getuid(),'job_id':'123'})
        log=self.m.logs/'test-123.log';log.write_text('keep log')
        with patch.object(self.m,'job',return_value={'state':'RUNNING'}),self.assertRaises(RuntimeError):
            self.m.delete_expired('test-123')
        self.assertTrue(path.exists())
        with patch.object(self.m,'job',side_effect=OSError('scheduler unavailable')),self.assertRaises(OSError):
            self.m.delete_expired('test-123')
        self.assertTrue(path.exists())
        with patch.object(self.m,'job',return_value=None):self.m.delete_expired('test-123')
        self.assertFalse(path.exists());self.assertEqual(log.read_text(),'keep log')

    def test_log_stream_keeps_saved_copy(self):
        log=self.root/'server.log';log.write_text('startup\n')
        output=io.StringIO()
        with contextlib.redirect_stdout(output):
            with stream_log(log):
                with log.open('a') as f:f.write('ready\n')
        self.assertEqual(output.getvalue(),'startup\nready\n')
        self.assertEqual(log.read_text(),output.getvalue())

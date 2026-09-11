import json
import os
from pathlib import Path
import socket
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch
from nersc_ollama_manager.session_state import apply_session_state
from nersc_ollama_manager.ssh_session import session_rc
from types import SimpleNamespace
import shlex

class SessionStateTests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='nersc-codex.',dir='/tmp');self.addCleanup(self.tmp.cleanup)
        self.root=Path(self.tmp.name)
        for name in ('sqlite','tmp'):(self.root/name).mkdir(mode=0o700)
        self.binding={'uid':os.getuid(),'host':socket.gethostname().split('.')[0],'server':'test-1','config':'/tmp/test.json'}
        marker=self.root/'session.json';marker.write_text(json.dumps(self.binding));marker.chmod(0o600)
        self.env={'NERSC_CODEX_SESSION_DIR':str(self.root),'NERSC_OLLAMA_SERVER':'test-1','NERSC_OLLAMA_CONFIG':'/tmp/test.json'}

    def test_child_process_applies_exported_state(self):
        code='from nersc_ollama_manager.session_state import apply_session_state; import os,json; apply_session_state(); print(json.dumps([os.environ[k] for k in ("CODEX_HOME","CODEX_SQLITE_HOME","TMPDIR")]))'
        proc=subprocess.run(['bash','--noprofile','--norc','-c',shlex.join([sys.executable,'-c',code])],env={**os.environ,**self.env,'CODEX_HOME':'/wrong/shared/home'},capture_output=True,text=True,check=True)
        self.assertEqual(json.loads(proc.stdout),[str(self.root),str(self.root/'sqlite'),str(self.root/'tmp')])

    def test_reject_wrong_node_server_and_permissions(self):
        with patch.dict(os.environ,self.env):
            with patch('socket.gethostname',return_value='wrong-node'),self.assertRaises(RuntimeError):apply_session_state()
            with patch.dict(os.environ,{'NERSC_OLLAMA_SERVER':'different'}),self.assertRaises(RuntimeError):apply_session_state()
            self.root.chmod(0o755)
            with self.assertRaises(RuntimeError):apply_session_state()
            self.root.chmod(0o700)
            marker=self.root/'session.json';marker.unlink();marker.symlink_to(self.root/'sqlite')
            with self.assertRaises(RuntimeError):apply_session_state()

    def test_missing_binding_fails_closed(self):
        with patch.dict(os.environ,{'NERSC_OLLAMA_SERVER':'test-1'},clear=True),self.assertRaises(RuntimeError):apply_session_state()

    def test_generated_setup_exports_binding_to_child(self):
        manager=SimpleNamespace(config_path=Path('/tmp/test.json'))
        record={'id':'test-1','host':socket.gethostname().split('.')[0]}
        script=session_rc(manager,record).replace('if [[ -f ~/.bashrc ]]; then source ~/.bashrc; fi','')
        code='import os; from nersc_ollama_manager.session_state import apply_session_state; apply_session_state(); print(os.environ["CODEX_HOME"])'
        script+='\n'+shlex.join([sys.executable,'-c',code])
        result=subprocess.run(['bash','--noprofile','--norc','-c',script],capture_output=True,text=True,check=True)
        path=Path(result.stdout.splitlines()[-1]);self.assertEqual(path.parent,Path('/tmp'));self.assertTrue(path.name.startswith('nersc-codex.'))
        import shutil
        shutil.rmtree(path)

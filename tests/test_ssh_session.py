import json
import os
from pathlib import Path
import shlex
import socket
import stat
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from nersc_ollama_manager.ssh_session import ssh_command


class SSHSessionTests(unittest.TestCase):
    def test_named_private_setup_and_local_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            fake=root/'fake python'
            fake.write_text('#!/usr/bin/env python3\nimport os,sys,json,sqlite3\np=os.environ["CODEX_SQLITE_HOME"]+"/test.sqlite"\nc=sqlite3.connect(p);c.execute("pragma journal_mode=wal");c.execute("create table if not exists t(n)");c.commit();c.close()\nprint(json.dumps([sys.argv[1:],{k:os.environ[k] for k in ("CODEX_HOME","CODEX_SQLITE_HOME","TMPDIR","NERSC_OLLAMA_SERVER","NERSC_OLLAMA_CONFIG")}]))\n')
            fake.chmod(0o700)
            manager=SimpleNamespace(runtime=root,config_path=root/'config $(literal).json')
            record={'id':'one','host':socket.gethostname().split('.')[0]}
            with patch('nersc_ollama_manager.ssh_session.sys.executable',str(fake)):
                first=ssh_command(manager,record,'test:1')
                second=ssh_command(manager,record,'test:1')
            setup=Path(shlex.split(first[-1])[2])
            self.assertNotEqual(first[-1],second[-1])
            self.assertEqual(stat.S_IMODE(setup.stat().st_mode),0o600)
            subprocess.run(['bash','-n',str(setup)],check=True)
            # Avoid user shell startup files in the test; exercise all remaining setup.
            source=setup.read_text().replace('if [[ -f ~/.bashrc ]]; then source ~/.bashrc; fi','')
            args=['exec','-m','test:1','a prompt with $(literal)','--json']
            shell=source+'\n'+shlex.join(['codex-local',*args])+'\n'+shlex.join(['codex-local',*args])+'\n'
            result=subprocess.run(['bash','--noprofile','--norc','-c',shell],text=True,capture_output=True,check=True)
            values=[json.loads(line) for line in result.stdout.splitlines() if line.startswith('[')]
            self.assertEqual(len(values),2)
            self.assertEqual(values[0][0],['-m','nersc_ollama_manager.local',*args])
            env=values[0][1]
            self.assertEqual(env,values[1][1])
            self.assertTrue(env['CODEX_HOME'].startswith('/tmp/nersc-codex.'))
            self.assertEqual(env['NERSC_OLLAMA_CONFIG'],str(manager.config_path))
            self.assertTrue(Path(env['CODEX_SQLITE_HOME'],'test.sqlite').is_file())
            # These are only freshly created test artifacts, no user session data.
            import shutil
            shutil.rmtree(env['CODEX_HOME'])

if __name__=='__main__':unittest.main()

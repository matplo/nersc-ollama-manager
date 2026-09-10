import contextlib
import io
import unittest
from unittest.mock import MagicMock, patch
from nersc_ollama_manager.core import run


class CommandErrorsTests(unittest.TestCase):
    def test_slurm_reason_survives_foreground_command(self):
        proc=MagicMock()
        proc.__enter__.return_value=proc
        proc.stdout=iter(['salloc: error: QOSMaxWallDurationPerJobLimit\n',
                          'salloc: error: Job violates accounting/QOS policy\n'])
        proc.wait.return_value=1
        output=io.StringIO()
        with patch('subprocess.Popen',return_value=proc),contextlib.redirect_stdout(output),self.assertRaises(RuntimeError) as exc:
            run(['salloc','--time','05:30:00'],stream_errors=True)
        self.assertIn('QOSMaxWallDurationPerJobLimit',str(exc.exception))
        self.assertIn('Job violates accounting/QOS policy',str(exc.exception))
        self.assertIn('QOSMaxWallDurationPerJobLimit',output.getvalue())

    def test_successful_output_is_streamed(self):
        proc=MagicMock();proc.__enter__.return_value=proc
        proc.stdout=iter(['ready\n']);proc.wait.return_value=0
        with patch('subprocess.Popen',return_value=proc),contextlib.redirect_stdout(io.StringIO()) as output:
            self.assertEqual(run(['salloc'],stream_errors=True).returncode,0)
        self.assertEqual(output.getvalue(),'ready\n')

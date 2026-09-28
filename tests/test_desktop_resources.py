import importlib
import io
import os
from pathlib import Path
import runpy
import sys
import tempfile
import unittest
from unittest.mock import patch

from soci_ai.desktop import resources


class ResourcePathsTests(unittest.TestCase):
    def test_accounts_and_frozen_resources_are_separate(self):
        with tempfile.TemporaryDirectory() as root:
            for account in ('account-a', 'account-b'):
                local = Path(root) / account
                with patch.dict(os.environ, {'LOCALAPPDATA': str(local)}), patch.object(sys, 'argv', ['app']), patch.object(sys, '_MEIPASS', root, create=True), patch.object(sys, 'frozen', True, create=True):
                    self.assertEqual(resources.output_root(), local / 'SOCI-AI-Desktop')
                    self.assertEqual(resources.model_root(), Path(root) / 'models')
                    self.assertFalse(local.exists())

    def test_test_override_requires_diagnostic_and_absolute_path(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'test'
            with patch.dict(os.environ, {'SOCI_AI_TEST_DATA_DIR': str(target), 'LOCALAPPDATA': root}):
                with patch.object(sys, 'argv', ['app']):
                    self.assertEqual(resources.output_root(), Path(root) / 'SOCI-AI-Desktop')
                for mode in ('--self-test-models', '--smoke-test', '--self-test-history'):
                    with patch.object(sys, 'argv', ['app', mode]):
                        self.assertEqual(resources.output_root(), target)
                        self.assertEqual(resources.log_root(), target / 'logs')
                        self.assertEqual(resources.export_root(), target / 'exports')
                self.assertFalse(target.exists())
            with patch.dict(os.environ, {'SOCI_AI_TEST_DATA_DIR': 'relative'}), patch.object(sys, 'argv', ['app', '--smoke-test']):
                with self.assertRaises(ValueError):
                    resources.output_root()

    def test_no_localappdata_uses_current_account(self):
        with patch.dict(os.environ, {}, clear=True), patch.object(sys, 'argv', ['app']), patch.object(Path, 'home', return_value=Path('D:/account')):
            self.assertEqual(resources.output_root(), Path('D:/account/AppData/Local/SOCI-AI-Desktop'))

    def test_launcher_log_failure_still_enters_main(self):
        fake = importlib.import_module('soci_ai.desktop.__main__')
        with patch.object(fake, 'main', return_value=0) as main, patch.object(Path, 'mkdir', side_effect=PermissionError('read-only')), patch.object(sys, 'stderr', io.StringIO()):
            with self.assertRaises(SystemExit) as result:
                runpy.run_path('desktop_launcher.py', run_name='__main__')
            self.assertEqual(result.exception.code, 0)
            main.assert_called_once()

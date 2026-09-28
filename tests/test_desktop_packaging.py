from pathlib import Path
import tempfile
import unittest

import build_desktop


class PackagingTests(unittest.TestCase):
    def test_onefile_contains_all_resources_and_separate_output(self):
        one = build_desktop.build_command(mode='onefile', clean=True)
        folder = build_desktop.build_command(mode='onedir')
        self.assertIn('--onefile', one)
        self.assertNotIn('--onedir', one)
        self.assertIn('--clean', one)
        for flag in ('--distpath', '--workpath', '--specpath'):
            self.assertNotEqual(one[one.index(flag)+1], folder[folder.index(flag)+1])
        self.assertEqual(one[one.index('--name')+1], 'SOCI-AI')
        self.assertTrue(one[one.index('--distpath')+1].endswith('SOCI-AI-Onefile'))
        self.assertIn('--onedir', folder)
        for resource in build_desktop.resource_files(build_desktop.ROOT):
            self.assertTrue(any(str(resource) in arg for arg in one))
        for package in ('torch','tensorflow','soci_ai.live.runtime','PySide6.QtWebEngineCore'):
            self.assertIn(package, one)

    def test_invalid_mode_rejected(self):
        with self.assertRaises(ValueError):
            build_desktop.build_command(mode='bad')

    def test_notice_generation_contains_full_dependency_licenses_and_no_user_data(self):
        with tempfile.TemporaryDirectory() as root:
            target = Path(root) / 'third-party.txt'
            build_desktop.write_notices(target)
            body = target.read_text(encoding='utf-8')
            for name in ('PySide6','sherpa','mediapipe','numpy','Copyright'):
                self.assertIn(name, body)
            self.assertNotIn('history.sqlite3', body)
            self.assertGreater(len(body), 10000)

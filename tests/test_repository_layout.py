from pathlib import Path
import unittest


class RepositoryLayoutTests(unittest.TestCase):
    def test_research_directories_are_present(self):
        root = Path(__file__).resolve().parents[1]
        for directory in ("src","experiments","tests","data","figures","docs","manuscript"):
            with self.subTest(directory=directory):
                self.assertTrue((root / directory / "README.md").is_file())

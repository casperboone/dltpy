import unittest
import tempfile
from os import path
from project_filter import ProjectFilter


class TestCloner(unittest.TestCase):
    def setUp(self) -> None:
        self.directory = tempfile.TemporaryDirectory()

    def tearDown(self) -> None:
        self.directory.cleanup()

    def test_cloning_a_repository(self) -> None:
        open(path.join(self.directory.name, 'python_file.py'), "w+").close()
        open(path.join(self.directory.name, 'non_python_file.txt'), "w+").close()

        ProjectFilter().filter_directory(self.directory.name)

        self.assertTrue(path.isfile(path.join(self.directory.name, 'python_file.py')))
        self.assertFalse(path.isfile(path.join(self.directory.name, 'non_python_file.txt')))


if __name__ == '__main__':
    unittest.main()

import os
import shutil
import unittest
from cloner import Cloner


class TestCloner(unittest.TestCase):
    def test_cloning_a_repository(self) -> None:
        """
        Tests that we can actually clone a real repository.
        """
        shutil.rmtree('./projects/requests__requests')

        Cloner().clone('requests', 'requests')

        self.assertTrue(os.path.isfile('./projects/requests__requests/README.md'))


if __name__ == '__main__':
    unittest.main()

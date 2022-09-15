import unittest
import sys
import os

sys.path.append(os.path.join("../streamlit/pages"))
import util

TEST_ONE = True

class TestUtil(unittest.TestCase):
    def test_sort(self):
        cand_top_n = [{'id':'Proj A', 'score': 1.0},
                      {'id':'Proj B', 'score': 2.0}]
        n = 1
        top_n = util.select_top_n(cand_top_n, n)
        self.assertEqual(top_n[0]['id'], cand_top_n[1]['id'])

if __name__ == '__main__':
    unittest.main()

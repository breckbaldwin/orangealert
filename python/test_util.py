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

    def test_random_n(self):
        budget = 2
        min_threshold = 2.0
        candidates = [{'id':'Proj Below', 'score': 1.5}] * 100
        candidates.extend([{'id':'Proj Above', 'score': 2.0}] * 2)
        for i in range(100):
            winners = \
                util.select_random_n(budget, candidates, min_threshold)
            self.assertEqual(len(winners), budget)
            for winner in winners:
                self.assertEqual(winner['id'], 'Proj Above')


if __name__ == '__main__':
    unittest.main()

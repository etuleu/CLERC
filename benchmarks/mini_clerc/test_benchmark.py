import unittest
from benchmark import metrics


class MetricsTest(unittest.TestCase):
    def test_missing_queries_remain_in_denominator(self):
        scores = metrics({'a':['wrong','gold']}, {'a':['gold'], 'b':['other']})
        self.assertEqual(scores['recall@1'], 0)
        self.assertEqual(scores['recall@10'], 0.5)
        self.assertEqual(scores['mrr@10'], 0.25)

    def test_duplicates_do_not_consume_ranks(self):
        scores = metrics({'a':['wrong','wrong','gold']}, {'a':['gold']})
        self.assertEqual(scores['mrr@10'], 0.5)

    def test_multiple_positives(self):
        scores = metrics({'a':['one']}, {'a':['one','two']})
        self.assertEqual(scores['recall@10'], 0.5)


if __name__ == '__main__':
    unittest.main()

import unittest
from drex import best_passages, passages, payload, probability


class DrexTest(unittest.TestCase):
    def test_windows_cover_tail(self):
        text = ' '.join(map(str, range(510)))
        self.assertIn('509', passages(text)[-1])

    def test_relevant_passage_not_just_prefix(self):
        text = 'unrelated ' * 300 + 'fiduciary ' * 100
        result = best_passages(text, {'fiduciary'}, str.split, {'fiduciary':2}, count=1)
        self.assertIn('fiduciary', result[0]['text'])
        self.assertGreater(result[0]['window'], 0)

    def test_no_ids_or_ranks_in_request(self):
        request = payload('query', {'docid':'gold', 'bm25_rank':1, 'passages':[{'window':0, 'text':'evidence'}]})
        self.assertNotIn('docid', request['state'])
        self.assertNotIn('bm25_rank', request['state'])
        self.assertEqual(request['questions']['relevant']['type'], 'noul')

    def test_probability(self):
        self.assertEqual(probability({'answers':{'relevant':{'noul':0.73}}}), 0.73)
        for value in (True, '0.5', float('nan'), -1, 2):
            with self.assertRaises(ValueError):
                probability({'answers':{'relevant':{'noul':value}}})


if __name__ == '__main__':
    unittest.main()

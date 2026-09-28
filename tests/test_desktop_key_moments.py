import unittest

from soci_ai.desktop.advice import key_moments


def action(at, value=.8, title='唇部收紧增强'):
    return {'at_ms': at, 'modality': 'vision', 'value': value, 'title': title,
            'evidence': '唇部收紧相对中性基线增强，动作强度 100/100；可先放松。'}


class KeyMomentSelectionTests(unittest.TestCase):
    def test_unchanging_repeated_face_events_become_one_episode(self):
        events = [action(at) for at in [8000, 15000, 26000, 34000, 40000]]
        result = key_moments(events)
        self.assertEqual(len(result), 1)
        self.assertEqual(result[0]['start_ms'], 8000)
        self.assertEqual(result[0]['end_ms'], 40000)
        self.assertNotIn('100/100', result[0]['detail'])
        self.assertEqual(len(events), 5, 'Raw exported evidence must remain intact')

    def test_distinct_language_is_prioritized_without_duplicate_phrases(self):
        events = [action(at) for at in range(0, 60000, 1000)]
        for at, text in [(5000, '女生不适合学工科'), (6000, '学历低的人更适合跑腿'), (45000, '女生不适合学工科')]:
            events.append({'at_ms': at, 'modality': 'semantic', 'value': .9,
                           'text': text, 'evidence': text, 'rewrite': '按经验和实际表现讨论。'})
        result = key_moments(events)
        self.assertLessEqual(len(result), 3)
        semantic = [x for x in result if x['kind'] == 'semantic']
        self.assertEqual(len(semantic), 2)
        self.assertTrue(any('女生' in x['detail'] for x in semantic))
        self.assertTrue(any('学历' in x['detail'] for x in semantic))

    def test_well_separated_face_changes_are_not_merged(self):
        result = key_moments([action(0, .6), action(1000, .9), action(60000, .8)])
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0]['at_ms'], 1000)
        self.assertEqual(result[1]['at_ms'], 60000)

    def test_distinct_action_is_not_crowded_out_by_three_stronger_repeats(self):
        result = key_moments([action(0,.9), action(20000,.8), action(40000,.7),
                              action(50000,.6,'眉部收紧增强')])
        self.assertEqual(len(result), 3)
        self.assertTrue(any(item['title'] == '眉部收紧增强' for item in result))


if __name__ == '__main__':
    unittest.main()

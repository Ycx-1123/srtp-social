"""Per-sample labels follow actual evidence, not the shared score threshold."""
import unittest

from soci_ai.desktop.core import RealtimeAssessment


class TimelineCueRecordingTests(unittest.TestCase):
    def test_different_language_categories_have_different_short_labels(self):
        for text, label in [('学历低一点的人更适合跑腿', '疑似学历偏见'),
                            ('女生不适合学工科', '疑似性别能力偏见')]:
            with self.subTest(text=text):
                engine = RealtimeAssessment()
                engine.observe_text(text, 0, True)
                engine.update(0)
                row = engine.report(0, {}, {})['history'][0]
                self.assertEqual(row.get('cues'), [label])
                self.assertNotIn(text, str(row['cues']))

    def test_face_and_voice_labels_are_tied_to_sample_time(self):
        engine = RealtimeAssessment()
        for at, features, pressure, expected in [
            (0, {'brow_tension': .9, 'micro_expression': .9}, 0, ['眉部紧张增强']),
            (.6, {'mouth_downturn': .9, 'micro_expression': .9}, .8, ['嘴角下压增强', '声音压力升高']),
        ]:
            engine.update(at, vision={'face_detected': True, 'captured_at': at,
                                      'features': {'baseline_ready': True, **features}},
                          audio={'captured_at': at, 'pressure': pressure})
            self.assertEqual(engine.history[-1].get('cues'), expected)
        self.assertEqual(engine.history[0]['cues'], ['眉部紧张增强'])

    def test_recovery_keeps_origin_label_but_marks_it_as_recovery(self):
        engine = RealtimeAssessment()
        for at in (0, .3, .6):
            engine.update(at, vision={'face_detected': True, 'captured_at': at,
                                      'features': {'baseline_ready': True, 'mouth_downturn': 1, 'micro_expression': 1}})
        state = engine.update(1.2, vision={'face_detected': True, 'captured_at': 1.2,
                                         'features': {'baseline_ready': True}})
        self.assertGreater(state['sbi'], 25)
        self.assertEqual(engine.history[-1].get('cues'), ['嘴角下压增强'])
        self.assertEqual(engine.history[-1].get('cue_phase'), 'recovering')
        engine.update(20)
        self.assertIsNone(engine.history[-1]['sbi'])
        self.assertEqual(engine.history[-1]['cues'], [])

    def test_reading_card_does_not_leak_an_old_language_label_into_new_evidence(self):
        engine = RealtimeAssessment()
        engine.observe_text('学历低一点的人更适合跑腿', 0, True)
        engine.update(0)
        engine.observe_text('我们根据实际经验来分工', .6, True)
        state = engine.update(.6, vision={'face_detected': True, 'captured_at': .6,
                                         'features': {'baseline_ready': True, 'brow_tension': 1, 'micro_expression': 1}})
        self.assertTrue(state['language_alert'], 'The readable old card still exists')
        self.assertEqual(engine.history[-1].get('cues'), ['眉部紧张增强'])

    def test_brief_unsampled_change_still_explains_the_following_recovery_sample(self):
        engine = RealtimeAssessment()
        engine.update(0, vision={'face_detected': True, 'captured_at': 0,
                                 'features': {'baseline_ready': True}})
        engine.observe_text('女生不适合学工科', .1, True)
        engine.update(.1)
        engine.update(.3)
        # The language has changed before the next half-second chart sample.
        engine.observe_text('按实际经验来分工', .6, True)
        state = engine.update(.6)
        self.assertGreater(state['sbi'], 25)
        self.assertEqual(engine.history[-1].get('cues'), ['疑似性别能力偏见'])
        self.assertEqual(engine.history[-1].get('cue_phase'), 'recovering')

    def test_weak_new_voice_cue_does_not_replace_a_red_semantic_recovery(self):
        engine = RealtimeAssessment()
        engine.observe_text('学历低一点的人更适合跑腿', 0, True)
        for at in (0, .1, .2, .3, .4):
            engine.update(at)
        engine.observe_text('我们按实际经验来分工', .5, True)
        state = engine.update(.5, audio={'captured_at':.5, 'pressure':.45})
        self.assertGreater(state['sbi'], 60)
        self.assertEqual(engine.history[-1].get('cues'), ['疑似学历偏见'])
        self.assertEqual(engine.history[-1].get('cue_phase'), 'recovering')
        engine.update(1, audio={'captured_at':1, 'pressure':0})
        self.assertEqual(engine.history[-1].get('cues'), ['疑似学历偏见'])


if __name__ == '__main__':
    unittest.main()

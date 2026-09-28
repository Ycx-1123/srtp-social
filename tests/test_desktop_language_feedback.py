"""Whole-library, no-device tests for visible and readable language feedback."""
import json
from pathlib import Path
import unittest

from soci_ai.desktop.core import RealtimeAssessment
from soci_ai.desktop.language import analyze_text


CORPUS = Path(__file__).resolve().parents[1] / 'resources/language/social_bias_examples.json'


class LanguageFeedbackTests(unittest.TestCase):
    def test_all_300_risk_rows_expose_labels_quote_and_rewrite(self):
        rows = json.loads(CORPUS.read_text(encoding='utf-8'))['examples']
        missing, categories, labels = [], set(), set()
        for row in rows:
            if row['label'] != 'bias':
                continue
            engine = RealtimeAssessment()
            engine.observe_text(row['text'], 0, True)
            alert = engine.update(0).get('language_alert') or {}
            if (alert.get('category') != row['category'] or
                    not alert.get('category_label') or len(alert.get('tags', [])) < 2 or
                    alert.get('text') != row['text'] or not alert.get('rewrite')):
                missing.append(row['id'])
            categories.add(alert.get('category'))
            labels.update(alert.get('tags', []))
        self.assertEqual(missing, [], 'Every authored risk entry needs actionable visible feedback')
        self.assertEqual(len(categories), 12)
        self.assertGreaterEqual(len(labels), 20)

    def test_all_controls_and_rejected_library_rows_do_not_show_bias_alerts(self):
        rows = json.loads(CORPUS.read_text(encoding='utf-8'))['examples']
        for row in rows:
            text = row['text'] if row['label'] == 'control' else '不要说' + row['text']
            engine = RealtimeAssessment()
            engine.observe_text(text, 0, True)
            self.assertFalse(engine.update(0).get('language_alert'), (row['id'], text))

    def test_reading_hold_does_not_keep_semantic_scores_artificially_high(self):
        engine = RealtimeAssessment()
        engine.observe_text('学历低一点的人更适合跑腿', 0, True)
        states = [engine.update(i / 10, audio={'captured_at': i / 10, 'pressure': 0}) for i in range(101)]
        early, readable, expired = states[10], states[90], states[100]
        self.assertTrue(readable.get('language_alert'))
        self.assertIn('学历', readable['suggestion_title'])
        self.assertLess(readable['sbi'], early['sbi'])
        self.assertFalse(expired.get('language_alert'))

    def test_new_language_category_preempts_previous_reading_hold(self):
        engine = RealtimeAssessment()
        engine.observe_text('学历低一点的人更适合跑腿', 0, True)
        first = engine.update(0).get('language_alert') or {}
        engine.observe_text('女生就是学不好工科吧', 1, True)
        second = engine.update(1).get('language_alert') or {}
        self.assertEqual(first.get('category'), 'education_bias')
        self.assertEqual(second.get('category'), 'gender_ability')

    def test_neutral_return_stays_readable_but_new_action_preempts_hold(self):
        engine = RealtimeAssessment()
        def face(at, **features):
            return {'face_detected': True, 'captured_at': at, 'features': {'baseline_ready': True, **features}}
        first = engine.update(0, vision=face(0, brow_tension=.7, micro_expression=.7))
        neutral = engine.update(.1, vision=face(.1))
        changed = engine.update(1, vision=face(1, smile=.8))
        self.assertEqual(neutral['suggestion_title'], first['suggestion_title'])
        self.assertIn('微笑', changed['suggestion_title'])
        self.assertLessEqual(len(first['suggestion']), 45)

    def test_spoken_contractions_are_matched_across_categories(self):
        cases = (
            ('学历低点的人更适合跑腿', 'education_bias'),
            ('生过孩子的人学新东西会慢些吧', 'family_status'),
            ('胖点的人做事大概也不自律', 'appearance_bias'),
            ('行动慢点的人思考也会慢吧', 'disability_bias'),
            ('女生的空间想象力本来就弱些', 'gender_ability'),
        )
        for text, category in cases:
            result = analyze_text(text)
            self.assertGreater(result['score'], 0)
            self.assertEqual(result['category'], category)

    def test_one_character_asr_error_gets_review_hint_not_a_fabricated_score(self):
        engine = RealtimeAssessment()
        original_asr = '学历低一点的人更适合跑退'
        engine.observe_text(original_asr, 0, True)
        state = engine.update(0)
        self.assertEqual(state['sbi'], 0)
        alert = state.get('language_alert') or {}
        self.assertTrue(alert.get('review_required'))
        self.assertEqual(alert.get('text'), original_asr)
        self.assertEqual(alert.get('category'), 'education_bias')

    def test_changed_polarity_is_not_silently_corrected_to_a_risk_sample(self):
        result = analyze_text('学历高一点的人更适合跑腿')
        self.assertEqual(result['score'], 0)
        self.assertFalse(result.get('review_required'))

    def test_retracted_partial_warning_is_removed_after_final_rejection(self):
        engine = RealtimeAssessment()
        engine.observe_text('女生不适合学工科', 0, False)
        self.assertTrue(engine.update(0).get('language_alert'))
        engine.observe_text('女生不适合学工科这种说法不对', 1, True)
        self.assertFalse(engine.update(1).get('language_alert'))


if __name__ == '__main__':
    unittest.main()

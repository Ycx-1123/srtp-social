import math
import unittest

from soci_ai.desktop.charts import series_segments, timeline_points


class SessionTimelineDataTests(unittest.TestCase):
    def test_missing_values_break_each_series_without_inventing_a_complement(self):
        points = timeline_points([
            {'at_ms': 0, 'sbi': 15, 'friendliness': 85},
            {'at_ms': 500, 'sbi': None, 'friendliness': 80},
            {'at_ms': 1000, 'sbi': 30, 'friendliness': None},
        ])
        self.assertEqual(series_segments(points, 1), [[(0, 15)], [(1, 30)]])
        self.assertEqual(series_segments(points, 2), [[(0, 85), (.5, 80)]])

    def test_time_spacing_is_measured_and_ordered_not_sample_index(self):
        points = timeline_points([
            {'at_ms': 9400, 'sbi': 20},
            {'at_ms': 500, 'sbi': 10},
            {'at_ms': 1750, 'sbi': 15},
        ])
        self.assertEqual([point[0] for point in points], [.5, 1.75, 9.4])

    def test_invalid_measurements_preserve_gaps_and_invalid_times_are_rejected(self):
        points = timeline_points([
            {'at_ms': None, 'sbi': 15},
            {'at_ms': -1, 'sbi': 15},
            {'at_ms': 0, 'sbi': math.nan, 'friendliness': math.inf},
            {'at_ms': 500, 'sbi': 150, 'friendliness': -5},
        ])
        self.assertEqual(points, [(0, None, None), (.5, 100, 0)])


if __name__ == '__main__':
    unittest.main()

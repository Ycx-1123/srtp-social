import unittest
from soci_ai.desktop import __main__ as desktop


class InitialGeometryTests(unittest.TestCase):
    def geometry(self, available):
        helper = getattr(desktop, "initial_geometry", None)
        self.assertTrue(callable(helper), "Startup must fit the screen's available logical area")
        return helper(available)

    def test_small_hidpi_workarea_keeps_controls_inside_screen(self):
        x, y, width, height = self.geometry((0, 0, 1280, 680))
        self.assertLessEqual(width, 1216)
        self.assertLessEqual(height, 616)
        self.assertGreaterEqual(x, 0)
        self.assertGreaterEqual(y, 0)
        self.assertLessEqual(x + width, 1280)
        self.assertLessEqual(y + height, 680)

    def test_secondary_monitor_with_negative_origin_stays_on_that_monitor(self):
        x, y, width, height = self.geometry((-1920, 40, 1920, 1000))
        self.assertGreaterEqual(x, -1920)
        self.assertGreaterEqual(y, 40)
        self.assertLessEqual(x + width, 0)
        self.assertLessEqual(y + height, 1040)


if __name__ == "__main__":
    unittest.main()

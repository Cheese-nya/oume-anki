"""Interval mapping from an OUME card to Anki's four buttons."""

import copy
import unittest

from oume.scheduler_core import Card, CardState, EngineConfig, UserState
from oume.schedule import scheduling_intervals


class ScheduleMappingTests(unittest.TestCase):
    def test_new_card_good_is_a_same_day_pulse(self):
        intervals = scheduling_intervals(Card(card_id="1"), UserState(user_id="local"), 0.0, 1)
        self.assertLess(intervals["good"], 86400)
        self.assertGreaterEqual(intervals["good"], 60)

    def test_graduated_card_good_is_at_least_one_day(self):
        card = Card(card_id="1", state=CardState.REVIEW, S=6.0, D=5.0, last_review_t=0.0)
        intervals = scheduling_intervals(card, UserState(user_id="local"), 2000.0, 3)
        self.assertGreaterEqual(intervals["good"], 86400)
        self.assertIn("again", intervals)
        self.assertIn("easy", intervals)

    def test_interval_preview_does_not_mutate_the_live_card(self):
        card = Card(card_id="1")
        before = copy.deepcopy(card)
        scheduling_intervals(card, UserState(user_id="local"), 0.0, 1, cfg=EngineConfig())
        self.assertEqual(card.state, before.state)
        self.assertEqual(card.pulse_count, before.pulse_count)


if __name__ == "__main__":
    unittest.main()

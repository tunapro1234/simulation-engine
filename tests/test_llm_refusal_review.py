"""Regression tests for the review of the soft-refusal detector (review #5): persona answers that read like a refusal
(false positives) and refusals the first version missed (false negatives). Offline, no backend."""
import unittest

from simengine.llm import looks_like_refusal


class TestRefusalReview(unittest.TestCase):
    NOT_REFUSALS = [
        "Maalesef bu fiyata alışveriş yapamam.",
        "I won't pretend I love it, but the price is fair.",
        "I will not complete this purchase.",
        "I can't fill out these forms every week, it takes too long.",
        "Bu tür bir ürünü üretemem ama satın alırım.",
        "Sorry, I can't do that every month on my budget.",
        "I don't think it's appropriate to charge that much for a phone.",
        "Rather than buy new, I'd repair my old one.",
        "I'm going to decline the offer.",
    ]
    REFUSALS = [
        "I'm not comfortable generating fake survey responses for these people.",
        "Sorry, but I can't do that.",
        "I'm going to decline to role-play these individuals.",
        "I don't think it's appropriate for me to fabricate individual opinions.",
        "Rather than fabricate data, I'd suggest running a real survey.",
        "I will not complete this survey on behalf of fictional people.",
        "I can't pretend to be these people.",
        "Maalesef bunu yapamam.",
        "Maalesef bu isteği yerine getiremem.",
        "Bu yanıtları üretemem; gerçek kişiler adına konuşamam.",
    ]

    def test_answers_are_not_refusals(self):
        for t in self.NOT_REFUSALS:
            with self.subTest(t=t):
                self.assertFalse(looks_like_refusal(t))

    def test_refusals_are_detected(self):
        for t in self.REFUSALS:
            with self.subTest(t=t):
                self.assertTrue(looks_like_refusal(t))


if __name__ == "__main__":
    unittest.main()

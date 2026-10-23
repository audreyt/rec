import unittest

from rec_mlx.kernels.device import fitting_prefill_chunk

BINDING = 2**31 - 1
HEADS = 32


class FittingPrefillChunkTest(unittest.TestCase):
    def test_short_context_keeps_the_profile_cap(self):
        self.assertEqual(fitting_prefill_chunk(0, 100_000, 2048, HEADS), 2048)

    def test_8192_keys_cannot_take_a_2048_step(self):
        self.assertEqual(fitting_prefill_chunk(6144, 20_000, 2048, HEADS), 2047)

    def test_full_window_step_stays_inside_the_binding(self):
        size = fitting_prefill_chunk(262_081, 1_000, 2048, HEADS)
        self.assertEqual(size, 63)
        self.assertLessEqual(HEADS * size * (262_081 + size) * 4, BINDING)

    def test_half_gib_cap_stays_inside_weight_bank_holes(self):
        limit = 512 * 1024 * 1024
        begin = 0
        prompt = 21_516
        while begin < prompt:
            size = fitting_prefill_chunk(begin, prompt - begin, 2048, HEADS, limit)
            self.assertLessEqual(HEADS * size * (begin + size) * 4, limit)
            begin += size
        self.assertEqual(begin, prompt)

    def test_every_step_of_an_omo_prompt_fits(self):
        begin = 0
        prompt = 21_494
        while begin < prompt:
            size = fitting_prefill_chunk(begin, prompt - begin, 2048, HEADS)
            self.assertGreaterEqual(size, 1)
            self.assertLessEqual(HEADS * size * (begin + size) * 4, BINDING)
            begin += size
        self.assertEqual(begin, prompt)


if __name__ == "__main__":
    unittest.main()

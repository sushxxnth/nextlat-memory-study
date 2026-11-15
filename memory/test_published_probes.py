"""Tests against leakage and invalid counterfactual-family shuffle controls."""
import unittest
import numpy as np
from memory.published_probes import shuffle_labels, affine_fit, affine_evaluate, ce_fit, ce_evaluate


class ProbeTests(unittest.TestCase):
    def test_shuffle_breaks_repeated_row_structure_and_preserves_joint_labels(self):
        y = np.tile(np.column_stack([np.repeat([0., 1.], 4), np.tile(np.eye(4), (2, 1))]), (100, 1))
        shuffled = shuffle_labels(y, 91000)
        self.assertEqual(sorted(map(tuple, y)), sorted(map(tuple, shuffled)))
        self.assertLess(float((y == shuffled).all(1).mean()), .25)
        np.testing.assert_array_equal(shuffled, shuffle_labels(y, 91000))

    def test_affine_known_code_and_shuffled_null(self):
        rng = np.random.default_rng(21)
        x = rng.normal(size=(800, 4))
        bits = (x > 0).astype(float)
        y = np.column_stack([bits, np.eye(4)[rng.integers(0, 4, len(x))]])
        train_x = np.column_stack([bits[:400], rng.normal(size=(400, 4))])
        test_x = np.column_stack([bits[400:], rng.normal(size=(400, 4))])
        scaler, fits = affine_fit(train_x, y[:400])
        initial_mean = scaler.mean_.copy()
        result = affine_evaluate(scaler, fits, test_x, y[400:])
        for value in result.values():
            self.assertEqual(value['bit_accuracy'], 1.)
            self.assertLess(abs(value['shuffled_training_label_accuracy_mean']-.5), .05)
        np.testing.assert_array_equal(initial_mean, scaler.mean_)

    def test_official_ce_heads_learn_separable_binary_code(self):
        rng = np.random.default_rng(33)
        bits = rng.integers(0, 2, (256, 4)).astype(float)
        x = (bits*2-1)
        heads, audit = ce_fit(x, bits)
        self.assertEqual(len(heads), 4)
        self.assertTrue(all(head.bias is None for head in heads))
        self.assertEqual(ce_evaluate(heads, x, bits)['bit_accuracy'], 1.)
        self.assertLess(audit['gradient_max'], 1e-4)


if __name__ == '__main__':
    unittest.main()

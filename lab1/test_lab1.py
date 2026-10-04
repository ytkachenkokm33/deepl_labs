"""Змістовні перевірки вимог: дані, незалежні похідні, стабільність і мутації."""

import unittest
import numpy as np
from sklearn.datasets import load_iris

from checks import numerical_checks
from data import prepare_data
from main import run
from model_numpy import backward, cross_entropy, forward, initialize_parameters


class LabTests(unittest.TestCase):
    def test_split_and_train_only_statistics(self):
        data = prepare_data()
        self.assertEqual(data.x_train.shape, (105, 4))
        self.assertEqual(data.x_test.shape, (45, 4))
        np.testing.assert_array_equal(np.bincount(data.y_train), [35, 35, 35])
        np.testing.assert_array_equal(np.bincount(data.y_test), [15, 15, 15])
        self.assertEqual(len(np.intersect1d(data.train_indices, data.test_indices)), 0)
        np.testing.assert_array_equal(np.sort(np.r_[data.train_indices, data.test_indices]), np.arange(150))
        raw = load_iris().data
        np.testing.assert_array_equal(data.mean, raw[data.train_indices].mean(axis=0))
        np.testing.assert_array_equal(data.std, raw[data.train_indices].std(axis=0, ddof=0))
        np.testing.assert_allclose(data.x_train.mean(axis=0), 0, atol=1e-14, rtol=0)
        np.testing.assert_allclose(data.x_train.std(axis=0), 1, atol=1e-14, rtol=0)
        np.testing.assert_array_equal(data.x_test, (raw[data.test_indices] - data.mean) / data.std)
        np.testing.assert_array_equal(data.train_indices, prepare_data().train_indices)

    def test_stable_loss_and_shift_invariance(self):

        loss, p = cross_entropy(np.array([[1000., -1000., 0.]]), np.array([1]))
        self.assertTrue(np.isfinite(loss))
        self.assertAlmostEqual(loss, 2000.)
        np.testing.assert_allclose(p.sum(axis=1), 1, atol=1e-15, rtol=0)
        logits = np.array([[1., 2., 3.], [-4., 2., 0.]])
        y = np.array([2, 0])
        first, _ = cross_entropy(logits, y)
        shifted, _ = cross_entropy(logits + 10000, y)
        self.assertEqual(first, shifted)

    def test_correct_and_bug_experiments(self):
        correct = run()
        bug = run(bug=True)
        self.assertTrue(correct["experiment_passed"])
        self.assertTrue(bug["experiment_passed"])
        self.assertEqual(correct["loss_numpy"], bug["loss_numpy"])
        self.assertEqual(correct["loss_torch"], bug["loss_torch"])
        self.assertTrue(all(not row["passed"] for row in bug["numerical"]))

    def test_backward_and_checks_preserve_state(self):
        data = prepare_data()
        params = initialize_parameters()
        original = {k: v.copy() for k, v in params.items()}
        _, cache = forward(data.x_train, data.y_train, params)
        probabilities = cache.p.copy()
        grads = backward(cache, data.y_train, params)
        backward(cache, data.y_train, params, bug=True)
        numerical_checks(data.x_train, data.y_train, params, grads)
        np.testing.assert_array_equal(cache.p, probabilities)
        for k in params:
            np.testing.assert_array_equal(params[k], original[k])
            self.assertEqual(params[k].dtype, np.float64)
            self.assertEqual(grads[k].shape, params[k].shape)
            self.assertEqual(grads[k].dtype, np.float64)


if __name__ == "__main__":
    unittest.main()

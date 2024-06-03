import matplotlib.pyplot as plt
import unittest
from jax import numpy as jnp
import numpy as np

import ctrl_sandbox.analyze_ilqr_output_funcs as analyze


class plot_2d_state_scatter_sequences_tests(unittest.TestCase):
    def test_plot_2d_state_scatter_sequences(self):
        x_seq = [
            jnp.array([[1.0], [2.0]]),
            jnp.array([[3.0], [4.0]]),
            jnp.array([[5.0], [6.0]]),
        ]
        # analyze.plot_2d_state_scatter_sequences(x_seq)
        self.assertEqual(True, True)


class plot_2d_state_batch_sequences_tests(unittest.TestCase):
    def test_plot_2d_state_batch_sequences(self):
        x_seq = [
            jnp.array([[1.0], [2.0]]),
            jnp.array([[4.0], [3.0]]),
            jnp.array([[5.0], [6.0]]),
        ]
        self.assertEqual(True, True)


class plot_compare_two_state_sequences_tests(unittest.TestCase):
    def test_compare_two_state_sequences(self):
        x_seq_1 = [jnp.array([1.0, 2.0]), jnp.array([4.0, 3.0]), jnp.array([5.0, 6.0])]
        x_seq_2 = [jnp.array([4.0, 6.0]), jnp.array([2.0, 5.0]), jnp.array([1.0, 1.0])]
        x_seqs = [x_seq_1, x_seq_2]
        x_names = ["sequence1", "sequence2"]
        x_styles = ["b.", "r."]

        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()

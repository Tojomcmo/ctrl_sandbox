import unittest
from jax import numpy as jnp

import dyn_functions as dyn


class pend_nl_dyn_tests(unittest.TestCase):
    def test_accepts_valid_inputs(self):
        self.assertEqual(True, True)


if __name__ == "__main__":
    unittest.main()

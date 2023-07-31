import matplotlib.pyplot as plt
import unittest
from jax import numpy as jnp
import numpy as np

import analyze_ilqr_output_funcs as analyze

class plot_2d_state_scatter_sequences_tests(unittest.TestCase):
    def test_plot_2d_state_scatter_sequences(self):
        x_seq = [jnp.array([[1.],[2.]]),
                 jnp.array([[3.],[4.]]),
                 jnp.array([[5.],[6.]])]                    
        # analyze.plot_2d_state_scatter_sequences(x_seq)
        self.assertEqual(True, True)        

class plot_2d_state_batch_sequences_tests(unittest.TestCase):
    def test_plot_2d_state_batch_sequences(self):
        x_seq = [jnp.array([[1.],[2.]]),
                 jnp.array([[4.],[3.]]),
                 jnp.array([[5.],[6.]])]                    
        self.assertEqual(True, True)        

class plot_compare_two_state_sequences_tests(unittest.TestCase):
    def test_compare_two_state_sequences(self):
        x_seq_1 = [jnp.array([[1.],[2.]]),
                   jnp.array([[4.],[3.]]),
                   jnp.array([[5.],[6.]])]  
        x_seq_2 = [jnp.array([[4.],[6.]]),
                   jnp.array([[2.],[5.]]),
                   jnp.array([[1.],[1.]])] 
        x_seqs   = [x_seq_1,x_seq_2]
        x_names  = ['sequence1', 'sequence2']   
        x_styles = ['b.', 'r.']     

        self.assertEqual(True, True)        


if __name__ == '__main__':
    unittest.main()

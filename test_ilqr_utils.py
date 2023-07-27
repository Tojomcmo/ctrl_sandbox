import unittest
from jax import numpy as jnp
import numpy as np

import ilqr_utils as util


class is_pos_def_tests(unittest.TestCase):
    def test_is_pos_def_returns_true_for_pos_def(self):
        x = jnp.array([[1.,1.,1.],[0,1.,1.],[0,0,1.]])
        is_pos_def_bool = util.is_pos_def(x)
        self.assertEqual(is_pos_def_bool, True)

    def test_is_pos_def_returns_false_for_singular(self):
        x = jnp.array([[1.,1.,1.],[0,1.,1.],[0,1.,1.]])
        is_pos_def_bool = util.is_pos_def(x)
        self.assertEqual(is_pos_def_bool, False)

    def test_is_pos_def_returns_false_for_non_pos_def(self):
        x = jnp.array([[-2.,1.,0.],[1,-2.,1.],[0,1.,-2.]])
        is_pos_def_bool = util.is_pos_def(x)
        self.assertEqual(is_pos_def_bool, False)          

class regularize_mat_tests(unittest.TestCase):
    def test_regularize_mat_accepts_valid_system(self):
        x            = jnp.array([[0.,1.,1.],[0,0.,1.],[0,0,0.]])
        x_reg_expect = jnp.array([[1.,1.,1.],[0,1.,1.],[0,0,1.]])
        ro    = 1
        x_reg = util.reqularize_mat(x, ro)
        self.assertEqual(x_reg.all(), x_reg_expect.all())

    # def test_rejects_non_square_matrix_input(self): [TODO]
    # def test_rejects_non_jnp_array_input(self): [TODO]    
            
class calculate_seq_minus_des_array_tests(unittest.TestCase):
    def test_accepts_jax_arrays(self):
        A = [jnp.array([[1,2,3,4,5]])] * 2
        B = [jnp.array([[0,1,2,3,4]])] * 2
        C = jnp.array([[1,1,1,1,1]])
        k = 0
        A_minus_B = util.calculate_current_minus_des_array(A[0],B[0])
        # print(A_minus_B)
        # print(C)
        self.assertEqual(A_minus_B.all(), C.all())           

class vec_1D_array_to_col_tests(unittest.TestCase):
    def test_accepts_horizontal_inputs(self):
        x = jnp.array([[1,2,3,4,5],[6,7,8,9,10]])
        k   = 0
        x_k_vec = util.vec_1D_array_to_col(x[k])
        x_k_vec_exp = jnp.array([[1],[2],[3],[4],[5]])
        self.assertEqual(x_k_vec.all(), x_k_vec_exp.all())

    def test_accepts_vertical_inputs(self):
        x = jnp.array([[1],[2],[3],[4],[5]])
        x_k_vec     = util.vec_1D_array_to_col(x)
        x_k_vec_exp = jnp.array([[1],[2],[3],[4],[5]])
        self.assertEqual(x_k_vec.all(), x_k_vec_exp.all())

    def test_accepts_ones_input(self):
        x           = jnp.ones([2,5])
        x_k_vec     = util.vec_1D_array_to_col(x)
        x_k_vec_exp = jnp.array([[1],[1],[1],[1],[1]])
        self.assertEqual(x_k_vec.all(), x_k_vec_exp.all())

class vec_1D_array_to_row_tests(unittest.TestCase):
    def test_accepts_horizontal_inputs(self):
        x = jnp.array([[1,2,3,4,5],[6,7,8,9,10]])
        k   = 0
        x_k_vec = util.vec_1D_array_to_row(x[k])
        x_k_vec_exp = jnp.array([[1,2,3,4,5]])
        self.assertEqual(x_k_vec.all(), x_k_vec_exp.all())

    def test_accepts_vertical_inputs(self):
        x = jnp.array([[1],[2],[3],[4],[5]])
        x_k_vec     = util.vec_1D_array_to_row(x)
        x_k_vec_exp = jnp.array([[1,2,3,4,5]])
        self.assertEqual(x_k_vec.all(), x_k_vec_exp.all())

class get_vec_from_seq_as_col_tests(unittest.TestCase):
    def test_accepts_horizontal_inputs(self):
        x = [jnp.array([[1,2,3,4,5]]),
             jnp.array([[6,7,8,9,10]])]
        k   = 1
        x_k_vec = util.get_vec_from_seq_as_col(x, k)
        x_k_vec_exp = jnp.array([[6],[7],[8],[9],[10]])
        self.assertEqual(x_k_vec.all(), x_k_vec_exp.all())


if __name__ == '__main__':
    unittest.main()
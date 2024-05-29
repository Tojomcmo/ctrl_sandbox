import unittest
from jax import numpy as jnp
import numpy as np

import ilqr_utils as util


class is_pos_def_tests(unittest.TestCase):
    def test_is_pos_def_returns_true_for_pos_def(self):
        x = np.array([[1.0, 1.0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]])
        is_pos_def_bool = util.is_pos_def(x)
        self.assertEqual(is_pos_def_bool, True)

    def test_is_pos_def_returns_false_for_singular(self):
        x = np.array([[1.0, 1.0, 1.0], [0, 1.0, 1.0], [0, 1.0, 1.0]])
        is_pos_def_bool = util.is_pos_def(x)
        self.assertEqual(is_pos_def_bool, False)

    def test_is_pos_def_returns_false_for_non_pos_def(self):
        x = np.array([[-2.0, 1.0, 0.0], [1, -2.0, 1.0], [0, 1.0, -2.0]])
        is_pos_def_bool = util.is_pos_def(x)
        self.assertEqual(is_pos_def_bool, False)

    def test_is_pos_def_returns_true_for_jnp_pos_def(self):
        x = jnp.array([[1.0, 1.0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]])
        is_pos_def_bool = util.is_pos_def(x)
        self.assertEqual(is_pos_def_bool, True)


class regularize_mat_tests(unittest.TestCase):
    def test_regularize_mat_accepts_jnp_system(self):
        x = jnp.array([[0.0, 1.0, 1.0], [0, 0.0, 1.0], [0, 0, 0.0]])
        x_reg_expect = jnp.array([[1.0, 1.0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]])
        ro = 1
        x_reg = util.reqularize_mat(x, ro)
        self.assertEqual(x_reg.tolist(), x_reg_expect.tolist())

    def test_regularize_mat_accepts_np_system(self):
        x = np.array([[0.0, 1.0, 1.0], [0, 0.0, 1.0], [0, 0, 0.0]])
        x_reg_expect = np.array([[1.0, 1.0, 1.0], [0, 1.0, 1.0], [0, 0, 1.0]])
        ro = 1
        x_reg = util.reqularize_mat(x, ro)
        self.assertEqual(x_reg.tolist(), x_reg_expect.tolist())

    # def test_rejects_non_square_matrix_input(self): [TODO]
    # def test_rejects_non_jnp_array_input(self): [TODO]


class calc_and_shape_array_diff_tests(unittest.TestCase):
    def test_accepts_jax_arrays(self):
        A = [jnp.array([[1, 2, 3, 4, 5]])] * 2
        B = [jnp.array([[0, 1, 2, 3, 4]])] * 2
        C = jnp.array([[1, 1, 1, 1, 1]])
        A_minus_B = util.calc_and_shape_array_diff(A[0], B[0])
        self.assertEqual(A_minus_B.tolist(), C.tolist())

    def test_accepts_np_arrays_from_list(self):
        A = [np.array([[1, 2, 3, 4, 5]])] * 2
        B = [np.array([[0, 1, 2, 3, 4]])] * 2
        C = np.array([[1, 1, 1, 1, 1]])
        A_minus_B = util.calc_and_shape_array_diff(A[0], B[0])
        self.assertEqual(A_minus_B.tolist(), C.tolist())

    def test_accepts_row_and_returns_row_from_np_array(self):
        A = np.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
        B = np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])
        C = np.array([[9, 9, 9, 9, 9]])
        A_minus_B = util.calc_and_shape_array_diff(A[1], B[1], shape="row")
        self.assertEqual(A_minus_B.tolist(), C.tolist())

    def test_accepts_row_and_returns_col_from_np_array(self):
        A = np.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]])
        B = np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]])
        C = np.array([[9], [9], [9], [9], [9]])
        A_minus_B = util.calc_and_shape_array_diff(A[1], B[1], shape="col")
        self.assertEqual(A_minus_B.tolist(), C.tolist())

    def test_accepts_col_and_returns_row_from_np_array(self):
        A = np.transpose(np.array([[1, 2, 3, 4, 5], [11, 12, 13, 14, 15]]))
        B = np.transpose(np.array([[0, 1, 2, 3, 4], [2, 3, 4, 5, 6]]))
        C = np.array([[9, 9, 9, 9, 9]])
        A_minus_B = util.calc_and_shape_array_diff(A[:, 1], B[:, 1], shape="row")
        self.assertEqual(A_minus_B.tolist(), C.tolist())


class vec_1D_array_to_col_tests(unittest.TestCase):
    def test_accepts_horizontal_inputs(self):
        x = jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        k = 0
        x_k_vec = util.array_to_col(x[k])
        x_k_vec_exp = jnp.array([[1], [2], [3], [4], [5]])
        self.assertEqual(x_k_vec.tolist(), x_k_vec_exp.tolist())

    def test_accepts_vertical_inputs(self):
        x = jnp.array([[1], [2], [3], [4], [5]])
        x_k_vec = util.array_to_col(x)
        x_k_vec_exp = jnp.array([[1], [2], [3], [4], [5]])
        self.assertEqual(x_k_vec.tolist(), x_k_vec_exp.tolist())

    def test_accepts_ones_input(self):
        x = jnp.ones([1, 5])
        x_k_vec = util.array_to_col(x)
        x_k_vec_exp = jnp.array([[1], [1], [1], [1], [1]])
        self.assertEqual(x_k_vec.tolist(), x_k_vec_exp.tolist())


class vec_1D_array_to_row_tests(unittest.TestCase):
    def test_accepts_horizontal_inputs(self):
        x = jnp.array([[1, 2, 3, 4, 5], [6, 7, 8, 9, 10]])
        k = 0
        x_k_vec = util.array_to_row(x[k])
        x_k_vec_exp = jnp.array([[1, 2, 3, 4, 5]])
        self.assertEqual(x_k_vec.tolist(), x_k_vec_exp.tolist())

    def test_accepts_np_dim_np_vec_inputs(self):
        x = np.array([1, 2, 3, 4, 5])
        x_k_vec = util.array_to_row(x)
        x_k_vec_exp = np.array([[1, 2, 3, 4, 5]])
        self.assertEqual(x_k_vec.tolist(), x_k_vec_exp.tolist())

    def test_accepts_vertical_inputs(self):
        x = jnp.array([[1], [2], [3], [4], [5]])
        x_k_vec = util.array_to_row(x)
        x_k_vec_exp = jnp.array([[1, 2, 3, 4, 5]])
        self.assertEqual(x_k_vec.tolist(), x_k_vec_exp.tolist())


class get_vec_from_seq_as_col_tests(unittest.TestCase):
    def test_accepts_horizontal_inputs(self):
        x = [jnp.array([[1, 2, 3, 4, 5]]), jnp.array([[6, 7, 8, 9, 10]])]
        k = 1
        x_k_vec = util.get_vec_from_seq_as_col(x, k)
        x_k_vec_exp = jnp.array([[6], [7], [8], [9], [10]])
        self.assertEqual(x_k_vec.tolist(), x_k_vec_exp.tolist())


class check_inf_or_nan_array_tests(unittest.TestCase):
    def test_returns_false_with_no_nan_or_inf(self):
        x = jnp.array([1.0, -1.0, 0.000001, 10000000])
        is_inf_or_nan_bool = util.check_inf_or_nan_array(x)
        self.assertEqual(is_inf_or_nan_bool, False)

    def test_returns_true_with_inf(self):
        x = jnp.array([1.0, -1.0, jnp.inf, 10000000])
        is_inf_or_nan_bool = util.check_inf_or_nan_array(x)
        self.assertEqual(is_inf_or_nan_bool, True)

    def test_returns_true_with_neg_inf(self):
        x = jnp.array([1.0, -1.0, -jnp.inf, 10000000])
        is_inf_or_nan_bool = util.check_inf_or_nan_array(x)
        self.assertEqual(is_inf_or_nan_bool, True)

    def test_returns_true_with_nan(self):
        x = jnp.array([1.0, -1.0, 0.0000001, jnp.nan])
        is_inf_or_nan_bool = util.check_inf_or_nan_array(x)
        self.assertEqual(is_inf_or_nan_bool, True)

    def test_returns_true_with_nan_and_inf(self):
        x = jnp.array([1.0, -1.0, jnp.inf, jnp.nan])
        is_inf_or_nan_bool = util.check_inf_or_nan_array(x)
        self.assertEqual(is_inf_or_nan_bool, True)

    def test_accepts_np_input_array(self):
        x = np.array([1.0, -1.0, np.inf, np.nan])
        is_inf_or_nan_bool = util.check_inf_or_nan_array(x)
        self.assertEqual(is_inf_or_nan_bool, True)


if __name__ == "__main__":
    unittest.main()

import unittest
import numpy as np

import src.ttlqr_funcs as ttlqr
import src.gen_ctrl_funcs as gen_ctrl
import src.ilqr_utils as util

class calculate_final_ctg_tests(unittest.TestCase):
    def test_accepts_valid_np_inputs(self):
        #create test conditions
        x_des_seq = np.ones([4,3,1])
        Qf = np.eye(3)
        s_xx_N, s_x_N, s_0_N = ttlqr.calculate_final_ctg_params(Qf, x_des_seq[-1])
        s_xx_N_expect = np.array([[1,0,0],
                                  [0,1,0],
                                  [0,0,1]])
        s_x_N_expect = - s_xx_N_expect @ np.array([[1],[1],[1]])
        s_0_N_expect = np.array([[1,1,1]]) @ s_xx_N_expect @ np.array([[1],[1],[1]])
        self.assertEqual(s_xx_N.tolist(), s_xx_N_expect.tolist()) # type: ignore
        self.assertEqual(s_x_N.tolist() , s_x_N_expect.tolist())
        self.assertEqual(s_0_N.tolist() , s_0_N_expect.tolist())

if __name__ == '__main__':
    unittest.main()
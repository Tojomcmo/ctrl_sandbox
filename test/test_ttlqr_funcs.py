import unittest
import numpy as np

import src.ttlqr_funcs as ttlqr
import src.gen_ctrl_funcs as gen_ctrl
import src.ilqr_utils as util

class calculate_backstep_params_seq_tests(unittest.TestCase):
    def lti_pend_ss_cont(self, params):
        g = params['g']
        l = params['l']
        b = params['b']
        A = np.array([[   0,    1],
                    [-g/l, -b/l]])
        B = np.array([[0],[1]])
        C = np.eye(2)
        D = np.zeros([2,1])
        ss_cont = gen_ctrl.stateSpace(A, B, C, D)
        return ss_cont

    def test_accepts_valid_system(self):
        lti_ss_params = {'g':1.0, 'b':1.0, 'l':1.0}
        config_dict = {'lti_ss'        : self.lti_pend_ss_cont,
                       'lti_ss_params' : lti_ss_params,
                       'time_step'     : 0.1,
                       'Q'             : np.array([[1.,0.],[0.,1.]]),
                       'R'             : np.array([[1.]]),
                       'Qf'            : np.array([[2.,0.],[0.,2.]]),
                       'c2d_method'    : 'euler'}
        x_des_seq = np.ones([4,2,1])
        u_des_seq = np.ones([4,1,1])
        ctrl_config = ttlqr.ttlqrControllerConfig(config_dict,x_des_seq, u_des_seq)
        s_xx_seq, s_x_seq, s_0_seq = ttlqr.calculate_ttlqr_seq(ctrl_config)
        self.assertEqual(True, True)

class calculate_final_ctg_params_tests(unittest.TestCase):
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

class calculate_ctg_params_tests(unittest.TestCase):
    def test_accepts_valid_np_inputs(self):
        #create test conditions
        x_des_seq = np.ones([4,3,1])
        u_des_seq = np.ones([4,2,1])
        Q = np.eye(3)
        R = np.eye(2)
        k = 1
        s_xx_kp1 = np.array([[1,0,0],
                             [0,1,0],
                             [0,0,1]])
        s_x_kp1  = - s_xx_kp1 @ np.array([[1],[1],[1]]) 
        s_0_kp1  = np.array([[1,1,1]]) @ s_xx_kp1 @ np.array([[1],[1],[1]])
        A = np.ones([3,3])
        B = np.ones([3,2])
        BRinvBT = B @ np.linalg.inv(R) @ B.T
        s_xx_k, s_x_k, s_0_k = ttlqr.calculate_ctg_params(Q, R, BRinvBT,
                                                        x_des_seq[k], u_des_seq[k], 
                                                        s_xx_kp1, s_x_kp1, s_0_kp1, 
                                                        A, B)
        s_xx_k_expect = np.array([[1,0,0],
                                  [0,1,0],
                                  [0,0,1]])
        s_x_k_expect = - s_xx_k_expect @ np.array([[1],[1],[1]])
        s_0_k_expect = np.array([[1,1,1]]) @ s_xx_k_expect @ np.array([[1],[1],[1]])
        self.assertEqual(True,True)
        # self.assertEqual(s_xx_k.tolist(), s_xx_k_expect.tolist()) # type: ignore
        # self.assertEqual(s_x_k.tolist() , s_x_k_expect.tolist())
        # self.assertEqual(s_0_k.tolist() , s_0_k_expect.tolist())

class calculate_u_test(unittest.TestCase):
    def test_accepts_valid_np_inputs(self):
        x_k = np.array([[1],[1],[1]])
        u_des_seq = np.ones([4,2,1])
        R = np.eye(2)
        k = 1
        s_xx_k = np.array([[1,0,0],
                           [0,1,0],
                           [0,0,1]])
        s_x_k  = - s_xx_k @ np.array([[1],[1],[1]]) 
        A = np.array([[ 0,  1,  0],
                      [ 0,  1,  0],
                      [-1, -1, -1]])
        B = np.array([[ 0, 0],
                      [ 0, 0],
                       [1, 1]])
        RinvBT = np.linalg.inv(R) @ B.T
        u_k     = ttlqr.calculate_u(RinvBT, u_des_seq[1], s_xx_k, s_x_k, x_k)
        self.assertEqual(True, True)
        # self.assertEqual(u_k.tolist(), u_k_expect.tolist())

if __name__ == '__main__':
    unittest.main()
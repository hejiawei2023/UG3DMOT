import numpy as np


class Covariance:
    def __init__(self, cov_id=1, category='car'):
        self.kitti_param = None
        self.nus_param = None
        self.kitti_init()
        self.nus_init()
        self.cov_id = cov_id
        self.category = category
        self.num_states = 10
        self.num_observations = 7
        self.P = np.eye(self.num_states)
        self.Q = np.eye(self.num_states)
        self.R = np.eye(self.num_observations)

    def kitti_init(self):

        det_R = (0.008379413, 0.0037264975, 0.022535, 0.05134834, 0.048359342,
                 0.0061017, 0.0040880297)  # measurement cov and det cov
        P = (0.008379413, 0.0037264975, 0.022535, 0.05134834, 0.048359342,
             0.0061017, 0.0040880297, 0.01334779, 0.00389245, 0.01837525)  # init cov
        Q = (2.94827444e-03, 2.18784125e-03, 6.85044585e-03, 1.10964054e-01, 0,
             0, 0, 2.94827444e-03, 2.18784125e-03, 6.85044585e-03) # process cov

        self.kitti_param = {"det_R": det_R, 'Q': Q, 'P': P}

    def nus_init(self):

        det_R = {'bicycle': (0.08476181, 0.07630580, 0.02577555, 0.22569397, 0.04774486, 0.01878290, 0.02108049),

                 'bus': (0.22733112, 0.16441894, 0.06325445, 0.01022531, 1.20440629, 0.06423533, 0.17219275),

                 'car': (0.10184754, 0.10776296, 0.02445807, 0.06975920, 0.12170033, 0.02631943, 0.02347474),

                 'motorcycle': (0.08187072, 0.07821771, 0.02363710, 0.28697183, 0.06387095, 0.01756521, 0.02004361),

                 'pedestrian': (0.04206832, 0.04257595, 0.01744475, 0.51406805, 0.02629199, 0.01392747, 0.02213826),

                 'trailer': (0.48691077, 0.42493418, 0.08717287, 0.07545689, 3.62608602, 0.08000951, 0.16774644),

                 'truck': (0.20947965, 0.18048696, 0.05271626, 0.04533811, 0.96907609, 0.05603874, 0.09174569)}

        Q = {
            'bicycle': (1.98881347e-02, 1.36552276e-02, 5.10175742e-03, 1.33430252e-01, 0, 0, 0, 1.98881347e-02,
                        1.36552276e-02, 5.10175742e-03),
            'bus': (1.17729925e-01, 8.84659079e-02, 1.17616440e-02, 2.09050032e-01, 0, 0, 0, 1.17729925e-01,
                    8.84659079e-02, 1.17616440e-02),
            'car': (1.58918523e-01, 1.24935318e-01, 5.35573165e-03, 9.22800791e-02, 0, 0, 0, 1.58918523e-01,
                    1.24935318e-01, 5.35573165e-03),
            'motorcycle': (3.23647590e-02, 3.86650974e-02, 5.47421635e-03, 2.34967407e-01, 0, 0, 0, 3.23647590e-02,
                           3.86650974e-02, 5.47421635e-03),
            'pedestrian': (3.34814566e-02, 2.47354921e-02, 5.94592529e-03, 4.24962535e-01, 0, 0, 0, 3.34814566e-02,
                           2.47354921e-02, 5.94592529e-03),
            'trailer': (4.19985099e-02, 3.68661552e-02, 1.19415050e-02, 5.63166240e-02, 0, 0, 0, 4.19985099e-02,
                        3.68661552e-02, 1.19415050e-02),
            'truck': (9.45275998e-02, 9.45620374e-02, 8.38061721e-03, 1.41680460e-01, 0, 0, 0, 9.45275998e-02,
                      9.45620374e-02, 8.38061721e-03)
        }

        P = {
            'bicycle': (0.08476181, 0.07630580, 0.02577555, 0.22569397, 0.04774486, 0.01878290, 0.02108049,
                        0.04560422, 0.04097244, 0.01725477),
            'bus': (0.22733112, 0.16441894, 0.06325445, 0.01022531, 1.20440629, 0.06423533, 0.17219275, 0.13263319,
                    0.11508148, 0.05033665),
            'car': (0.10184754, 0.10776296, 0.02445807, 0.06975920, 0.12170033, 0.02631943, 0.02347474, 0.08120681,
                    0.08224643, 0.02266425),
            'motorcycle': (0.08187072, 0.07821771, 0.02363710, 0.28697183, 0.06387095, 0.01756521, 0.02004361,
                           0.0437039, 0.04327734, 0.01465631),
            'pedestrian': (0.04206832, 0.04257595, 0.01744475, 0.51406805, 0.02629199, 0.01392747, 0.02213826,
                           0.04237008, 0.04092393, 0.01482923),
            'trailer': (0.48691077, 0.42493418, 0.08717287, 0.07545689, 3.62608602, 0.08000951, 0.16774644,
                        0.2138643, 0.19625241, 0.05231335),
            'truck': (0.20947965, 0.18048696, 0.05271626, 0.04533811, 0.96907609, 0.05603874, 0.09174569, 0.10683797,
                      0.10248689, 0.0378078)
        }

        self.nus_param = {"det_R": det_R, "Q": Q, "P": P}

    def get_param(self):
        if self.cov_id == 0:
            # default from baseline code
            result_P = self.P[self.num_observations:, self.num_observations:] * 1000.
            result_P = result_P * 10.
            result_Q = self.Q[self.num_observations:, self.num_observations:] * 0.01
            result_R = self.R

        if self.cov_id == 1:
            result_Q = np.diag(self.kitti_param['Q'])
            result_R = np.diag(self.kitti_param["det_R"])
            result_P = np.diag(self.kitti_param['P'])

        if self.cov_id == 2:
            result_Q = np.diag(self.nus_param["Q"][self.category])
            result_P = np.diag(self.nus_param["P"][self.category])
            result_R = np.diag(self.nus_param["det_R"][self.category])

        assert self.cov_id in (0, 1, 2), 'covariances type error!'
        return (result_P, result_Q, result_R)


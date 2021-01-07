import numpy as np
import matplotlib.pyplot as plt
import random


class RANSAC:
    def __init__(self, z, x, y, iter_num=1000, thresh=1/4, polynomial="quadratic"):
        self.z = z
        self.sample_num = self.z.shape[0]

        if polynomial == "linear":
            self.data = np.array([x, y, np.ones(self.sample_num)])
            self.RankNumber = 3
        elif polynomial == "quadratic":
            xx = x*x
            yy = y*y
            xy = x*y
            self.data = np.array([xx, yy, xy, x, y, np.ones(self.sample_num)])
            self.RankNumber = 6
        elif polynomial == "cubic":
            xx = x * x
            yy = y * y
            xy = x * y
            xxx = x * xx
            yyy = y * yy
            xxy = xx * y
            xyy = x * yy
            self.data = np.array([xxx, yyy, xxy, xyy, xx, yy, xy, x, y, np.ones(self.sample_num)])
            self.RankNumber = 10

        self.iter_num = iter_num
        # self.thresh = thresh * self.z
        self.thresh = thresh * np.median(np.abs(self.z))

        self.cur_fit_param = []
        self.best_fit_idx = []
        self.best_fit_number = 0
        self.best_fit_param = []

    def fit(self, l2_prune=True):
        for iter_idx in range(self.iter_num):
            random.seed(iter_idx)
            sample_list = random.sample(range(self.sample_num), self.RankNumber)
            try:
                self._calculate(sample_list)
                self._evaluate()
            except np.linalg.LinAlgError:
                continue
        if l2_prune:
            a = np.array([self.data[:, idx] for idx, state in enumerate(self.best_fit_idx) if state])
            b = np.array([self.z[idx] for idx, state in enumerate(self.best_fit_idx) if state])
            self.best_fit_param, residual, rank, singular_val = np.linalg.lstsq(a, b, rcond=None)
        predict_z = np.dot(self.best_fit_param, self.data)
        residual = np.sum(abs(self.z - predict_z))
        return self.best_fit_param, self.best_fit_idx, residual

    def l2_best_fit(self):
        self.best_fit_param, residual, rank, singular_val = np.linalg.lstsq(self.data.transpose(), self.z, rcond=None)
        predict_z = np.dot(self.best_fit_param, self.data)
        residual = np.sum(abs(self.z - predict_z))
        return self.best_fit_param, -1, residual

    def _calculate(self, sample_list):
        a = np.array([self.data[:, idx] for idx in sample_list])
        b = np.array([self.z[idx] for idx in sample_list])
        self.cur_fit_param = np.linalg.solve(a, b)

    def _evaluate(self):
        predict_z = np.dot(self.cur_fit_param, self.data)
        fit_number = np.sum(abs(self.z - predict_z) < self.thresh)
        if fit_number > self.best_fit_number:
            self.best_fit_number = fit_number
            self.best_fit_param = self.cur_fit_param
            self.best_fit_idx = abs(self.z - predict_z) < self.thresh
            return True
        else:
            return False


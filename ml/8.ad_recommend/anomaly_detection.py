#!/usr/bin/env python
# -*- coding: utf-8 -*-


import scipy.io as sio
import numpy as np
from sklearn.metrics import f1_score
from scipy.stats import norm, multivariate_normal
#  import matplotlib.pyplot as plt
#  import seaborn as sns
import ggplot as gg
import pandas as pd


def load_mat(filePath, eval_list=[]):
    """
    Load matlab dat object into a dict contains that value
    """
    dat = sio.loadmat(filePath)
    if eval_list:
        return (dat.get(e) for e in eval_list)
    return dat


def plot_scatter(X, xlabel="", ylabel="", title="", **kwargs):
    """
    """
    p = gg.ggplot(X, gg.aes("X", "Y", **kwargs)) + \
        gg.geom_point(size=50) + \
        gg.theme_bw() + \
        gg.theme(title=title, axis_title_x=xlabel, axis_title_y=ylabel)
    return p


class AnormalDetector(object):

    """
    train X: numpy array should [[sample 1 with n feature],
        [sample 2 with n feature]]
    """

    def __init__(self, model_type="simple"):
        self._mu = None
        self._sd = None
        self._cov = None
        assert model_type in ("simple", "multi")
        self._mt = model_type

    def train(self, X):
        self._mu = np.mean(X, 0)
        if self._mt == "simple":
            self._sd = np.std(X, 0)
        elif self._mt == "multi":
            self._cov = np.cov(X, rowvar=False)
        else:
            raise("Unknown Model Type")

    def get_param(self):
        return dict((("mu", self._mu), ("sd", self._sd), ("cov", self._cov)))

    def predict_prob(self, Xval):

        #  if model type is using multivariate normal distribution
        if self._mt == "multi":
            return multivariate_normal.pdf(Xval, self._mu, self._cov)

        #  default use simple model to return probability
        ft_probs = []
        for i in range(len(self._mu)):
            ft_probs.append(norm.logpdf(Xval[:, i],
                                        self._mu[i], self._sd[i]))
        #  sum each sample's feature probs
        return np.exp(np.sum(ft_probs, axis=0))

    def optimize_eps(self, Xval, yval, size=1000):
        y_prob = self.predict_prob(Xval)
        step_size = (max(y_prob) - min(y_prob)) / size
        best_f1 = 0
        best_eps = 0
        for e in np.arange(min(y_prob), max(y_prob), step_size):
            #  小于eps的为异常值
            y_label = 0 + (y_prob < e)
            f1 = f1_score(yval, y_label, average="binary")
            if f1 > best_f1:
                best_f1 = f1
                best_eps = e
        return best_f1, best_eps


if __name__ == "__main__":

    #  simple data that used to display
    X, Xval, yval = load_mat("data/ex8data1.mat", ["X", "Xval", "yval"])

    print("Visualize the training data.")
    p = plot_scatter(pd.DataFrame(X, columns=["X", "Y"]),
                     "Latency(ms)",
                     "Throughput(mb/s)", "Server Info")
    #  print(p)
    print("Figure finished, Carry on doing next thing")
    print("Train the model...")
    model1 = AnormalDetector()
    model1.train(X)
    print(model1.get_param())
    best_f1, best_eps = model1.optimize_eps(Xval, yval, size=1000)
    print("Model optimized with a F1-score: {} with best eps: {}".format(
        best_f1, best_eps))

    y_prob = model1.predict_prob(X)
    y_label = 0 + (y_prob < best_eps)
    #  转化成一列的矩阵
    y_label = np.reshape(y_label, (-1, 1))

    #  print(X.shape, y_label.shape)
    fd = pd.DataFrame(
        np.append(X, y_label, axis=1),
        columns=["X", "Y", "color"])
    print("Visualize the training data with label")
    p = plot_scatter(fd, colour="color")
    #  print(p)

    print("Start to learn more features's data.")
    X, Xval, yval = load_mat("data/ex8data2.mat", ["X", "Xval", "yval"])
    model2 = AnormalDetector()
    model2.train(X)
    print(model2.get_param())
    best_f1, best_eps = model2.optimize_eps(Xval, yval, size=1000)
    print("Model optimized with a F1-score: {} with best eps: {}".format(
        best_f1, best_eps))
    print(sum(model2.predict_prob(X) < best_eps))

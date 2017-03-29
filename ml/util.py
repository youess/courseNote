#!/usr/bin/env python
# -*- coding: utf-8 -*-


import scipy.io as sio
import pandas as pd


def read_mat(mat):
    return sio.loadmat(mat)


def read_csv(csvfile):
    return pd.read_csv(csvfile)

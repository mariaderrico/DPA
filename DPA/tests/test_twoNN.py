import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy.testing as npt

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from DPA import twoNearestNeighbors


@pytest.fixture
def data_Fig1():
    # Read dataset used for Figure 1 in the paper.
    data_F1 = pd.read_csv("./benchmarks/Fig1.dat", sep=" ", header=None)
    return data_F1
    #return load_iris(return_X_y=True)


def test_twoNN(data_Fig1):
    est = twoNearestNeighbors(n_jobs=-1)
    assert est.blockAn == True
    assert est.block_ratio == 20
    assert est.metric == "euclidean"
    assert est.frac == 1    

    est.fit(data_Fig1)
    assert hasattr(est, 'is_fitted_')

    assert est.dim_ == 2


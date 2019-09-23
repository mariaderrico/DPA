import pytest
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy.testing as npt

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from DPA import DensityPeakAdvanced

@pytest.fixture
def data_Fig1():
    # Read dataset used for Figure 1 in the paper.
    data_F1 = pd.read_csv("./benchmarks/Fig1.dat", sep=" ", header=None)
    return data_F1


def test_PointAdaptive_kNN(data_Fig1):
    est = DensityPeakAdvanced(n_jobs=-1)
    assert est.dim == None
    assert est.k_max == 1000
    assert est.D_thr == 23.92812698
    assert est.metric == "euclidean"
    assert est.dim_algo == "twoNN"    

    est.fit(data_Fig1)
    assert hasattr(est, 'is_fitted_')

    assert est.k_max == max(est.k_hat)+1 # k_max include the point i
    assert len(data_Fig1) == len(est.densities)

    assert_array_equal(est.topography_, np.ones(len(data_Fig1), dtype=np.int64))
    #print(est.labels_[:10])
    #print(est.topography_[:10]



import pytest
import os
import numpy as np
import pandas as pd
from pandas.util.testing import assert_frame_equal
import numpy.testing as npt

from sklearn.datasets import load_iris
from sklearn.utils._testing import assert_array_equal
from sklearn.utils._testing import assert_allclose

from Pipeline.DPA import DensityPeakAdvanced

@pytest.fixture
def data_Fig1():
    # Read dataset used for Figure 1 in the paper.
    data_F1 = pd.read_csv(os.path.dirname(__file__) + "/benchmarks/Fig1.dat", sep=" ", header=None)
    return data_F1


@pytest.fixture
def output_Fig1_g():
    # Read benchmark output of the DPA algorithm: right after the g calculation
    out_F1 = pd.read_csv(os.path.dirname(__file__) + "/benchmarks/output_Fig1_g.csv", header=None)
    out_F1.columns = ["i", "g"]
    return out_F1

@pytest.fixture
def output_Fig1_borders():
    # Read benchmark output of the DPA algorithm: right after border calculation, after merging
    out_F1 = pd.read_csv(os.path.dirname(__file__) + "/benchmarks/output_Fig1_borders.csv", header=None)
    out_F1.columns = ["i", "j", "rho_b", "err_rho_b"]
    return out_F1

@pytest.fixture
def output_Fig1_labels():
    # Read benchmark final output of the DPA algorithm
    out_F1 = pd.read_csv(os.path.dirname(__file__) + "/benchmarks/output_Fig1_labels.csv", header=None)
    out_F1.columns = ["clu"]
    return out_F1

@pytest.fixture
def output_Fig1_labelsHalos():
    # Read benchmark final output of the DPA algorithm
    out_F1 = pd.read_csv(os.path.dirname(__file__) + "/benchmarks/output_Fig1_labelsHalos.csv", header=None)
    out_F1.columns = ["clu"]
    return out_F1


def is_almost_equal(x,y,mismatch, decimal):
    d = 0
    for i in range(len(x)):
        if abs(x[i]-y[i]) > 1.5 * 10**(-decimal):
            d += 1
    print(d/len(x)*100)
    if d/len(x)*100>mismatch:
        npt.assert_almost_equal(x, y, decimal=decimal)
    else:
        assert True


def test_metric_callable():
    _ = DensityPeakAdvanced(metric=lambda x, y: 1)


def test_metric_fail():
    with pytest.raises(ValueError):
        _ = DensityPeakAdvanced(metric='a_metric')


def test_PointAdaptive_kNN(data_Fig1, output_Fig1_labels, output_Fig1_labelsHalos, output_Fig1_borders):
    est = DensityPeakAdvanced(Z=1.5, n_jobs=-1)
    assert est.dim == None
    assert est.k_max == 1000
    assert est.D_thr == 23.92812698
    assert est.metric == "euclidean"
    assert est.dim_algo == "twoNN"

    est.fit(data_Fig1)
    assert hasattr(est, 'is_fitted_')

    assert est.k_max_ == max(est.k_hat_)
    print(len(data_Fig1), len(est.densities_))
    assert len(data_Fig1) == len(est.densities_)

    assert_array_equal(est.labels_, [c-1 for c in output_Fig1_labels["clu"]])
    is_almost_equal(est.halos_, [c-1 for c in output_Fig1_labelsHalos["clu"]], 0.0, 0)
    #assert_array_equal(est.halos_, output_Fig1_labelsHalos["clu"])

    assert_array_equal([est.topography_[i][0]+1 for i in range(len(est.topography_))], output_Fig1_borders["i"])
    assert_array_equal([est.topography_[i][1]+1 for i in range(len(est.topography_))], output_Fig1_borders["j"])
    npt.assert_almost_equal([est.topography_[i][2] for i in range(len(est.topography_))], output_Fig1_borders["rho_b"], decimal=3)
    npt.assert_almost_equal([est.topography_[i][3] for i in range(len(est.topography_))], output_Fig1_borders["err_rho_b"], decimal=3)




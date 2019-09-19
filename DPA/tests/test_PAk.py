import pytest
import numpy as np

from sklearn.datasets import load_iris
from sklearn.utils.testing import assert_array_equal
from sklearn.utils.testing import assert_allclose

from DPA import PointAdaptive_kNN


@pytest.fixture
def data():
    return load_iris(return_X_y=True)

def test_PointAdaptive_kNN(data):
    est = PointAdaptive_kNN()
    assert est.dim == None
    assert est.k_max == 1000
    assert est.D_thr == 23.92812698
    assert est.metric == "euclidean"
    assert est.dim_algo == "auto"    

    est.fit(data[0])
    assert hasattr(est, 'is_fitted_')

    print(est.distances_)




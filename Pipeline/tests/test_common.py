import pytest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.estimator_checks import (
    _construct_instance,
    _set_checking_parameters,
    _set_check_estimator_ids,
    check_parameters_default_constructible,
    check_class_weight_balanced_linear_classifier,
    parametrize_with_checks)

from Pipeline.DPA import PointAdaptive_kNN
from Pipeline.DPA import twoNearestNeighbors
from Pipeline.DPA import DensityPeakAdvanced
#from DPA import NR

@pytest.mark.parametrize(
    "Estimator", [DensityPeakAdvanced, PointAdaptive_kNN, twoNearestNeighbors] 
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

@parametrize_with_checks([DensityPeakAdvanced, PointAdaptive_kNN, twoNearestNeighbors])
def test_scikitlearn_compatible_estimator(estimator, check):
    check(estimator)




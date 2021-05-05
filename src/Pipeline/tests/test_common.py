import pytest

from sklearn.utils.estimator_checks import check_estimator
from sklearn.utils.estimator_checks import (
    _construct_instance,
    _set_checking_parameters,
    _get_check_estimator_ids,
    check_parameters_default_constructible,
    check_class_weight_balanced_linear_classifier,
    parametrize_with_checks)

from Pipeline.DPA import PointAdaptive_kNN
from Pipeline.DPA import twoNearestNeighbors
from Pipeline.DPA import DensityPeakAdvanced

@parametrize_with_checks([DensityPeakAdvanced(), PointAdaptive_kNN(), twoNearestNeighbors()])
def test_scikitlearn_compatible_estimator(estimator, check):
    check(estimator)




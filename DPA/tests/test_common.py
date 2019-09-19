import pytest

from sklearn.utils.estimator_checks import check_estimator

from DPA import TemplateEstimator
from DPA import TemplateClassifier
from DPA import TemplateTransformer
from DPA import PointAdaptive_kNN

@pytest.mark.parametrize(
    "Estimator", [PointAdaptive_kNN, TemplateEstimator, TemplateTransformer, TemplateClassifier]
)
def test_all_estimators(Estimator):
    return check_estimator(Estimator)

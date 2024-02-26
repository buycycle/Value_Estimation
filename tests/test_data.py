"""
    Test data realted functions
"""
from tests.test_fixtures import inputs, testdata, testmodel

def test_get_data_length(testdata):
    X_train, y_train, X_test, y_test = testdata

    assert len(X_train) >= 5000, f"training data X_train has {len(X_train)} rows, which is less than 5k rows"



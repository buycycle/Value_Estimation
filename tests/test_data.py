"""
    Test data realted functions
"""
from tests.test_fixtures import testdata

def test_get_data_length(testdata):
    model_store, X_train, X_test, y_train, y_test = testdata

    assert len(X_train) >= 5000, f"training data X_train has {len(X_train)} rows, which is less than 5k rows"

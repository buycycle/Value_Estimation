"""
    Test data realted functions
"""
from tests.test_fixtures import inputs, testdata, testmodel

def test_get_data_length(testdata):
    X_train, y_train, X_test, y_test = testdata

    assert len(X_train) >= 5000, f"training data X_train has {len(X_train)} rows, which is less than 5k rows"


def test_columns_in_X_train(testdata):
    """All features are in the training data"""

    X_train, y_train, X_test, y_test = testdata

    input_feature_names = [
        "template_id",
        "msrp",
        "bike_created_at_year",
        "bike_created_at_month",
        "bike_year",
        "sales_duration",
        "sales_country_id",
        "bike_type_id",
        "bike_category_id",
        "mileage_code",
        "city",
        "condition_code",
        "frame_size",
        "rider_height_min",
        "rider_height_max",
        "brake_type_code",
        "frame_material_code",
        "shifting_code",
        "bike_component_id",
        "color",
        "family_model_id",
        "brand_id",
        "quality_score",
        "is_mobile",
        "currency_id",
        "seller_id",
        "is_ebike",
        "is_frameset",
    ]
    # Check if each feature is in the X_train DataFrame's columns
    for feature in input_feature_names:
        assert feature in X_train.columns, f"{feature} is not in the dataframe"

"""test fixutres used in the tests"""

import time

from fastapi.testclient import TestClient
from model.app import app
from unittest.mock import Mock, patch
from buycycle.logger import Logger
import pytest

@pytest.fixture(scope="package")
def logger_mock():
    "mock the Logger"
    # Create a mock Logger instance
    logger_mock = Mock(spec=Logger)

    return logger_mock

@pytest.fixture(scope="package")
def app_mock(logger_mock):
    "patch the model with the logger mock version and prevent threads from starting"

    with patch("buycycle.logger.Logger", return_value=logger_mock), patch("src.data.ModelStore.read_data"):
        # The above patches will replace the actual methods with mocks that do nothing
        from model.app import app  # Import inside the patch context to apply the mock

        yield app  # Use yield to make it a fixture



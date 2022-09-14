import os
import logging
import churn_library as cls
import pytest
import pandas as pd

logging.basicConfig(
    filename='./logs/churn_library.log',
    level=logging.INFO,
    filemode='w',
    format='%(name)s - %(levelname)s - %(message)s')


@pytest.fixture(scope="module")
def import_data():
    return cls.import_data


def test_import(import_data):
    '''
    test data import - this example is completed for you to assist with the other test functions
    '''
    try:
        df = import_data("./data/bank_data.csv")
        logging.info("Testing import_data: SUCCESS")
    except FileNotFoundError as err:
        logging.error("Testing import_eda: The file wasn't found")
        raise err

    try:
        assert df.shape[0] > 0
        assert df.shape[1] > 0
    except AssertionError as err:
        logging.error(
            "Testing import_data: The file doesn't appear to have rows and columns")
        raise err


@pytest.fixture(scope="module")
def perform_eda():
    return cls.perform_eda


def test_eda(perform_eda):
    '''
    test perform eda function
    '''
    imgs_name = [
        'churn',
        'customer-age',
        'heatmap-correlation',
        'marital-status',
        'total-trans-ct'
    ]

    try:
        df = cls.import_data("./data/bank_data.csv")
        perform_eda(df)

        for img in imgs_name:
            if not os.path.isfile(f'./images/eda/{img}.png'):
                logging.error(
                    f"Testing test_eda: file ./images/eda/{img}.png doesn't exist")
            assert os.path.isfile(f'./images/eda/{img}.png')

        logging.info("Testing test_eda: SUCCESS")
    except AssertionError as err:
        raise err


@pytest.fixture(scope="module")
def encoder_helper():
    return cls.encoder_helper


def test_encoder_helper(encoder_helper):
    '''
    test encoder helper
    '''
    data = {
        'feature': ['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'],
        'Churn': [1, 0, 0, 0, 1, 1, 1, 0]
    }

    df = encoder_helper(pd.DataFrame(data), ['feature'])

    try:
        for row in df.iterrows():
            if row[1]['feature'] == 'a':
                assert row[1]['feature_Churn'] == 0.25
            else:
                assert row[1]['feature_Churn'] == 0.75

        logging.info("Testing encoder_helper: SUCCESS")
    except AttributeError as err:
        logging.error(
            "Testing encoder_helper: Not a DataFrame returned")
        raise err
    except AssertionError as err:
        logging.error(
            "Testing encoder_helper: Wrong value for churn")
        raise err


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''


def test_train_models(train_models):
    '''
    test train_models
    '''


if __name__ == "__main__":
    pytest.main([
        'churn_script_logging_and_tests.py::test_import',
        'churn_script_logging_and_tests.py::test_eda',
                'churn_script_logging_and_tests.py::test_encoder_helper'
                ])

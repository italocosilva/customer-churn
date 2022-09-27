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
                    f"Testing perform_eda: file ./images/eda/{img}.png doesn't exist")
            assert os.path.isfile(f'./images/eda/{img}.png')

        logging.info("Testing perform_eda: SUCCESS")
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


@pytest.fixture(scope="module")
def perform_feature_engineering():
    return cls.perform_feature_engineering


def test_perform_feature_engineering(perform_feature_engineering):
    '''
    test perform_feature_engineering
    '''

    # Load data
    df = cls.import_data("./data/bank_data.csv").head(100)

    # Encoding categories
    category_lst = [
        "Gender",
        "Education_Level",
        "Marital_Status",
        "Income_Category",
        "Card_Category"
    ]
    cls.perform_eda(df)
    df = cls.encoder_helper(df, category_lst)

    X_train, X_test, y_train, y_test = perform_feature_engineering(df)

    try:
        assert X_train.shape[0] == 70
        assert y_train.shape[0] == 70
        assert X_test.shape[0] == 30
        assert y_test.shape[0] == 30
        logging.info("Testing perform_feature_engineering: SUCCESS")
    except AssertionError as err:
        logging.error("Testing perform_feature_engineering: Wrong split size")
        raise err

    try:
        assert X_train.shape[1] == 19
        assert X_test.shape[1] == 19
    except AssertionError as err:
        logging.error(
            "Testing perform_feature_engineering: Wrong number of features")
        raise err


@pytest.fixture(scope="module")
def train_models():
    return cls.train_models


def test_train_models(train_models):
    '''
    test train_models
    '''
    imgs_name = [
        'clf-rep-rf',
        'clf-rep-lr',
        'feat-imp-rf',
    ]

    try:
        df = cls.import_data("./data/bank_data.csv")
        cls.perform_eda(df)

        # Encoding categories
        category_lst = [
            "Gender",
            "Education_Level",
            "Marital_Status",
            "Income_Category",
            "Card_Category"
        ]
        df = cls.encoder_helper(df, category_lst)

        X_train, X_test, y_train, y_test = cls.perform_feature_engineering(df)

        train_models(X_train, X_test, y_train, y_test)

        for img in imgs_name:
            if not os.path.isfile(f'./images/results/{img}.png'):
                logging.error(
                    f"Testing train_models: file ./images/results/{img}.png doesn't exist")
            assert os.path.isfile(f'./images/results/{img}.png')

        logging.info("Testing train_models: SUCCESS")
    except AssertionError as err:
        raise err


if __name__ == "__main__":
    pytest.main([
        'churn_script_logging_and_tests.py::test_import',
        'churn_script_logging_and_tests.py::test_eda',
        'churn_script_logging_and_tests.py::test_encoder_helper',
        'churn_script_logging_and_tests.py::test_perform_feature_engineering',
        'churn_script_logging_and_tests.py::test_train_models',
    ])

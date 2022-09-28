# Predict Customer Churn

- Project **Predict Customer Churn** of ML DevOps Engineer Nanodegree Udacity

### Table of Contents

1. [Installation](#installation)
2. [Project Motivation](#motivation)
3. [File Descriptions](#files)
4. [Using Instructions](#instructions)
5. [Licensing, Authors, and Acknowledgements](#licensing)

## Installation <a name="installation"></a>

It was used Python 3 from Anaconda distribution.

1. Create a new virtual enviroment and activate

```console
conda create -n customer-churn python=3.8
conda activate customer-churn
```

2. Install `requirements.txt`

```console
pip install -r requirements.txt
```

3. Install ipykernel and add to Jupyter

```console
pip install --user ipykernel
python -m ipykernel install --user --name=customer-churn
```

## Project Motivation<a name="motivation"></a>

The objective of this project is to practice `Clean Code Principles`. So I started with a Data Science project in a Jupyter Notebook and the goal was to create two scripts (first two files below).

## File Descriptions <a name="files"></a>

1. `churn_library.py`: contains refactored code from the original notebook `churn_notebook.ipynb`.
2. `churn_script_logging_and_tests.py`: contains a series of test for the functions created in the previous script.
3. `churn_notebook.ipynb`: original file with "raw" code for a Data Science Project.
4. `images folder`: contains a series of .png files from EDA and results of a classication problem.

## Using Instructions<a name="instructions"></a>

> Requirements:
> * virtual-env activated
> * be inside the folder of the project

1. Tests running

```console
ipython churn_script_logging_and_tests.py
```

2. Main code running running

```console
ipython churn_library.py
```

## Licensing, Authors, Acknowledgements<a name="licensing"></a>

Thanks Udacity for the great course and files provided to develop best practices on coding.




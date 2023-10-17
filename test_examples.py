import pytest
import warnings

"""
This script contains tests of the multirobot examples and evaluation scripts. 
"""

def blueprint_test(test_main):
    """
    Blueprint for environment tests.
    An environment main always has the four arguments:
        - n_steps: int
        - render: bool

    The function verifies if the main returns a list of observations.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings("ignore")
        history = test_main(n_steps=100, render=False)
    assert isinstance(history, dict)

def test_pointmass_static():
    from examples.example_pointmasses_static import run_point_example
    blueprint_test(run_point_example)

def test_pointmass_dynamic():
    from examples.example_pointmasses_dynamic import run_point_example
    blueprint_test(run_point_example)

def test_example_pandas_jointspace():
    from examples.example_pandas_Jointspace import define_run_panda_example
    blueprint_test(define_run_panda_example)

def test_example_pandas_cartesian():
    from examples.example_pandas_cartesian import define_run_panda_example
    blueprint_test(define_run_panda_example)

# def test_evaluations():
#     from evaluation.evaluate_random_dynamic_scenarios import define_run_evaluations
#     blueprint_test(define_run_evaluations)
#
# def test_evaluations_horizon():
#     from evaluation.evaluate_horizon import define_run_evaluations
#     blueprint_test(define_run_evaluations)



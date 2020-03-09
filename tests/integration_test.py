#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Perform integration tests for `orion.algo.skopt`."""
import os

import numpy
import pytest

from orion.algo.space import Fidelity, Integer, Real, Space
import orion.core.cli
import orion.core.io.experiment_builder as experiment_builder
from orion.core.utils.tests import OrionState
from orion.core.worker.primary_algo import PrimaryAlgo


@pytest.fixture(scope='session')
def database():
    """Return Mongo database object to test with example entries."""
    from pymongo import MongoClient
    client = MongoClient(username='user', password='pass', authSource='orion_test')
    database = client.orion_test
    yield database
    client.close()


@pytest.fixture()
def clean_db(database):
    """Clean insert example experiment entries to collections."""
    database.experiments.drop()
    database.trials.drop()
    database.workers.drop()
    database.resources.drop()


@pytest.fixture()
def space():
    """Return an optimization space"""
    space = Space()
    dim1 = Integer('yolo1', 'uniform', -3, 6)
    space.register(dim1)
    dim2 = Real('yolo2', 'uniform', 0, 1)
    space.register(dim2)
    dim3 = Fidelity('yolo3', 1, 4, 2)
    space.register(dim3)

    return space


def test_seed_rng(space):
    """Test that algo is seeded properly"""
    optimizer = PrimaryAlgo(space, 'hyperband')
    optimizer.seed_rng(1)
    a = optimizer.suggest(1)
    # Hyperband will always return the full first rung
    assert numpy.allclose(a, optimizer.suggest(1))

    optimizer.seed_rng(2)
    assert not numpy.allclose(a, optimizer.suggest(1))


def test_set_state(space):
    """Test that state is reset properly"""
    optimizer = PrimaryAlgo(space, 'hyperband')
    optimizer.seed_rng(1)
    state = optimizer.state_dict
    points = optimizer.suggest(1)
    # Hyperband will always return the full first rung
    assert numpy.allclose(points, optimizer.suggest(1))

    optimizer.seed_rng(2)
    assert not numpy.allclose(points, optimizer.suggest(1))

    optimizer.set_state(state)
    assert numpy.allclose(points, optimizer.suggest(1))


def test_optimizer(monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for single shaped dimension."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "5", "--config",
                             "./benchmark/hyperband.yaml",
                             "./benchmark/rosenbrock.py",
                             "-x~uniform(-5, 5)",
                             "-y~fidelity(1, 4, 2)"])


def test_int(monkeypatch):
    """Check support of integer values."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "5", "--config",
                             "./benchmark/hyperband.yaml",
                             "./benchmark/rosenbrock.py",
                             "-x~uniform(-5, 5, discrete=True)",
                             "-y~fidelity(1, 4, 2)"])


def test_categorical(monkeypatch):
    """Check support of categorical values."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "5", "--config",
                             "./benchmark/hyperband.yaml",
                             "./benchmark/rosenbrock.py",
                             "-x~choices([-5, -2, 0, 2, 5])",
                             "-y~fidelity(1, 4, 2)"])


def test_optimizer_two_inputs(monkeypatch):
    """Check functionality of BayesianOptimizer wrapper for 2 dimensions."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "5", "--config",
                             "./benchmark/hyperband.yaml",
                             "./benchmark/rosenbrock.py",
                             "-x~uniform(-5, 5)", "-y~uniform(-10, 10)",
                             "-z~fidelity(1, 4, 2)"])


def test_optimizer_actually_optimize(monkeypatch):
    """Check if Bayesian Optimizer has better optimization than random search."""
    monkeypatch.chdir(os.path.dirname(os.path.abspath(__file__)))
    best_random_search = 23.403275057472825

    with OrionState(experiments=[], trials=[]):

        orion.core.cli.main(["hunt", "--name", "exp", "--max-trials", "20", "--config",
                             "./benchmark/hyperband.yaml",
                             "./benchmark/rosenbrock.py",
                             "-x~uniform(-50, 50)",
                             "-y~fidelity(1, 4, 2)"])

        with open("./benchmark/hyperband.yaml", "r") as f:
            exp = experiment_builder.build_view_from_args(
                {'name': 'exp', 'config': f})

        objective = exp.stats['best_evaluation']

        assert best_random_search > objective

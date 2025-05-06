from __future__ import annotations

import math
import os
import sys
from typing import Any
from unittest.mock import MagicMock
from unittest.mock import patch
import warnings

from cmaes import CMA
import numpy as np
import optuna
from optuna._transform import _SearchSpaceTransform
from optuna.testing.storages import StorageSupplier
from optuna.trial import FrozenTrial
import pytest


sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from restart_cmaes import RestartCmaEsSampler


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("popsize", [None, 8])
def test_init_cmaes_opts(popsize: int | None) -> None:
    sampler = RestartCmaEsSampler(
        x0={"x": 0, "y": 0},
        sigma0=0.1,
        seed=1,
        n_startup_trials=1,
        popsize=popsize,
    )
    study = optuna.create_study(sampler=sampler)

    with patch("restart_cmaes.cmaes.CMA") as cma_class:
        cma_obj = MagicMock()
        cma_obj.ask.return_value = np.array((-1, -1))
        cma_obj.generation = 0
        cma_class.return_value = cma_obj
        study.optimize(
            lambda t: t.suggest_float("x", -1, 1) + t.suggest_float("y", -1, 1), n_trials=2
        )

        assert cma_class.call_count == 1

        _, actual_kwargs = cma_class.call_args
        assert np.array_equal(actual_kwargs["mean"], np.array([0.5, 0.5]))
        assert actual_kwargs["sigma"] == 0.1
        assert np.allclose(actual_kwargs["bounds"], np.array([(0, 1), (0, 1)]))
        assert actual_kwargs["seed"] == np.random.RandomState(1).randint(1, np.iinfo(np.int32).max)
        assert actual_kwargs["n_max_resampling"] == 10 * 2
        expected_popsize = 4 + math.floor(3 * math.log(2)) if popsize is None else popsize
        assert actual_kwargs["population_size"] == expected_popsize


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
def test_should_raise_exception() -> None:
    with pytest.raises(ValueError):
        RestartCmaEsSampler(
            restart_strategy="invalid-restart-strategy",
        )


def test_infer_relative_search_space_1d() -> None:
    sampler = RestartCmaEsSampler()
    study = optuna.create_study(sampler=sampler)

    # The distribution has only one candidate.
    study.optimize(lambda t: t.suggest_int("x", 1, 1), n_trials=1)
    assert sampler.infer_relative_search_space(study, study.best_trial) == {}


def test_sample_relative_1d() -> None:
    independent_sampler = optuna.samplers.RandomSampler()
    sampler = RestartCmaEsSampler(independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)

    # If search space is one dimensional, the independent sampler is always used.
    with patch.object(
        independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
    ) as mock_object:
        study.optimize(lambda t: t.suggest_int("x", -1, 1), n_trials=2)
        assert mock_object.call_count == 2


def test_sample_relative_n_startup_trials() -> None:
    independent_sampler = optuna.samplers.RandomSampler()
    sampler = RestartCmaEsSampler(n_startup_trials=2, independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)

    def objective(t: optuna.Trial) -> float:
        value = t.suggest_int("x", -1, 1) + t.suggest_int("y", -1, 1)
        if t.number == 0:
            raise Exception("first trial is failed")
        return float(value)

    # The independent sampler is used for Trial#0 (FAILED), Trial#1 (COMPLETE)
    # and Trial#2 (COMPLETE). The CMA-ES is used for Trial#3 (COMPLETE).
    with patch.object(
        independent_sampler, "sample_independent", wraps=independent_sampler.sample_independent
    ) as mock_independent, patch.object(
        sampler, "sample_relative", wraps=sampler.sample_relative
    ) as mock_relative:
        study.optimize(objective, n_trials=4, catch=(Exception,))
        assert mock_independent.call_count == 6  # The objective function has two parameters.
        assert mock_relative.call_count == 4


def _create_trials() -> list[FrozenTrial]:
    trials = []
    trials.append(
        FrozenTrial(
            number=0,
            value=1.0,
            state=optuna.trial.TrialState.COMPLETE,
            user_attrs={},
            system_attrs={},
            params={},
            distributions={},
            intermediate_values={},
            datetime_start=None,
            datetime_complete=None,
            trial_id=0,
        )
    )
    trials.append(
        FrozenTrial(
            number=1,
            value=None,
            state=optuna.trial.TrialState.PRUNED,
            user_attrs={},
            system_attrs={},
            params={},
            distributions={},
            intermediate_values={0: 2.0},
            datetime_start=None,
            datetime_complete=None,
            trial_id=0,
        )
    )
    return trials


@pytest.mark.parametrize(
    "options, key",
    [
        ({"with_margin": False, "use_separable_cma": False}, "cma:"),
        ({"with_margin": True, "use_separable_cma": False}, "cmawm:"),
        ({"with_margin": False, "use_separable_cma": True}, "sepcma:"),
    ],
)
def test_sampler_attr_key(options: dict[str, bool], key: str) -> None:
    # Test sampler attr_key property.
    sampler = RestartCmaEsSampler()

    for restart_strategy in ["ipop", "bipop"]:
        sampler._restart_strategy = restart_strategy
        for i in range(3):
            assert sampler._attr_keys.generation(i).startswith(
                ("{}:restart_{}:".format(restart_strategy, i) + "generation")
            )


@pytest.mark.parametrize("popsize", [None, 16])
def test_population_size_is_multiplied_when_enable_ipop(popsize: int | None) -> None:
    inc_popsize = 2
    sampler = RestartCmaEsSampler(
        x0={"x": 0, "y": 0},
        sigma0=0.1,
        seed=1,
        n_startup_trials=1,
        restart_strategy="ipop",
        popsize=popsize,
        inc_popsize=inc_popsize,
    )
    study = optuna.create_study(sampler=sampler)

    def objective(trial: optuna.Trial) -> float:
        _ = trial.suggest_float("x", -1, 1)
        _ = trial.suggest_float("y", -1, 1)
        return 1.0

    with patch("restart_cmaes.cmaes.CMA") as cma_class_mock, patch(
        "restart_cmaes.pickle"
    ) as pickle_mock:
        pickle_mock.dump.return_value = b"serialized object"

        should_stop_mock = MagicMock()
        should_stop_mock.return_value = True

        cma_obj = CMA(
            mean=np.array([-1, -1], dtype=float),
            sigma=1.3,
            bounds=np.array([[-1, 1], [-1, 1]], dtype=float),
            population_size=popsize,  # Already tested by test_init_cmaes_opts().
        )
        cma_obj.should_stop = should_stop_mock
        cma_class_mock.return_value = cma_obj

        initial_popsize = cma_obj.population_size
        study.optimize(objective, n_trials=2 + initial_popsize)
        assert cma_obj.should_stop.call_count == 1

        _, actual_kwargs = cma_class_mock.call_args
        assert actual_kwargs["population_size"] == inc_popsize * initial_popsize


def test_restore_optimizer_from_substrings() -> None:
    popsize = 8
    sampler = RestartCmaEsSampler(popsize=popsize)
    optimizer = sampler._restore_optimizer([])
    assert optimizer is None

    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=popsize + 2)
    optimizer = sampler._restore_optimizer(study.trials)

    assert optimizer is not None
    assert optimizer.generation == 1
    assert isinstance(optimizer, CMA)


@pytest.mark.parametrize(
    "sampler_opts",
    [
        {"restart_strategy": "ipop"},
        {"restart_strategy": "bipop"},
    ],
)
def test_restore_optimizer_after_restart(sampler_opts: dict[str, Any]) -> None:
    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    cma_class = CMA
    with patch.object(cma_class, "should_stop") as mock_method:
        mock_method.return_value = True
        sampler = RestartCmaEsSampler(popsize=5, **sampler_opts)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=5 + 2)

    optimizer = sampler._restore_optimizer(study.trials, 1)
    assert optimizer is not None
    assert optimizer.generation == 0


@pytest.mark.parametrize(
    "sampler_opts",
    [
        {"restart_strategy": "ipop"},
        {"restart_strategy": "bipop"},
    ],
)
def test_get_solution_trials(sampler_opts: dict[str, Any]) -> None:
    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    popsize = 5
    sampler = RestartCmaEsSampler(popsize=popsize, **sampler_opts)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=popsize + 2)

    # The number of solutions for generation 0 equals population size.
    assert len(sampler._get_solution_trials(study.trials, 0, 0)) == popsize

    # The number of solutions for generation 1 is 1.
    assert len(sampler._get_solution_trials(study.trials, 1, 0)) == 1


@pytest.mark.parametrize(
    "sampler_opts, restart_strategy",
    [
        ({}, "ipop"),
        ({}, "bipop"),
    ],
)
def test_get_solution_trials_with_other_options(
    sampler_opts: dict[str, Any], restart_strategy: str
) -> None:
    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    sampler = RestartCmaEsSampler(popsize=5, restart_strategy=restart_strategy)
    study = optuna.create_study(sampler=sampler)
    study.optimize(objective, n_trials=5 + 2)

    # The number of solutions is 0 after changed samplers
    sampler = RestartCmaEsSampler(**sampler_opts)
    assert len(sampler._get_solution_trials(study.trials, 0, 0)) == 0


@pytest.mark.parametrize(
    "sampler_opts",
    [
        {"restart_strategy": "ipop"},
        {"restart_strategy": "bipop"},
    ],
)
def test_get_solution_trials_after_restart(sampler_opts: dict[str, Any]) -> None:
    def objective(trial: optuna.Trial) -> float:
        x1 = trial.suggest_float("x1", -10, 10, step=1)
        x2 = trial.suggest_float("x2", -10, 10)
        return x1**2 + x2**2

    cma_class = CMA

    popsize = 5
    with patch.object(cma_class, "should_stop") as mock_method:
        mock_method.return_value = True
        sampler = RestartCmaEsSampler(popsize=popsize, **sampler_opts)
        study = optuna.create_study(sampler=sampler)
        study.optimize(objective, n_trials=popsize + 2)

    # The number of solutions for generation=0 and n_restarts=0 equals population size.
    assert len(sampler._get_solution_trials(study.trials, 0, 0)) == popsize

    # The number of solutions for generation=1 and n_restarts=0 is 0.
    assert len(sampler._get_solution_trials(study.trials, 1, 0)) == 0

    # The number of solutions for generation=0 and n_restarts=1 is 1 since it was restarted.
    assert len(sampler._get_solution_trials(study.trials, 0, 1)) == 1


@pytest.mark.parametrize(
    "dummy_optimizer_str,attr_len",
    [
        ("012", 1),
        ("01234", 1),
        ("012345", 2),
    ],
)
def test_split_and_concat_optimizer_string(dummy_optimizer_str: str, attr_len: int) -> None:
    sampler = RestartCmaEsSampler()
    with patch("restart_cmaes._SYSTEM_ATTR_MAX_LENGTH", 5):
        attrs = sampler._split_optimizer_str(dummy_optimizer_str)
        assert len(attrs) == attr_len
        actual = sampler._concat_optimizer_attrs(attrs)
        assert dummy_optimizer_str == actual


def test_call_after_trial_of_base_sampler() -> None:
    independent_sampler = optuna.samplers.RandomSampler()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", optuna.exceptions.ExperimentalWarning)
        sampler = RestartCmaEsSampler(independent_sampler=independent_sampler)
    study = optuna.create_study(sampler=sampler)
    with patch.object(
        independent_sampler, "after_trial", wraps=independent_sampler.after_trial
    ) as mock_object:
        study.optimize(lambda _: 1.0, n_trials=1)
        assert mock_object.call_count == 1


def test_is_compatible_search_space() -> None:
    transform = _SearchSpaceTransform(
        {
            "x0": optuna.distributions.FloatDistribution(2, 3),
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        }
    )

    assert optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
            "x0": optuna.distributions.FloatDistribution(2, 3),
        },
    )

    # Same search space size, but different param names.
    assert not optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x0": optuna.distributions.FloatDistribution(2, 3),
            "foo": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        },
    )

    # x2 is added.
    assert not optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x0": optuna.distributions.FloatDistribution(2, 3),
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
            "x2": optuna.distributions.FloatDistribution(2, 3, step=0.1),
        },
    )

    # x0 is not found.
    assert not optuna.samplers._cmaes._is_compatible_search_space(
        transform,
        {
            "x1": optuna.distributions.CategoricalDistribution(["foo", "bar", "baz", "qux"]),
        },
    )


@pytest.mark.filterwarnings("ignore::optuna.exceptions.ExperimentalWarning")
@pytest.mark.parametrize("storage_name", ["sqlite", "journal"])
def test_rdb_storage(storage_name: str) -> None:
    # Confirm `study._storage.set_trial_system_attr` does not fail in several storages.
    def objective(trial: optuna.Trial) -> float:
        x = trial.suggest_float("x", -10, 10)
        y = trial.suggest_int("y", -10, 10)
        return x**2 + y

    with StorageSupplier(storage_name) as storage:
        study = optuna.create_study(
            sampler=RestartCmaEsSampler(),
            storage=storage,
        )
        study.optimize(objective, n_trials=3)

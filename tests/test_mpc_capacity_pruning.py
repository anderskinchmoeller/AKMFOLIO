"""Capacity pruning keeps every inequality that can bind inside the bounds."""
import numpy as np

from akm_hrp.allocators.retail_alpha_ml_mpc import _binding_capacity_rows


def test_capacity_mask_keeps_tight_boundaries_and_limits():
    lower = np.array([[.01, .0], [.0, .03]])
    upper = np.array([[.1, .2], [.2, .1]])
    previous = np.array([.04, .1])
    capacity = np.array([.1, .1])
    # First buy: [.06, .10], first sale: [.03, .10];
    # second buy: [.19, .10], second sale: [.10, .17].
    np.testing.assert_array_equal(
        _binding_capacity_rows(lower, upper, previous, capacity, [0, 1]),
        [False, True, False, True, True, True, True, True],
    )
    assert not _binding_capacity_rows(lower, upper, previous, np.ones(2), [0, 1]).any()
    assert _binding_capacity_rows(lower, upper, previous, capacity, []).size == 0


def test_pruned_rows_are_strictly_satisfied_throughout_the_box():
    rng = np.random.default_rng(932)
    lower = rng.uniform(0, .02, (3, 15))
    upper = lower + rng.uniform(.01, .1, (3, 15))
    previous = rng.uniform(0, .1, 15)
    capacity = rng.uniform(.01, .2, 15)
    for steps in ([0, 1, 2], [1, 2]):
        mask = _binding_capacity_rows(lower, upper, previous, capacity, steps)
        for _ in range(100):
            path = rng.uniform(lower, upper)
            values = []
            for step in steps:
                trade = path[step] - (previous if step == 0 else path[step - 1])
                values.extend([capacity - trade, capacity + trade])
            assert (np.concatenate(values)[~mask] > 0).all()

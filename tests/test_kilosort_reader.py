import warnings

import numpy as np
import pytest

from element_array_ephys.readers.kilosort import Kilosort


def _make_kilosort(pc_features=None):
    """Build a Kilosort reader with in-memory data, bypassing file loading.

    Probe: 3 channels at y = 0, 20, 40 um. Two templates; each template's
    peak is on a different channel, and both use all three channels as
    PC feature channels.
    """
    ks = object.__new__(Kilosort)
    data = {
        "channel_positions": np.array([[0, 0], [0, 20], [0, 40]], dtype=float),
        "channel_map": np.array([0, 1, 2]),
        "pc_feature_ind": np.array([[0, 1, 2], [0, 1, 2]]),
        # templates: (n_templates, n_samples, n_channels)
        "templates": np.array(
            [
                [[0.0, 5.0, 1.0], [0.0, -2.0, 0.0]],  # peak on channel 1
                [[0.0, 0.0, -9.0], [1.0, 0.0, 0.0]],  # peak on channel 2
            ]
        ),
    }
    if pc_features is not None:
        data["spike_templates"] = np.zeros(len(pc_features), dtype=int)
        # (n_spikes, n_pcs, n_feature_channels); only the 1st PC is used
        data["pc_features"] = np.stack([pc_features, np.ones_like(pc_features)], axis=1)
    else:
        data["spike_templates"] = np.array([0, 1])
    ks._data = data
    return ks


def test_spike_depths_center_of_mass():
    ks = _make_kilosort(
        np.array(
            [
                [0.0, 1.0, 0.0],  # all weight on y=20
                [1.0, 0.0, 1.0],  # equal weight on y=0 and y=40
                [-3.0, 0.0, 2.0],  # negative feature is clamped to 0
            ]
        )
    )
    ks.extract_spike_depths()
    np.testing.assert_allclose(ks.data["spike_depths"], [20.0, 20.0, 40.0])


@pytest.mark.parametrize(
    "bad_row",
    [
        [0.0, 0.0, 0.0],  # all zero
        [-1.0, -2.0, -3.0],  # all negative -> all zero after clamping
        [-0.0, 0.0, -1e-9],  # signed zeros / tiny negatives
    ],
)
def test_spike_depths_zero_weight_spike_is_nan_without_warning(bad_row):
    ks = _make_kilosort(np.array([[0.0, 1.0, 0.0], bad_row]))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ks.extract_spike_depths()
    depths = ks.data["spike_depths"]
    assert depths[0] == 20.0
    assert np.isnan(depths[1])


def test_spike_depths_all_spikes_zero_weight():
    ks = _make_kilosort(-np.ones((4, 3)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ks.extract_spike_depths()
    depths = ks.data["spike_depths"]
    assert depths.shape == (4,)
    assert np.isnan(depths).all()


def test_spike_depths_no_spikes():
    ks = _make_kilosort(np.empty((0, 3)))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ks.extract_spike_depths()
    assert ks.data["spike_depths"].shape == (0,)


def test_spike_depths_none_without_pc_features():
    ks = _make_kilosort()
    ks.extract_spike_depths()
    assert ks.data["spike_depths"] is None
    np.testing.assert_array_equal(ks.data["spike_sites"], [1, 2])

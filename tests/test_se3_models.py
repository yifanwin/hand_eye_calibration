import numpy as np
import pytest

from hand_eye_calibration.models import Observation
from hand_eye_calibration.se3 import as_transform


def test_transform_validation_and_float64():
    transform = as_transform(np.eye(4, dtype=np.float32))
    assert transform.dtype == np.float64
    bad = np.eye(4)
    bad[0, 0] = 2.0
    with pytest.raises(ValueError, match="orthonormal"):
        as_transform(bad)


def test_observation_round_trip(synthetic_case):
    _, observations = synthetic_case
    restored = Observation.from_dict(observations[0].to_dict())
    np.testing.assert_allclose(restored.T_base_ee, observations[0].T_base_ee)
    assert restored.joint_positions_rad.shape == (7,)


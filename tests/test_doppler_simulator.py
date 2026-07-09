import numpy as np
import pytest

from gnss_doppler_lab.doppler_simulator import (
    GPS_L1_HZ,
    SPEED_OF_LIGHT_MPS,
    add_common_spoofing_drift,
    carrier_doppler_hz,
    doppler_observation_matrix,
    los_unit_vectors,
)


def test_los_unit_vectors_point_from_receiver_to_each_satellite():
    receiver_position = np.array([1.0, 1.0, 0.0])
    satellite_positions = np.array([
        [2.0, 1.0, 0.0],
        [1.0, 4.0, 0.0],
    ])

    los = los_unit_vectors(satellite_positions, receiver_position)

    np.testing.assert_allclose(los, [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])


def test_carrier_doppler_matches_relative_velocity_projection():
    wavelength = SPEED_OF_LIGHT_MPS / GPS_L1_HZ
    satellite_positions = np.array([[20_200_000.0, 0.0, 0.0]])
    receiver_position = np.array([0.0, 0.0, 0.0])
    satellite_velocities = np.array([[0.0, 0.0, 0.0]])
    receiver_velocity = np.array([10.0, 0.0, 0.0])

    doppler = carrier_doppler_hz(
        satellite_positions,
        satellite_velocities,
        receiver_position,
        receiver_velocity,
        clock_drift_hz=3.0,
    )

    np.testing.assert_allclose(doppler, [10.0 / wavelength + 3.0])


def test_carrier_doppler_rejects_zero_range_geometry():
    with pytest.raises(ValueError, match="zero range"):
        carrier_doppler_hz(
            np.array([[0.0, 0.0, 0.0]]),
            np.array([[0.0, 0.0, 0.0]]),
            np.array([0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0]),
        )


def test_doppler_observation_matrix_simulates_epochs_and_prns():
    satellite_positions = np.array([
        [[20_200_000.0, 0.0, 0.0], [0.0, 20_200_000.0, 0.0]],
        [[20_200_000.0, 0.0, 0.0], [0.0, 20_200_000.0, 0.0]],
    ])
    satellite_velocities = np.zeros_like(satellite_positions)
    receiver_positions = np.array([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
    ])
    receiver_velocities = np.array([
        [10.0, 0.0, 0.0],
        [0.0, 20.0, 0.0],
    ])

    doppler = doppler_observation_matrix(
        satellite_positions,
        satellite_velocities,
        receiver_positions,
        receiver_velocities,
        clock_drift_hz=np.array([1.0, 2.0]),
    )

    wavelength = SPEED_OF_LIGHT_MPS / GPS_L1_HZ
    np.testing.assert_allclose(
        doppler,
        [
            [10.0 / wavelength + 1.0, 1.0],
            [2.0, 20.0 / wavelength + 2.0],
        ],
    )


def test_add_common_spoofing_drift_adds_epoch_common_component_and_prn_offsets():
    authentic = np.array([[1.0, 2.0, 3.0], [10.0, 20.0, 30.0]])

    spoofed = add_common_spoofing_drift(
        authentic,
        common_drift_hz=np.array([100.0, 200.0]),
        prn_offsets_hz=np.array([0.0, 1.0, -1.0]),
    )

    np.testing.assert_allclose(
        spoofed,
        [[101.0, 103.0, 102.0], [210.0, 221.0, 229.0]],
    )

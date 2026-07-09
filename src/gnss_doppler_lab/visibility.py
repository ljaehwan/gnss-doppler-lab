from __future__ import annotations

from dataclasses import dataclass

from gnss_doppler_lab.coordinates import azimuth_elevation
from gnss_doppler_lab.satellites import SatelliteState


@dataclass(slots=True)
class VisibilityRecord:
    prn: str
    azimuth_deg: float
    elevation_deg: float
    range_m: float


def visible_satellites(
    receiver_ecef: tuple[float, float, float],
    satellites: list[SatelliteState],
    latitude_deg: float,
    longitude_deg: float,
    mask_angle_deg: float,
) -> list[VisibilityRecord]:
    visible: list[VisibilityRecord] = []
    for satellite in satellites:
        azimuth_deg, elevation_deg, range_m = azimuth_elevation(
            receiver_ecef=receiver_ecef,
            satellite_ecef=satellite.position_ecef_m,
            latitude_deg=latitude_deg,
            longitude_deg=longitude_deg,
        )
        if elevation_deg >= mask_angle_deg:
            visible.append(
                VisibilityRecord(
                    prn=satellite.prn,
                    azimuth_deg=azimuth_deg,
                    elevation_deg=elevation_deg,
                    range_m=range_m,
                )
            )
    return visible

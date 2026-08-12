"""Descriptor-backed adapter for the byte-pinned GCMR ephemeris parser."""
from __future__ import annotations

import xml.etree.ElementTree as ET

from .gcmr_geometry import (GpsEphemeris, _OPTIONAL_XML_FIELDS, _XML_FIELDS,
                            _direct_text, _finite, _local_name,
                            _validate_ephemeris)


def parse_ephemeris_handle(handle):
    """Equivalent strict parser operating on an authenticated open descriptor."""
    try:
        root = ET.parse(handle).getroot()
    except (OSError, ET.ParseError) as exc:
        raise ValueError(f"invalid GPS ephemeris XML: {exc}") from exc
    maps = [node for node in root.iter() if "ephemeris_map" in _local_name(node.tag)]
    if len(maps) != 1: raise ValueError("GPS ephemeris XML must contain exactly one ephemeris_map")
    items = [node for node in maps[0] if _local_name(node.tag) == "item"]
    if not items: raise ValueError("GPS ephemeris XML contains no ephemerides")
    result = {}
    for item in items:
        second = next((node for node in item if _local_name(node.tag) == "second"), None)
        if second is None: raise ValueError("ephemeris map item is missing second")
        values = {name: _finite(name, _direct_text(second, name)) for name in _XML_FIELDS}
        direct = {_local_name(child.tag): child.text.strip() for child in second
                  if child.text is not None and child.text.strip()}
        for xml_name, field_name in _OPTIONAL_XML_FIELDS.items():
            if xml_name in direct: values[field_name] = _finite(xml_name, direct[xml_name])
        prn, week = values["PRN"], values["WN"]
        if not prn.is_integer() or not 1 <= prn <= 32: raise ValueError("PRN must be an integer in [1, 32]")
        if not week.is_integer() or week < 0: raise ValueError("WN must be a nonnegative integer")
        values["PRN"], values["WN"] = int(prn), int(week)
        for name in ("SV_health", "fit_interval_flag"):
            if name in values:
                if not values[name].is_integer(): raise ValueError(f"{name} must be an integer")
                values[name] = int(values[name])
        ephemeris = GpsEphemeris(**values); _validate_ephemeris(ephemeris)
        map_prn = _finite("map PRN", _direct_text(item, "first"))
        if not map_prn.is_integer() or int(map_prn) != ephemeris.PRN:
            raise ValueError("ephemeris map PRN does not match record PRN")
        if ephemeris.PRN in result: raise ValueError(f"duplicate ephemeris for PRN {ephemeris.PRN}")
        result[ephemeris.PRN] = ephemeris
    return result

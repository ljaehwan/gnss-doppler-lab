#!/usr/bin/env python3
from __future__ import annotations
import argparse, re
from datetime import datetime, timedelta
from pathlib import Path

float_re = re.compile(r"[-+ ]?\d*\.\d+(?:[eEdD][-+]?\d+)?|[-+ ]?\d+\.?(?:[eEdD][-+]?\d+)")

def fmt(v: str) -> str:
    x = float(v.replace('D','E').replace('d','e'))
    return f"{x:19.12E}".replace('E','D')

def ion_fmt(v: str) -> str:
    x = float(v.replace('D','E').replace('d','e'))
    return f"{x:12.4E}".replace('E','D')

def convert_block(block: list[str]) -> tuple[datetime, int, list[str]]:
    first = block[0]
    prn = int(first[1:3]); year = int(first[4:8]); yy = year % 100
    mon = int(first[9:11]); day = int(first[12:14]); hh = int(first[15:17]); mm = int(first[18:20]); ss = float(first[21:23])
    dt = datetime(year, mon, day, hh, mm, int(ss))
    vals = float_re.findall(first[23:])
    if len(vals) < 3:
        vals = float_re.findall(first)[6:9]
    out = [f"{prn:2d} {yy:02d} {mon:2d} {day:2d} {hh:2d} {mm:2d}{ss:5.1f}" + "".join(fmt(v) for v in vals[:3])]
    for cont in block[1:8]:
        vals = float_re.findall(cont)
        out.append("   " + "".join(fmt(v) for v in vals[:4]))
    return dt, prn, out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('src'); ap.add_argument('dst')
    ap.add_argument('--preserve-order', action='store_true')
    ap.add_argument('--recent-window-hours', type=float, default=48.0,
                    help='keep only records within this many hours of the newest GPS record; 0 disables')
    a = ap.parse_args()
    lines = Path(a.src).read_text(errors='ignore').splitlines()
    i = 0; gpsa = None; gpsb = None; leap = None
    while i < len(lines):
        line = lines[i]
        if line.startswith('GPSA'):
            vals = float_re.findall(line[4:60]); gpsa = vals[:4]
        elif line.startswith('GPSB'):
            vals = float_re.findall(line[4:60]); gpsb = vals[:4]
        elif 'LEAP SECONDS' in line:
            m = re.search(r"[-+]?\d+", line[:20]); leap = m.group(0) if m else None
        if 'END OF HEADER' in line:
            i += 1; break
        i += 1

    records: list[tuple[datetime, int, list[str]]] = []
    while i < len(lines):
        l = lines[i]
        if not re.match(r"^G\d{2}\s", l):
            i += 1; continue
        block = lines[i:i+8]
        if len(block) < 8:
            break
        records.append(convert_block(block))
        i += 8
    original_count = len(records)
    if a.recent_window_hours and records:
        newest = max(r[0] for r in records)
        cutoff = newest - timedelta(hours=a.recent_window_hours)
        records = [r for r in records if r[0] >= cutoff]
    if not a.preserve_order:
        records.sort(key=lambda x: (x[0], x[1]))

    out = []
    out.append(f"{2:6.2f}           NAVIGATION DATA                         RINEX VERSION / TYPE")
    out.append("gnss-doppler-lab  BKG/BRDC RINEX3 GPS filter           PGM / RUN BY / DATE ")
    out.append("Converted from RINEX 3 MIXED BRDC; GPS records only; epoch-sorted. COMMENT             ")
    if gpsa:
        out.append("".join(ion_fmt(v) for v in gpsa).ljust(60) + "ION ALPHA           ")
    if gpsb:
        out.append("".join(ion_fmt(v) for v in gpsb).ljust(60) + "ION BETA            ")
    if leap:
        out.append(f"{int(leap):6d}".ljust(60) + "LEAP SECONDS        ")
    out.append("".ljust(60) + "END OF HEADER       ")
    for _dt, _prn, block_out in records:
        out.extend(block_out)
    Path(a.dst).write_text("\n".join(out) + "\n")
    print(f"wrote {a.dst} gps_records={len(records)} original_records={original_count} sorted={not a.preserve_order}")
    if records:
        print(f"record_time_range={records[0][0].isoformat()}..{records[-1][0].isoformat()}")

if __name__=='__main__':
    main()

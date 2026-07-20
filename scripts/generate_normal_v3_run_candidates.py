#!/usr/bin/env python3
from __future__ import annotations
import csv, json, re
from datetime import datetime, timezone
from pathlib import Path

# 100 country capitals / major national capitals. Coordinates are city-centre approximate.
CAPITALS = [
    ("KOR", "Seoul", 37.5665, 126.9780, 50),
    ("JPN", "Tokyo", 35.6762, 139.6503, 40),
    ("CHN", "Beijing", 39.9042, 116.4074, 45),
    ("TWN", "Taipei", 25.0330, 121.5654, 20),
    ("MNG", "Ulaanbaatar", 47.8864, 106.9057, 1350),
    ("THA", "Bangkok", 13.7563, 100.5018, 5),
    ("VNM", "Hanoi", 21.0278, 105.8342, 20),
    ("KHM", "PhnomPenh", 11.5564, 104.9282, 12),
    ("LAO", "Vientiane", 17.9757, 102.6331, 174),
    ("MMR", "Naypyidaw", 19.7633, 96.0785, 115),
    ("MYS", "KualaLumpur", 3.1390, 101.6869, 66),
    ("SGP", "Singapore", 1.3521, 103.8198, 15),
    ("IDN", "Jakarta", -6.2088, 106.8456, 8),
    ("PHL", "Manila", 14.5995, 120.9842, 16),
    ("IND", "NewDelhi", 28.6139, 77.2090, 216),
    ("PAK", "Islamabad", 33.6844, 73.0479, 540),
    ("BGD", "Dhaka", 23.8103, 90.4125, 4),
    ("NPL", "Kathmandu", 27.7172, 85.3240, 1400),
    ("LKA", "Colombo", 6.9271, 79.8612, 5),
    ("KAZ", "Astana", 51.1605, 71.4704, 347),
    ("UZB", "Tashkent", 41.2995, 69.2401, 455),
    ("KGZ", "Bishkek", 42.8746, 74.5698, 800),
    ("TJK", "Dushanbe", 38.5598, 68.7870, 800),
    ("TKM", "Ashgabat", 37.9601, 58.3261, 220),
    ("IRN", "Tehran", 35.6892, 51.3890, 1200),
    ("IRQ", "Baghdad", 33.3152, 44.3661, 34),
    ("TUR", "Ankara", 39.9334, 32.8597, 938),
    ("SAU", "Riyadh", 24.7136, 46.6753, 612),
    ("ARE", "AbuDhabi", 24.4539, 54.3773, 27),
    ("QAT", "Doha", 25.2854, 51.5310, 10),
    ("ISR", "Jerusalem", 31.7683, 35.2137, 754),
    ("JOR", "Amman", 31.9539, 35.9106, 757),
    ("EGY", "Cairo", 30.0444, 31.2357, 23),
    ("ETH", "AddisAbaba", 8.9806, 38.7578, 2355),
    ("KEN", "Nairobi", -1.2921, 36.8219, 1795),
    ("TZA", "Dodoma", -6.1630, 35.7516, 1120),
    ("UGA", "Kampala", 0.3476, 32.5825, 1190),
    ("RWA", "Kigali", -1.9441, 30.0619, 1567),
    ("ZAF", "Pretoria", -25.7479, 28.2293, 1339),
    ("NGA", "Abuja", 9.0765, 7.3986, 456),
    ("GHA", "Accra", 5.6037, -0.1870, 61),
    ("SEN", "Dakar", 14.7167, -17.4677, 22),
    ("MAR", "Rabat", 34.0209, -6.8416, 75),
    ("DZA", "Algiers", 36.7538, 3.0588, 60),
    ("TUN", "Tunis", 36.8065, 10.1815, 4),
    ("LBY", "Tripoli", 32.8872, 13.1913, 81),
    ("GBR", "London", 51.5072, -0.1276, 35),
    ("IRL", "Dublin", 53.3498, -6.2603, 20),
    ("FRA", "Paris", 48.8566, 2.3522, 35),
    ("ESP", "Madrid", 40.4168, -3.7038, 667),
    ("PRT", "Lisbon", 38.7223, -9.1393, 45),
    ("DEU", "Berlin", 52.5200, 13.4050, 34),
    ("NLD", "Amsterdam", 52.3676, 4.9041, 2),
    ("BEL", "Brussels", 50.8503, 4.3517, 13),
    ("CHE", "Bern", 46.9480, 7.4474, 540),
    ("AUT", "Vienna", 48.2082, 16.3738, 170),
    ("ITA", "Rome", 41.9028, 12.4964, 21),
    ("GRC", "Athens", 37.9838, 23.7275, 70),
    ("POL", "Warsaw", 52.2297, 21.0122, 100),
    ("CZE", "Prague", 50.0755, 14.4378, 235),
    ("HUN", "Budapest", 47.4979, 19.0402, 96),
    ("ROU", "Bucharest", 44.4268, 26.1025, 70),
    ("BGR", "Sofia", 42.6977, 23.3219, 550),
    ("SRB", "Belgrade", 44.7866, 20.4489, 117),
    ("HRV", "Zagreb", 45.8150, 15.9819, 122),
    ("SWE", "Stockholm", 59.3293, 18.0686, 28),
    ("NOR", "Oslo", 59.9139, 10.7522, 23),
    ("DNK", "Copenhagen", 55.6761, 12.5683, 10),
    ("FIN", "Helsinki", 60.1699, 24.9384, 25),
    ("ISL", "Reykjavik", 64.1466, -21.9426, 15),
    ("EST", "Tallinn", 59.4370, 24.7536, 9),
    ("LVA", "Riga", 56.9496, 24.1052, 6),
    ("LTU", "Vilnius", 54.6872, 25.2797, 112),
    ("UKR", "Kyiv", 50.4501, 30.5234, 179),
    ("USA", "WashingtonDC", 38.9072, -77.0369, 7),
    ("CAN", "Ottawa", 45.4215, -75.6972, 70),
    ("MEX", "MexicoCity", 19.4326, -99.1332, 2240),
    ("GTM", "GuatemalaCity", 14.6349, -90.5069, 1500),
    ("PAN", "PanamaCity", 8.9824, -79.5199, 2),
    ("CUB", "Havana", 23.1136, -82.3666, 59),
    ("DOM", "SantoDomingo", 18.4861, -69.9312, 14),
    ("COL", "Bogota", 4.7110, -74.0721, 2640),
    ("ECU", "Quito", -0.1807, -78.4678, 2850),
    ("PER", "Lima", -12.0464, -77.0428, 154),
    ("BOL", "LaPaz", -16.4897, -68.1193, 3640),
    ("CHL", "Santiago", -33.4489, -70.6693, 520),
    ("ARG", "BuenosAires", -34.6037, -58.3816, 25),
    ("URY", "Montevideo", -34.9011, -56.1645, 43),
    ("PRY", "Asuncion", -25.2637, -57.5759, 43),
    ("BRA", "Brasilia", -15.7939, -47.8828, 1172),
    ("AUS", "Canberra", -35.2809, 149.1300, 577),
    ("NZL", "Wellington", -41.2865, 174.7762, 31),
    ("PNG", "PortMoresby", -9.4438, 147.1803, 35),
    ("FJI", "Suva", -18.1248, 178.4501, 5),
    ("RUS", "Moscow", 55.7558, 37.6173, 156),
    ("ARM", "Yerevan", 40.1792, 44.4991, 990),
    ("GEO", "Tbilisi", 41.7151, 44.8271, 380),
    ("AZE", "Baku", 40.4093, 49.8671, -20),
    ("AFG", "Kabul", 34.5553, 69.2075, 1790),
    ("ALB", "Tirana", 41.3275, 19.8189, 110),
]
TIMES = [("epoch-a", "00:00:00"), ("epoch-b", "04:00:00"), ("epoch-c", "08:00:00")]
DATE = "2026-07-18"
DURATION = 300
NAV = "data/ephemeris/brdc1990.26n"

def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", s.lower()).strip('-')

def split_for(i: int) -> str:
    # full-run split; validation/test are held-out by city index pattern
    if i % 10 == 8:
        return "val"
    if i % 10 == 9:
        return "test"
    return "train"

def main():
    outdir=Path("configs/generated/normal_v3_large_300")
    outdir.mkdir(parents=True, exist_ok=True)
    csv_path=outdir/"run_index.csv"
    rows=[]
    idx=0
    for city_idx, (cc, city, lat, lon, alt) in enumerate(CAPITALS):
        for tod, clock in TIMES:
            idx += 1
            utc=f"{DATE}T{clock}Z"
            rid=f"normal-v3-{idx:03d}-{cc.lower()}-{slug(city)}-{tod}-20260718"
            rows.append({
                "run_id": rid, "country_code": cc, "capital": city,
                "latitude_deg": lat, "longitude_deg": lon, "altitude_m": alt,
                "utc": utc, "duration_seconds": DURATION, "time_of_day": tod,
                "motion_type": "static", "rinex_nav": NAV,
                "split": split_for(city_idx),
            })
    fields=list(rows[0])
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        w=csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    manifest={
        "schema":"gnss-doppler-lab.normal-v3-large-run-candidates",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "row_count": len(rows), "duration_seconds_per_run": DURATION,
        "total_duration_seconds": len(rows)*DURATION,
        "location_count": len(CAPITALS), "times_per_location": [x[0] for x in TIMES],
        "rinex_nav": NAV,
        "split_counts": {s: sum(r["split"]==s for r in rows) for s in ["train","val","test"]},
        "policy":"100 capitals x three UTC epochs (00/04/08) within available RINEX2 ephemeris coverage; static 5-min normal candidates; full-run split by city index; no PRN-ID assumption",
    }
    (outdir/"run_index.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    print(json.dumps(manifest, sort_keys=True))
if __name__ == "__main__": main()

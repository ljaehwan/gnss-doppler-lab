import json

from gnss_doppler_lab.cli import main


def test_probe_cli_reports_machine_readable_status(capsys):
    code = main(["probe", "--executable", "/definitely/missing/gps-sdr-sim"])
    assert code == 1
    report = json.loads(capsys.readouterr().out)
    assert report["available"] is False
    assert report["provenance"] == "unverified"
    assert "osqzss/gps-sdr-sim@" in report["cli_contract"]
    assert report["executable_sha256"] is None

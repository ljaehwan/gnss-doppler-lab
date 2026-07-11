"""Command-line entry point for software-only IQ generation."""
import argparse, json, sys
from .gps_sdr_sim import GpsSdrSimRunner, SimulatorError
from .rf_config import load_rf_config, ConfigError
from .rf_pipeline import generate_iq


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="gnss-iq", description="Generate normal GPS L1 C/A software IQ")
    subs = parser.add_subparsers(dest="command", required=True)
    gen = subs.add_parser("generate"); gen.add_argument("config"); gen.add_argument("--executable")
    probe = subs.add_parser("probe"); probe.add_argument("--executable")
    args = parser.parse_args(argv)
    try:
        if args.command == "probe":
            runner = GpsSdrSimRunner(args.executable)
            report = runner.probe(); print(json.dumps(report, sort_keys=True)); return 0 if report["available"] else 1
        cfg = load_rf_config(args.config)
        runner = GpsSdrSimRunner(args.executable or cfg.simulator.executable)
        manifest = generate_iq(cfg, runner); print(manifest); return 0
    except (ConfigError, SimulatorError, OSError, FileExistsError) as exc:
        print(f"error: {exc}", file=sys.stderr); return 2

if __name__ == "__main__": raise SystemExit(main())

from __future__ import annotations

import argparse
import json

from .occurrence_intensity_e1_data_audit import run_real_data_audit


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the frozen E1 real-data audit")
    parser.add_argument("config")
    parser.add_argument("output_dir")
    args = parser.parse_args()
    result = run_real_data_audit(args.config, args.output_dir)
    print(json.dumps({"status": result["run_status"]["status"]}, sort_keys=True))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Outcome-agnostic server-CPU prototype for the frozen conformal baseline."""
from __future__ import annotations
import argparse,csv,json,os
from pathlib import Path
from conformal_area_reference import ARTIFACT_POLICY,evaluate,validate_interface

def run(source_path: Path, output_dir: Path, source_experiment: str) -> None:
    validate_interface({"source_experiment":source_experiment},ARTIFACT_POLICY)
    source=json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(source,dict) or set(source)!={"source_experiment","artifact_policy","member_areas","truth_area"}: raise ValueError("source JSON keys must be exact")
    validate_interface({"source_experiment":source["source_experiment"]},source["artifact_policy"])
    members=source["member_areas"]
    if not isinstance(members,list) or len(members)!=40 or any(not isinstance(row,list) or len(row)!=10 for row in members): raise ValueError("member_areas must have shape 40x10")
    result=evaluate([min(row) for row in members],[max(row) for row in members],source["truth_area"]); rows=result.pop("rows")
    output_dir.mkdir(parents=True,exist_ok=False)
    (output_dir/"conformal_summary.json").write_text(json.dumps(result,indent=2,sort_keys=True)+"\n",encoding="utf-8")
    with (output_dir/"conformal_per_case.csv").open("w",newline="",encoding="utf-8") as stream:
        writer=csv.DictWriter(stream,fieldnames=tuple(rows[0])); writer.writeheader(); writer.writerows(rows)

def main():
    parser=argparse.ArgumentParser(); parser.add_argument("--source-experiment",required=True); parser.add_argument("--output-dir",required=True,type=Path); args=parser.parse_args()
    source=os.environ.get("CONFORMAL_AREA_SOURCE_JSON")
    if not source: raise ValueError("CONFORMAL_AREA_SOURCE_JSON is required")
    run(Path(source),args.output_dir,args.source_experiment)
if __name__=="__main__": main()

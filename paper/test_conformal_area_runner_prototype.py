from __future__ import annotations
import json,os,subprocess,sys,tempfile,unittest
from pathlib import Path
from paper.conformal_area_reference import conformal_quantile,evaluate,purged_folds
PAPER=Path(__file__).resolve().parent; RUNNER=PAPER/"conformal_area_runner_prototype.py"

class ConformalAreaTests(unittest.TestCase):
    def test_folds_and_quantile_are_literal(self):
        folds=purged_folds(); self.assertEqual([len(t) for _,t in folds],[29,26,26,26,29]); self.assertEqual(folds[0][0],tuple(range(8))); self.assertEqual(folds[-1][0],tuple(range(32,40)))
        self.assertEqual(conformal_quantile(list(range(26))),(24,25)); self.assertEqual(conformal_quantile(list(range(29))),(26,27))
    def test_useful_and_negative_decisions(self):
        lower,upper=[0.0]*40,[10.0]*40; useful=evaluate(lower,upper,[5.0]*40); self.assertEqual(useful["decision"],"CONFORMAL_USEFUL"); self.assertEqual(useful["raw_median_width"],10.0); self.assertEqual(useful["conformal_median_width"],10.0)
        negative=evaluate(lower,upper,[100.0 if i%4==0 else 5.0 for i in range(40)]); self.assertEqual(negative["decision"],"CONFORMAL_NEGATIVE"); self.assertGreater(negative["width_ratio"],1.5)
    def test_cli_emits_only_two_compact_outputs(self):
        source={"source_experiment":"joint_full_condition_validation_2022","artifact_policy":"summary_only","member_areas":[[float(m) for m in range(10)] for _ in range(40)],"truth_area":[4.5]*40}
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); source_path=root/"source.json"; output=root/"output"; source_path.write_text(json.dumps(source),encoding="utf-8"); env=dict(os.environ); env["CONFORMAL_AREA_SOURCE_JSON"]=str(source_path)
            process=subprocess.run([sys.executable,str(RUNNER),"--source-experiment",source["source_experiment"],"--output-dir",str(output)],env=env,capture_output=True,text=True)
            self.assertEqual(process.returncode,0,process.stderr); self.assertEqual({p.name for p in output.iterdir()},{"conformal_summary.json","conformal_per_case.csv"}); self.assertEqual(json.loads((output/"conformal_summary.json").read_text())["decision"],"CONFORMAL_USEFUL")
    def test_malformed_intervals_fail_closed(self):
        with self.assertRaisesRegex(ValueError,"lower endpoint"): evaluate([2.0]*40,[1.0]*40,[1.5]*40)
if __name__=="__main__": unittest.main()

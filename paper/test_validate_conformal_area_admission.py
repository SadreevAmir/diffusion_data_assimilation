from __future__ import annotations
import hashlib,json,tempfile,unittest
from pathlib import Path
from paper.conformal_area_reference import frozen_contract_sha256
from paper.conformal_area_runner_prototype import run
from paper.validate_conformal_area_admission import directory_sha256,load_and_validate,load_and_validate_combined

def valid_record():
    return {"reviewed_mode":"validation_reviewed_conformal_area","publication_commit":"a"*40,
      "runner_sha256":"b"*64,"contract_sha256":frozen_contract_sha256(),
      "synthetic_result_sha256":"c"*64,"test_command":"python3 trusted_test.py",
      "test_sentinel":"trusted conformal adapter: PASS","decision_bearing_validation":"PASS","deviations":[]}

class AdmissionTests(unittest.TestCase):
    def check(self,record):
        with tempfile.TemporaryDirectory() as temporary:
            path=Path(temporary)/"record.json"; path.write_text(json.dumps(record),encoding="utf-8")
            return load_and_validate(path)
    def test_exact_record_passes(self): self.assertEqual(self.check(valid_record()),"validation_reviewed_conformal_area")
    def test_deviation_fails_closed(self):
        record=valid_record(); record["deviations"]=["changed fold"]
        with self.assertRaisesRegex(ValueError,"empty list"): self.check(record)
    def test_contract_substitution_fails_closed(self):
        record=valid_record(); record["contract_sha256"]="d"*64
        with self.assertRaisesRegex(ValueError,"frozen local contract"): self.check(record)
    def test_non_object_fails_closed(self):
        with self.assertRaisesRegex(ValueError,"JSON object"): self.check([])
    def compact_directory(self,root):
        source=root/"source.json"; output=root/"output"
        source.write_text(json.dumps({"source_experiment":"joint_full_condition_validation_2022","artifact_policy":"summary_only","member_areas":[[float(m) for m in range(10)] for _ in range(40)],"truth_area":[4.5]*40}),encoding="utf-8")
        run(source,output,"joint_full_condition_validation_2022")
        return output
    def test_combined_admission_binds_both_exact_inputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); record=root/"admission.json"; record.write_text(json.dumps(valid_record()),encoding="utf-8"); output=self.compact_directory(root)
            result=load_and_validate_combined(record,output)
            self.assertEqual(result.reviewed_mode,"validation_reviewed_conformal_area")
            self.assertEqual(result.admission_record_sha256,hashlib.sha256(record.read_bytes()).hexdigest())
            self.assertEqual(result.compact_directory_sha256,directory_sha256(output))
    def test_inconsistent_compact_decision_fails_closed(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary); record=root/"admission.json"; record.write_text(json.dumps(valid_record()),encoding="utf-8"); output=self.compact_directory(root)
            summary_path=output/"conformal_summary.json"; summary=json.loads(summary_path.read_text()); summary["decision"]="CONFORMAL_NEGATIVE"; summary_path.write_text(json.dumps(summary),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"frozen thresholds"): load_and_validate_combined(record,output)
if __name__=="__main__": unittest.main()

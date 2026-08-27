from __future__ import annotations
import json,tempfile,unittest
from pathlib import Path
from paper.conformal_area_reference import frozen_contract_sha256
from paper.validate_conformal_area_admission import load_and_validate

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
if __name__=="__main__": unittest.main()

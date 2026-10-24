import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

spec = importlib.util.spec_from_file_location("verify_artifacts", Path(__file__).with_name("verify_artifacts.py"))
artifacts = importlib.util.module_from_spec(spec)
spec.loader.exec_module(artifacts)


class ArtifactVerificationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        (self.root / "weights").mkdir()
        (self.root / "weights/policy.pt").write_bytes(b"known policy bytes")
        self.manifest = {"schema_version": 1, "release_id": "test",
                         "profiles": {"demo": {"artifacts": ["actor"]}},
                         "artifacts": [{"id": "actor", "path": "weights/policy.pt",
                                        "sha256": hashlib.sha256(b"known policy bytes").hexdigest(),
                                        "size_bytes": 18}]}
        self.path = self.root / "manifest.json"

    def load(self):
        self.path.write_text(json.dumps(self.manifest), encoding="utf-8")
        return artifacts.load_manifest(self.path)

    def status(self):
        return artifacts.verify(self.load(), self.root, "demo")["artifacts"][0]["status"]

    def test_correct_file(self):
        self.assertEqual(self.status(), "verified")

    def test_invalid_root_and_missing_hash_field(self):
        self.path.write_text('[]', encoding='utf-8')
        with self.assertRaises(ValueError):
            artifacts.load_manifest(self.path)
        del self.manifest['artifacts'][0]['sha256']
        with self.assertRaises(ValueError):
            self.load()

    def test_missing(self):
        (self.root / "weights/policy.pt").unlink()
        self.assertEqual(self.status(), "missing")

    def test_bad_bytes_same_length(self):
        (self.root / "weights/policy.pt").write_bytes(b"wrong policy bytes")
        self.assertEqual(self.status(), "hash_mismatch")

    def test_bad_length(self):
        (self.root / "weights/policy.pt").write_bytes(b"short")
        self.assertEqual(self.status(), "size_mismatch")

    def test_unknown_hash_is_not_a_pass(self):
        self.manifest["artifacts"][0]["sha256"] = None
        self.assertEqual(self.status(), "identity_not_pinned")

    def test_traversal_and_windows_paths(self):
        for path in ("../policy.pt", "/policy.pt", "C:/policy.pt", "a\\b", "a/../b", "a//b", "a/./b", ""):
            with self.subTest(path=path), self.assertRaises(ValueError):
                artifacts.safe_relative(path)

    def test_duplicate_destinations(self):
        self.manifest["artifacts"].append({"id": "other", "path": "Weights/POLICY.pt", "sha256": None})
        with self.assertRaises(ValueError):
            self.load()

    def test_duplicate_id(self):
        self.manifest["artifacts"].append({"id": "actor", "path": "other.pt", "sha256": None})
        with self.assertRaises(ValueError):
            self.load()

    def test_profile_rejects_unknown_or_duplicate_ids(self):
        for ids in (["missing"], ["actor", "actor"], []):
            self.manifest["profiles"]["demo"]["artifacts"] = ids
            with self.assertRaises(ValueError):
                self.load()

    def test_invalid_digest_and_size(self):
        for key, value in (("sha256", "abc"), ("sha256", 1), ("size_bytes", True), ("size_bytes", -1)):
            original = self.manifest["artifacts"][0][key]
            self.manifest["artifacts"][0][key] = value
            with self.assertRaises(ValueError):
                self.load()
            self.manifest["artifacts"][0][key] = original

    def test_duplicate_json_keys(self):
        self.path.write_text('{"schema_version":1,"schema_version":1}', encoding="utf-8")
        with self.assertRaises(ValueError):
            artifacts.load_manifest(self.path)

    def test_unknown_profile(self):
        with self.assertRaises(ValueError):
            artifacts.verify(self.load(), self.root, "unknown")

    def test_report_never_overwrites_artifact(self):
        self.load()
        target = self.root / "weights/policy.pt"
        self.assertEqual(artifacts.main(["--manifest", str(self.path), "--root", str(self.root), "--report", str(target)]), 2)
        self.assertEqual(target.read_bytes(), b"known policy bytes")

    def test_existing_report_is_preserved(self):
        self.load()
        report = self.root / "report.json"
        report.write_text("preserve", encoding="utf-8")
        self.assertEqual(artifacts.main(["--manifest", str(self.path), "--root", str(self.root), "--report", str(report)]), 2)
        self.assertEqual(report.read_text(encoding="utf-8"), "preserve")

    def test_cli_exit_codes_and_report(self):
        self.load()
        report = self.root / "report.json"
        args = ["--manifest", str(self.path), "--root", str(self.root), "--report", str(report)]
        self.assertEqual(artifacts.main(args), 0)
        self.assertTrue(json.loads(report.read_text(encoding="utf-8"))["complete"])
        (self.root / "weights/policy.pt").unlink()
        self.assertEqual(artifacts.main(args[:-2]), 1)

    def test_symlink_escape(self):
        outside = self.root.parent / (self.root.name + "-outside.bin")
        outside.write_bytes(b"known policy bytes")
        self.addCleanup(lambda: outside.unlink(missing_ok=True))
        target = self.root / "weights/policy.pt"
        target.unlink()
        try:
            target.symlink_to(outside)
        except OSError:
            self.skipTest("Symlinks require platform privileges")
        self.assertEqual(self.status(), "path_escapes_root")


if __name__ == "__main__":
    unittest.main()

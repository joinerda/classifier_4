from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import prepare_train_test


class PrepareTrainTestColormapWorkflowTest(unittest.TestCase):
    def test_colorscale_workflow_writes_env_file(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            source_root = tmp_path / "source_dataset"
            output_root = tmp_path / "colorscaled_dataset"
            env_file = tmp_path / "train_test_env.sh"
            source_root.mkdir()

            captured: dict[str, Path | bool | int | float | str] = {}

            def fake_colorscale_build(*, args, source_root, output_root, color_transfer_module):
                captured["source_root"] = source_root
                captured["output_root"] = output_root
                captured["color_transfer_module"] = color_transfer_module
                captured["group_by_magnification"] = args.group_by_magnification
                captured["n_bg_clusters"] = args.n_bg_clusters
                manifest_path = output_root / "manifest.csv"
                output_root.mkdir(parents=True, exist_ok=False)
                manifest_path.write_text("path,class_id,domain,split\n", encoding="utf-8")
                return manifest_path

            argv = [
                "prepare_train_test.py",
                "--colorscale-source-root",
                str(source_root),
                "--colorscale-output-root",
                str(output_root),
                "--train-domains",
                "lab,wild",
                "--test-domains",
                "wild",
                "--env-file",
                str(env_file),
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(prepare_train_test, "_run_colorscale_build", side_effect=fake_colorscale_build):
                    rc = prepare_train_test.main()

            self.assertEqual(rc, 0)
            self.assertEqual(captured["source_root"], source_root.resolve())
            self.assertEqual(captured["output_root"], output_root.resolve())
            self.assertEqual(
                captured["color_transfer_module"],
                prepare_train_test.DEFAULT_COLOR_TRANSFER_MODULE.resolve(),
            )
            self.assertTrue(captured["group_by_magnification"])
            self.assertEqual(captured["n_bg_clusters"], 4)
            self.assertTrue((output_root / "manifest.csv").exists())

            env_text = env_file.read_text(encoding="utf-8")
            self.assertIn(f'${{IMAGE_ROOT:={str(output_root.resolve())}}}', env_text)
            self.assertIn(f'${{MANIFEST:={str((output_root / "manifest.csv").resolve())}}}', env_text)
            self.assertIn('${TRAIN_DOMAINS:=lab,wild}', env_text)
            self.assertIn('${TEST_DOMAINS:=wild}', env_text)
            self.assertIn('${RUN_COMMENT:=lab,wild__wild__colorscaled}', env_text)

    def test_manifest_colormap_workflow_builds_base_and_derived_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            image_root = tmp_path / "dataset"
            env_file = tmp_path / "train_test_env.sh"
            base_manifest = image_root / "manifest_combined__wild.csv"
            derived_manifest = image_root / "manifest_combined__wild__colormapped.csv"
            metadata_path = image_root / "metadata.json"
            image_root.mkdir()
            metadata_path.write_text("[]\n", encoding="utf-8")

            calls: list[tuple[str, Path]] = []

            def fake_manifest_build(*, args, image_root, manifest_path, metadata_path):
                calls.append(("manifest", manifest_path))
                manifest_path.write_text("path,class_id,domain,split\n", encoding="utf-8")
                return 0

            def fake_manifest_colormap_build(*, args, image_root, manifest_path, output_manifest_path, output_dir):
                calls.append(("colormap", output_manifest_path))
                self.assertEqual(output_dir, "colormapped_manifest_combined__wild")
                output_manifest_path.write_text("path,class_id,domain,split\n", encoding="utf-8")
                return output_manifest_path

            argv = [
                "prepare_train_test.py",
                "--image-root",
                str(image_root),
                "--metadata",
                str(metadata_path),
                "--manifest",
                str(base_manifest),
                "--train-domains",
                "lab,wild",
                "--test-domains",
                "wild",
                "--env-file",
                str(env_file),
                "--colormap-output-dir",
                "colormapped_manifest_combined__wild",
            ]
            with patch.object(sys, "argv", argv):
                with patch.object(prepare_train_test, "_run_manifest_build", side_effect=fake_manifest_build):
                    with patch.object(
                        prepare_train_test,
                        "_run_manifest_colormap_build",
                        side_effect=fake_manifest_colormap_build,
                    ):
                        rc = prepare_train_test.main()

            self.assertEqual(rc, 0)
            self.assertEqual(calls, [("manifest", base_manifest.resolve()), ("colormap", derived_manifest.resolve())])
            env_text = env_file.read_text(encoding="utf-8")
            self.assertIn(f'${{IMAGE_ROOT:={str(image_root.resolve())}}}', env_text)
            self.assertIn(f'${{MANIFEST:={str(derived_manifest.resolve())}}}', env_text)
            self.assertIn('${TRAIN_DOMAINS:=lab,wild}', env_text)
            self.assertIn('${TEST_DOMAINS:=wild}', env_text)
            self.assertIn('${RUN_COMMENT:=lab,wild__wild__colorscaled}', env_text)


if __name__ == "__main__":
    unittest.main()

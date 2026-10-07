"""Multiscale decomposition and cache-free condition tests: no model, optimizer, or BEMD execution."""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from cine_4ch.bemd_dataset import BEMDSliceDataset, multiscale_ablation_specs, required_bimf_count
from cine_4ch.io import CasePair
from scripts import run_bemd_ablation as cli
from src.preprocessing import multiscale


def _synthetic_frame(seed: int, shape=(96, 80)) -> np.ndarray:
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[: shape[0], : shape[1]]
    blobs = sum(np.exp(-((y - rng.uniform(0, shape[0])) ** 2 + (x - rng.uniform(0, shape[1])) ** 2)
                       / (2 * rng.uniform(4, 20) ** 2)) for _ in range(6))
    return (2000 * blobs + 50 * rng.standard_normal(shape) + 500).astype(np.float32)


class DecompositionTests(unittest.TestCase):
    def test_exact_bounded_reconstruction(self):
        for decompose in (multiscale.gaussian_bands, multiscale.fabemd):
            d = decompose(_synthetic_frame(0))
            self.assertLess(d.reconstruction_rmse(), 1e-12)
            self.assertAlmostEqual(float(d.original.min()), 0.0)
            self.assertAlmostEqual(float(d.original.max()), 1.0)
            for c in d.components:
                self.assertLessEqual(float(np.abs(c).max()), 1.0 + 1e-9)

    def test_fabemd_windows_grow_and_reach_catalog_depth(self):
        d = multiscale.fabemd(_synthetic_frame(1))
        windows = d.meta["windows"]
        self.assertGreaterEqual(d.n_components, 4)
        self.assertTrue(all(w % 2 == 1 for w in windows))
        self.assertTrue(all(b > a for a, b in zip(windows, windows[1:])))

    def test_removal_is_amplitude_faithful(self):
        d = multiscale.gaussian_bands(_synthetic_frame(2))
        everything = multiscale.subtract_components(d.original, d.components, range(d.n_components))
        expected = multiscale.safe_minmax_normalize(d.residual, clip=True)
        np.testing.assert_allclose(everything, expected, atol=1e-6)
        one = multiscale.subtract_components(d.original, d.components, [3])
        self.assertGreater(float(np.abs(one - d.original).mean()), 1e-3)
        with self.assertRaises(IndexError):
            multiscale.subtract_components(d.original, d.components, [d.n_components])


class CacheFreeConditionTests(unittest.TestCase):
    def setUp(self):
        test_dir = Path(__file__).resolve().parent
        self.temp = tempfile.TemporaryDirectory(dir=test_dir)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        raw = np.stack([_synthetic_frame(3), _synthetic_frame(4)], axis=-1)
        image, anno = self.root / "image.nii", self.root / "anno.nii"
        nib.save(nib.Nifti1Image(raw, np.eye(4)), image)
        nib.save(nib.Nifti1Image(np.zeros_like(raw), np.eye(4)), anno)
        self.case = CasePair("CINE_4CH_001", "001", image, anno)
        self.out = self.root / "working"

    def test_every_condition_builds_without_bemd_cache(self):
        missing_cache = self.root / "no_cache_here"
        specs = multiscale_ablation_specs()
        self.assertEqual(required_bimf_count(specs), 0)
        with patch("src.preprocessing.bemd_square_pad.decompose_bemd_square_pad",
                   side_effect=AssertionError("No BEMD")):
            inputs = {}
            for spec in specs:
                ds = BEMDSliceDataset([self.case], spec, bemd_cache_root=missing_cache,
                                     enhanced_cache_root=self.out,
                                     excluded_frames=frozenset({(self.case.stem, 1)}))
                self.assertEqual(ds.index, [(0, 0)])
                image, _, _, _ = ds[0]
                self.assertEqual(tuple(image.shape), (1, 160, 160))
                self.assertTrue(np.isfinite(image.numpy()).all())
                inputs[spec.run_id] = image.numpy()
        self.assertFalse(missing_cache.exists())
        for run_id, arr in inputs.items():
            if run_id != "original":
                self.assertGreater(float(np.abs(arr - inputs["original"]).mean()), 1e-3, run_id)

    def test_config_controls_and_validate_only_needs_no_cache(self):
        config = Path(__file__).resolve().parents[1] / "configs/multiscale_ablation.yaml"
        args = cli._apply_yaml(cli.parse_args(["--config", str(config)]))
        self.assertEqual(args.catalog, "multiscale")
        self.assertEqual((args.epochs, args.batch_size, args.lr, args.seed), (15, 4, 0.001, 42))
        self.assertEqual(set(args.runs), {s.run_id for s in multiscale_ablation_specs()})

        splits = self.root / "splits.csv"
        splits.write_text("unused by mocked split loader")
        groups, start = {}, 0
        for name, count in [("train", 74), ("val", 16), ("test", 15)]:
            groups[name] = [CasePair(f"case_{i}", str(i), self.case.image_path, self.case.anno_path)
                            for i in range(start, start + count)]
            start += count
        argv = ["run_bemd_ablation.py", "--catalog", "multiscale", "--cache-root", str(self.root / "absent"),
                "--splits-csv", str(splits), "--output-root", str(self.out), "--validate-only"]
        with patch("sys.argv", argv), \
             patch.object(cli, "load_split_cases", side_effect=lambda p, s, **kw: groups[s]), \
             patch.object(cli, "audit_cache", side_effect=AssertionError("No cache audit")), \
             patch.object(cli, "train_bemd_run", side_effect=AssertionError("No training")), \
             patch.object(cli, "evaluate_bemd_test", side_effect=AssertionError("No evaluation")):
            self.assertEqual(cli.main(), 0)
        self.assertFalse(self.out.exists())


if __name__ == "__main__":
    unittest.main()

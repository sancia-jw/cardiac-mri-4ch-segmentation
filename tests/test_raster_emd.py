"""Gastro raster-EMD conditions: decomposition, amplitude-faithful removal, catalog, parallel cache."""
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from cine_4ch import bemd_dataset
from cine_4ch.bemd_dataset import BEMDSliceDataset, raster_emd_ablation_specs, required_bimf_count
from cine_4ch.io import CasePair
from scripts import run_bemd_ablation as cli
from src.preprocessing import multiscale
from src.preprocessing.emd_enhancement import EMEnhancementConfig, enhance_mri_slice
from tests.test_multiscale import _synthetic_frame


class RasterEMDTests(unittest.TestCase):
    def test_exact_reconstruction_in_image_units(self):
        d = multiscale.decompose(multiscale.RASTER_EMD_ID, _synthetic_frame(0))
        self.assertLess(d.reconstruction_rmse(), 1e-12)
        self.assertGreaterEqual(d.n_components, 4)
        self.assertEqual(d.meta["n_imfs"], d.n_components)
        for c in d.components:
            self.assertLessEqual(float(np.abs(c).max()), 2.0)

    def test_matches_gastro_emd2d_on_the_unit_frame(self):
        import emd

        frame = _synthetic_frame(1)
        d = multiscale.raster_emd(frame)
        ref = emd.sift.sift(d.original.reshape(-1), sift_thresh=1e-8)
        self.assertEqual(ref.shape[1], d.n_components)
        np.testing.assert_allclose(d.components[-1], ref[:, -1].reshape(frame.shape))

    def test_removal_moves_input_unlike_legacy_convention(self):
        frame = _synthetic_frame(2)
        d = multiscale.raster_emd(frame)
        for indices in ([0], [1], [-1, -2, -3]):
            out = multiscale.subtract_components(d.original, d.components, indices)
            self.assertGreater(float(np.abs(out - d.original).mean()), 1e-2, indices)
        legacy = enhance_mri_slice(frame, EMEnhancementConfig(mode="subtract", imf_indices=[-1, -2, -3]))
        self.assertLess(float(np.abs(legacy - d.original).mean()), 5e-3)


class RasterCatalogTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory(dir=Path(__file__).resolve().parent)
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name).resolve()
        self.cases = []
        for n in range(3):
            raw = np.stack([_synthetic_frame(10 * n + t, shape=(48, 40)) for t in range(3)], axis=-1)
            image, anno = self.root / f"image_{n}.nii", self.root / f"anno_{n}.nii"
            nib.save(nib.Nifti1Image(raw, np.eye(4)), image)
            nib.save(nib.Nifti1Image(np.zeros_like(raw), np.eye(4)), anno)
            self.cases.append(CasePair(f"CINE_4CH_00{n}", str(n), image, anno))

    def _build(self, spec, cache, **env):
        with patch.dict(os.environ, env):
            ds = BEMDSliceDataset(self.cases, spec, bemd_cache_root=self.root / "absent", enhanced_cache_root=cache)
        return np.stack([ds[i][0].numpy() for i in range(len(ds))])

    def test_catalog_needs_no_cache_and_parallel_matches_serial(self):
        specs = raster_emd_ablation_specs()
        self.assertEqual([s.run_id for s in specs][0], "original")
        self.assertEqual(required_bimf_count(specs), 0)
        trend = next(s for s in specs if s.run_id == "subtract_remd_trend")
        self.assertEqual(trend.bimf_indices, (-1, -2, -3))
        serial = self._build(trend, self.root / "serial", ONTHEFLY_WORKERS="1")
        with patch.object(bemd_dataset, "PARALLEL_MIN_FRAMES", 1):
            parallel = self._build(trend, self.root / "parallel", ONTHEFLY_WORKERS="2")
        np.testing.assert_array_equal(serial, parallel)
        self.assertEqual(serial.shape[0], 9)
        original = self._build(specs[0], self.root / "orig")
        self.assertGreater(float(np.abs(serial - original).mean()), 1e-3)

    def test_tail_condition_removes_imf5_to_last(self):
        spec = next(s for s in raster_emd_ablation_specs() if s.run_id == "subtract_remd_5plus")
        frame = _synthetic_frame(5)
        d = multiscale.raster_emd(frame)
        self.assertGreater(d.n_components, 5)
        expected = multiscale.subtract_components(d.original, d.components, range(5, d.n_components))
        np.testing.assert_allclose(bemd_dataset.enhance_frame_from_raw(frame, spec), expected)
        self.assertNotEqual(spec.cache_key(), raster_emd_ablation_specs()[0].cache_key())

    def test_lowfreq_config(self):
        config = Path(__file__).resolve().parents[1] / "configs/raster_emd_lowfreq.yaml"
        args = cli._apply_yaml(cli.parse_args(["--config", str(config)]))
        self.assertEqual(args.catalog, "raster_emd")
        self.assertEqual((args.epochs, args.batch_size, args.lr, args.seed), (15, 4, 0.001, 42))
        self.assertTrue(set(args.runs) <= {s.run_id for s in raster_emd_ablation_specs()})

    def test_config_controls(self):
        config = Path(__file__).resolve().parents[1] / "configs/raster_emd_ablation.yaml"
        args = cli._apply_yaml(cli.parse_args(["--config", str(config), "--seed", "43"]))
        self.assertEqual(args.catalog, "raster_emd")
        self.assertEqual((args.epochs, args.batch_size, args.lr, args.seed), (15, 4, 0.001, 43))
        self.assertEqual(args.runs, [s.run_id for s in raster_emd_ablation_specs()][:5])


if __name__ == "__main__":
    unittest.main()

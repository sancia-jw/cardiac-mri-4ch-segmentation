"""Training-recipe pipeline: cine loop detection, phase context, augmentation, models, metrics, CLI."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

from cine_4ch import improve
from cine_4ch.io import CasePair


def _cine(slices=3, phases=25, h=40, w=36, seed=0):
    """Synthetic (H, W, T) cine: each slice is a different pattern, phases change slowly."""
    rng = np.random.default_rng(seed)
    frames = []
    for s in range(slices):
        base = rng.uniform(100, 1000, size=(h, w))
        for p in range(phases):
            frames.append(base * (1 + 0.02 * np.sin(2 * np.pi * p / phases)))
    return np.stack(frames, axis=-1).astype(np.float32)


def _write_case(root, stem, image, label):
    for sub, arr in (("image", image), ("anno", label)):
        (root / sub).mkdir(parents=True, exist_ok=True)
        nib.save(nib.Nifti1Image(arr, np.eye(4)), str(root / sub / f"{stem}.nii.gz"))
    return CasePair(stem, stem[-3:], root / "image" / f"{stem}.nii.gz", root / "anno" / f"{stem}.nii.gz")


class SliceBlockTests(unittest.TestCase):
    def test_detects_slice_jumps(self):
        self.assertEqual(improve.slice_blocks(_cine(3, 25)), [(0, 25), (25, 50), (50, 75)])
        self.assertEqual(improve.slice_blocks(_cine(3, 30)), [(0, 30), (30, 60), (60, 90)])

    def test_falls_back_to_equal_blocks(self):
        flat = np.ones((8, 8, 90), dtype=np.float32)
        self.assertEqual(improve.slice_blocks(flat), [(0, 30), (30, 60), (60, 90)])

    def test_neighbours_wrap_within_loop(self):
        nbr = improve.neighbour_index([(0, 25), (25, 50)], improve.context_offsets(5))
        np.testing.assert_array_equal(nbr[0], [23, 24, 0, 1, 2])
        np.testing.assert_array_equal(nbr[26], [49, 25, 26, 27, 28])
        np.testing.assert_array_equal(improve.neighbour_index([(0, 4)], [0])[:, 0], np.arange(4))
        with self.assertRaises(ValueError):
            improve.context_offsets(2)


class PreprocessTests(unittest.TestCase):
    def test_normalize_volume(self):
        out = improve.normalize_volume(_cine())
        self.assertAlmostEqual(float(out.mean()), 0.0, places=4)
        self.assertAlmostEqual(float(out.std()), 1.0, places=3)

    def test_load_cases_keeps_native_labels(self):
        with tempfile.TemporaryDirectory() as tmp:
            image = _cine(3, 10, h=30, w=26)
            label = np.zeros_like(image, dtype=np.float32)
            label[5:15, 4:12] = 2
            case = _write_case(Path(tmp), "CINE_4CH_001", image, label)
            canvas = improve.canvas_shape([(30, 26)])
            self.assertEqual(canvas, (32, 32))
            vol = improve.load_cases([case], "canvas", 3, canvas)[0]
            self.assertEqual(vol.image.shape, (30, 32, 32))
            self.assertEqual(vol.native_label.shape, (30, 30, 26))
            top, left = vol.offset
            np.testing.assert_array_equal(vol.label[:, top:top + 30, left:left + 26], vol.native_label)
            up = improve.load_cases([case], "canvas", 1, improve.canvas_shape([(45, 39)]), upsample=1.5)[0]
            self.assertEqual(up.scaled_hw, (45, 39))
            self.assertEqual(up.image.shape, (30, 64, 64))
            self.assertEqual(up.native_label.shape, (30, 30, 26))
            model = torch.nn.Conv2d(1, 6, 1)
            probs = improve.predict_case([model], up, "canvas", torch.device("cpu"))
            self.assertEqual(probs.shape, (30, 6, 30, 26))
            legacy = improve.load_cases([case], "legacy", 1)[0]
            self.assertEqual(legacy.image.shape, (30, *improve.LEGACY_SIZE))
            self.assertLessEqual(float(legacy.image.max()), 1.0)


class AugmentTests(unittest.TestCase):
    def test_spatial_transform_keeps_labels_on_image(self):
        torch.manual_seed(0)
        label = torch.zeros(6, 48, 40, dtype=torch.long)
        label[:, 10:30, 8:25] = 3
        image = (label == 3).float()[:, None].repeat(1, 3, 1, 1) * 5.0
        cfg = improve.AugmentConfig(affine=1.0, elastic=1.0)
        img, lab = improve.augment(image, label, cfg)
        inside = img[:, 1][lab == 3]
        self.assertGreater(float(inside.mean()), 4.0)
        self.assertLess(float(img[:, 1][lab == 0].mean()), 1.0)
        torch.testing.assert_close(img[:, 0], img[:, 2])  # one transform per sample

    def test_no_aug_is_identity_and_legacy_flip_mirrors(self):
        image = torch.randn(4, 1, 16, 16)
        label = torch.randint(0, 6, (4, 16, 16))
        img, lab = improve.augment(image, label, improve.AugmentConfig())
        torch.testing.assert_close(img, image)
        img, lab = improve.augment(image, label, improve.AugmentConfig(hflip=1.0))
        torch.testing.assert_close(lab, label.flip(-1))

    def test_strong_aug_runs(self):
        img, lab = improve.augment(torch.randn(3, 5, 32, 32), torch.zeros(3, 32, 32, dtype=torch.long),
                                   improve.STRONG_AUG)
        self.assertEqual(img.shape, (3, 5, 32, 32))
        self.assertTrue(torch.isfinite(img).all())


class ModelMetricTests(unittest.TestCase):
    def test_resunet_shape(self):
        out = improve.ResUNet(5)(torch.randn(2, 5, 64, 32))
        self.assertEqual(out.shape, (2, 6, 64, 32))

    def test_dice_and_largest_component(self):
        gt = np.zeros((1, 10, 10), dtype=np.uint8)
        gt[0, 2:6, 2:6] = 1
        pred = gt.copy()
        pred[0, 8, 8] = 1  # stray blob
        self.assertAlmostEqual(improve.dice_per_class(gt, gt)[0], 1.0)
        self.assertTrue(np.isnan(improve.dice_per_class(gt, gt)[1]))
        self.assertLess(improve.dice_per_class(pred, gt)[0], 1.0)
        self.assertAlmostEqual(improve.dice_per_class(improve.largest_component(pred), gt)[0], 1.0)

    def test_lr_schedule(self):
        cfg = improve.RunConfig("x", epochs=10, steps_per_epoch=10, warmup_epochs=1)
        self.assertAlmostEqual(improve.lr_factor(cfg, 9, 100), 1.0)
        self.assertLess(improve.lr_factor(cfg, 99, 100), 0.01)
        with self.assertRaises(ValueError):
            improve.run_config_from_dict("x", {"epoch": 3})


class ConfigTests(unittest.TestCase):
    def test_repo_config_parses(self):
        import yaml

        raw = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs/improve.yaml").read_text())
        conds = {n: improve.run_config_from_dict(n, s) for n, s in raw["conditions"].items()}
        self.assertEqual(raw["seeds"], [42, 43, 44])
        self.assertEqual(conds["legacy"].preset, "legacy")
        self.assertEqual(conds["recipe_t5"].context, 5)
        self.assertTrue(conds["t5_full"].train_on_val)
        raw2 = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs/improve_round2.yaml").read_text())
        conds2 = {n: improve.run_config_from_dict(n, s) for n, s in raw2["conditions"].items()}
        self.assertEqual(conds2["t5_extra_hr"].upsample, 1.5)
        self.assertEqual(conds2["t5_hr"].extra_train, [])
        raw3 = yaml.safe_load((Path(__file__).resolve().parent.parent / "configs/improve_round3.yaml").read_text())
        conds3 = {n: improve.run_config_from_dict(n, s) for n, s in raw3["conditions"].items()}
        self.assertEqual(conds3["t5_tc_smooth"].unlabelled, ["4CH_TST"])
        self.assertEqual(conds3["recipe_cmrx"].external, "4CH_CMRX")


if __name__ == "__main__":
    unittest.main()

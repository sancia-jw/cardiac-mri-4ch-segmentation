"""Temporal-consistency / partial-label training and the CMRxRecon converter."""
import tempfile
import unittest
from pathlib import Path

import nibabel as nib
import numpy as np
import torch
import torch.nn.functional as F

from cine_4ch import improve, improve_semi
from scripts import convert_cmrxrecon as conv


def _vol(t=10, h=32, w=32, seed=0):
    rng = np.random.default_rng(seed)
    label = np.zeros((t, h, w), dtype=np.uint8)
    for f in range(t):
        label[f, 8:20, 6:14 + f % 3] = 1
        label[f, 20:28, 18:26] = 5
    image = rng.normal(size=(t, h, w)).astype(np.float32) + label
    blocks = [(0, t // 2), (t // 2, t)]
    return improve.CaseVolume("c", image, label, label, improve.neighbour_index(blocks, [0]), (h, w), (0, 0),
                              triplets=improve_semi.triplet_index(blocks))


class TripletTests(unittest.TestCase):
    def test_triplets_wrap_in_loop(self):
        trip = improve_semi.triplet_index([(0, 5), (5, 10)])
        np.testing.assert_array_equal(trip[0], [4, 0, 1])
        np.testing.assert_array_equal(trip[9], [8, 9, 5])

    def test_triplet_batch_keeps_labels_aligned(self):
        torch.manual_seed(0)
        frames = improve_semi.TripletFrames([_vol()], torch.device("cpu"))
        x, y = frames.triplet_batch(torch.tensor([1, 3, 7]), improve.STRONG_AUG)
        self.assertEqual(x.shape, (9, 1, 32, 32))
        self.assertEqual(y.shape, (9, 32, 32))
        self.assertTrue(set(np.unique(y.numpy())) <= {0, 1, 5})
        x0, y0 = frames.triplet_batch(torch.tensor([1]), improve.AugmentConfig())
        np.testing.assert_array_equal(y0.numpy(), frames.labels[torch.tensor([0, 1, 2])].long().numpy())


class LossTests(unittest.TestCase):
    def test_area_losses(self):
        labels = torch.zeros(6, 16, 16, dtype=torch.long)
        for i in range(6):
            labels[i, 2:8, 2:4 + i] = 1
        perfect = F.one_hot(labels, 6).permute(0, 3, 1, 2).float() * 30
        self.assertLess(float(improve_semi.area_dynamics_loss(perfect, labels)), 1e-3)
        shuffled = perfect[[2, 1, 0, 5, 4, 3]]
        self.assertGreater(float(improve_semi.area_dynamics_loss(shuffled, labels)), 0.05)
        self.assertLess(float(improve_semi.smoothness_loss(perfect, 0.04)), 1e-3)  # linear growth
        jumpy = perfect.clone()
        jumpy[1] = F.one_hot(torch.zeros(16, 16, dtype=torch.long), 6).permute(2, 0, 1).float() * 30
        self.assertGreater(float(improve_semi.smoothness_loss(jumpy, 0.04)), 0.1)

    def test_marginal_loss_ignores_myo_vs_background(self):
        labels = torch.zeros(2, 8, 8, dtype=torch.long)
        labels[:, 2:5, 2:5] = 1
        logits = torch.zeros(2, 6, 8, 8)
        logits[:, 1] = 8 * (labels == 1)
        as_bg = logits.clone()
        as_bg[:, 0] += 8 * (labels == 0)
        as_myo = logits.clone()
        as_myo[:, 2] += 8 * (labels == 0)
        a = improve_semi.marginal_loss(as_bg, labels, [0, 2])
        b = improve_semi.marginal_loss(as_myo, labels, [0, 2])
        self.assertAlmostEqual(float(a), float(b), places=4)
        self.assertGreater(float(improve.combined_loss(as_myo, labels)), float(improve.combined_loss(as_bg, labels)))

    def test_train_semi_runs(self):
        cfg = improve.RunConfig("x", context=1, epochs=1, steps_per_epoch=2, batch_size=6, warmup_epochs=0,
                                temporal_weight=0.5, smooth_weight=0.5, unlabelled_triplets=1,
                                external="EXT", external_frac=0.34)
        cfg.model = "resunet"
        with tempfile.TemporaryDirectory() as tmp:
            model = improve_semi.train_semi(cfg, [_vol()], [], torch.device("cpu"), 0, Path(tmp),
                                            unlabelled_vols=[_vol(seed=1)], external_vols=[_vol(seed=2)],
                                            log=lambda *_: None)
            self.assertTrue((Path(tmp) / "final.pt").exists())
            self.assertIn("temporal", (Path(tmp) / "train_log.csv").read_text().splitlines()[0])
        self.assertIsNotNone(model)


class ConverterTests(unittest.TestCase):
    def _heart(self):
        lab = np.zeros((40, 40), dtype=np.int64)
        lab[22:34, 14:24] = conv.LV   # apex down
        lab[6:16, 14:24] = conv.LA    # LA above LV
        lab[22:34, 26:34] = conv.RV   # RV to the right
        lab[6:16, 26:34] = conv.RA
        return lab

    def test_orientation_recovers_transform(self):
        ref = conv.chamber_vectors(self._heart())
        for k in range(8):
            moved = conv.dihedral(self._heart(), k)
            fixed = conv.dihedral(moved, conv.best_orientation(moved, ref))
            np.testing.assert_array_equal(fixed, self._heart())

    def test_convert_subject(self):
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            heart = self._heart()
            inv = {v: k for k, v in conv.CMRX_TO_PROJECT.items()}
            raw = np.zeros((40, 40, 3, 4), dtype=np.int16)  # x, y, slice group, frame
            for t in range(3):
                raw[:, :, 2, t] = np.vectorize(lambda v: inv.get(v, 0))(np.rot90(heart, 1))
            raw[:, :, 0, :] = 3  # a 2CH-like slice without RA
            image = np.random.default_rng(0).normal(size=raw.shape).astype(np.float32) + raw
            nib.save(nib.Nifti1Image(image, np.diag([1.5, 1.5, 8, 1])), str(tmp / "img.nii.gz"))
            nib.save(nib.Nifti1Image(raw, np.diag([1.5, 1.5, 8, 1])), str(tmp / "lab.nii.gz"))
            ref = conv.chamber_vectors(heart)
            img, lab, info = conv.convert_subject(tmp / "img.nii.gz", tmp / "lab.nii.gz", ref, 1.5, (32, 32))
            self.assertEqual(info["slice_group"], 2)
            self.assertEqual(info["frames"], 3)
            self.assertEqual(img.shape, (32, 32, 3))
            self.assertEqual(set(np.unique(lab)), {0, 1, 3, 4, 5})
            np.testing.assert_allclose(conv.chamber_vectors(lab[..., 0]), ref, atol=0.05)


if __name__ == "__main__":
    unittest.main()

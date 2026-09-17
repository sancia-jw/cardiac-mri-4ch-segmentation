"""Portability/preflight tests only: no model, optimizer, or BEMD execution."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import nibabel as nib
import numpy as np

from cine_4ch.bemd_dataset import BEMDSliceDataset, default_bemd_ablation_specs
from cine_4ch.bemd_validation import audit_cache, load_exclusions
from cine_4ch.io import CasePair
from scripts import run_bemd_ablation as cli


class PortabilityTests(unittest.TestCase):
    def setUp(self):
        test_dir = Path(__file__).resolve().parent
        self.temp = tempfile.TemporaryDirectory(dir=test_dir)
        self.root = Path(self.temp.name).resolve()
        assert self.root.is_relative_to(test_dir)
        self.addCleanup(self.temp.cleanup)
        self.cache = self.root / "readonly_input"
        self.out = self.root / "working"
        self.stem = "CINE_4CH_001"
        raw = np.arange(96, dtype=np.float32).reshape(8, 6, 2)
        image = self.root / "image.nii"
        anno = self.root / "anno.nii"
        nib.save(nib.Nifti1Image(raw, np.eye(4)), image)
        nib.save(nib.Nifti1Image(np.zeros_like(raw), np.eye(4)), anno)
        self.case = CasePair(self.stem, "001", image, anno)
        for f in range(2):
            directory = self.cache / self.stem / f"frame_{f:03d}"
            directory.mkdir(parents=True)
            for name in ["original", "residual", "bimf_0", "bimf_1", "bimf_2"]:
                arr = raw[..., f].astype(np.float64) if name in ("original", "residual") else np.zeros((8, 6))
                np.save(directory / f"{name}.npy", arr)
            meta = {"method_id": "bemd_default_square_pad", "case_stem": self.stem,
                    "frame_idx": f, "status": "ok" if f == 0 else "bad_recon", "n_bimf": 3,
                    "reconstruction": {"rmse": 0.0 if f == 0 else 1.0},
                    "original_shape": [8, 6], "padded_shape": [8, 8], "pad_hw": [0, 2]}
            (directory / "metadata.json").write_text(json.dumps(meta))
        self.excluded = frozenset({(self.stem, 1)})

    def test_external_cache_exclusions_and_all_five_inputs(self):
        snapshot = {p: p.read_bytes() for p in self.cache.rglob('*') if p.is_file()}
        with patch('src.preprocessing.bemd_square_pad.decompose_bemd_square_pad', side_effect=AssertionError('No BEMD')):
            for spec in default_bemd_ablation_specs():
                if spec.run_id == 'subtract_bimf_3':
                    with self.assertRaises(FileNotFoundError):
                        BEMDSliceDataset([self.case], spec, bemd_cache_root=self.cache,
                                         enhanced_cache_root=self.out, excluded_frames=self.excluded)
                    continue
                ds = BEMDSliceDataset([self.case], spec, bemd_cache_root=self.cache,
                                     enhanced_cache_root=self.out, excluded_frames=self.excluded)
                self.assertEqual(ds.index, [(0, 0)])
                image, mask, stem, frame = ds[0]
                self.assertEqual(tuple(image.shape), (1, 160, 160))
                self.assertTrue(np.isfinite(image.numpy()).all())
                self.assertEqual((stem, frame), (self.stem, 0))
        self.assertEqual(len(list(self.out.rglob('*.npy'))), 5)
        self.assertEqual(snapshot, {p: p.read_bytes() for p in self.cache.rglob('*') if p.is_file()})

    def test_audit_rejects_unlisted_failure_and_missing_component(self):
        report = audit_cache([self.case], self.cache, self.excluded, 3)
        self.assertEqual(report['usable_frames'], 1)
        self.assertEqual(report['bimf_availability'], {'0': 1, '1': 1, '2': 1, '3': 0})
        self.assertFalse(report['issues'])
        self.assertEqual(len(audit_cache([self.case], self.cache)['issues']), 1)
        (self.cache / self.stem / 'frame_000/bimf_2.npy').unlink()
        self.assertEqual(len(audit_cache([self.case], self.cache, self.excluded, 3)['issues']), 1)
        with self.assertRaises(ValueError):
            audit_cache([self.case], self.cache, frozenset({(self.stem, 99)}))

    def test_exclusion_file_requires_reason(self):
        path = self.root / 'exclude.json'
        payload = {'method_id': 'bemd_default_square_pad', 'excluded_frames': [
            {'case_stem': self.stem, 'frame_idx': 1, 'reason': 'reconstruction failure'}]}
        path.write_text(json.dumps(payload))
        self.assertEqual(load_exclusions(path), self.excluded)
        del payload['excluded_frames'][0]['reason']
        path.write_text(json.dumps(payload))
        with self.assertRaises(ValueError):
            load_exclusions(path)

    def test_audit_rejects_truncated_payload_and_nonfinite_rmse(self):
        component = self.cache / self.stem / 'frame_000/bimf_1.npy'
        original = component.read_bytes()
        component.write_bytes(original[:-8])
        report = audit_cache([self.case], self.cache, self.excluded, 3)
        self.assertEqual(report['usable_frames'], 0)
        self.assertTrue(report['issues'])
        component.write_bytes(original)
        meta_path = self.cache / self.stem / 'frame_000/metadata.json'
        meta = json.loads(meta_path.read_text())
        meta['reconstruction']['rmse'] = float('nan')
        meta_path.write_text(json.dumps(meta))
        self.assertTrue(audit_cache([self.case], self.cache, self.excluded, 3)['issues'])

    def test_primary_config_preserves_controls_and_secondary_support(self):
        config = Path(__file__).resolve().parents[1] / 'configs/bemd_ablation.yaml'
        args = cli._apply_yaml(cli.parse_args(['--config', str(config)]))
        self.assertEqual(args.runs, ['original', 'subtract_bimf_0', 'subtract_bimf_1',
                                    'subtract_bimf_2', 'subtract_bimf_0_1'])
        self.assertEqual((args.epochs, args.batch_size, args.lr, args.seed), (15, 4, 0.001, 42))
        self.assertIn('subtract_bimf_3', [spec.run_id for spec in default_bemd_ablation_specs()])

    def test_cli_overrides_yaml_paths_and_secondary_run(self):
        path = self.root / 'config.yaml'
        path.write_text('train:\n  cache_root: wrong\n  runs: [original]\n  epochs: 15\n')
        args = cli._apply_yaml(cli.parse_args(['--config', str(path), '--cache-root', str(self.cache),
                                             '--runs', 'subtract_bimf_3', '--validate-only']))
        self.assertEqual(args.cache_root, self.cache)
        self.assertEqual(args.runs, ['subtract_bimf_3'])
        self.assertEqual(args.epochs, 15)

    def test_validate_only_cannot_train_evaluate_or_create_outputs(self):
        splits = self.root / 'splits.csv'
        splits.write_text('unused by mocked split loader')
        groups = {}
        start = 0
        for name, count in [('train', 74), ('val', 16), ('test', 15)]:
            groups[name] = [CasePair(f'case_{i}', str(i), self.case.image_path, self.case.anno_path)
                            for i in range(start, start + count)]
            start += count
        argv = ['run_bemd_ablation.py', '--cache-root', str(self.cache), '--splits-csv', str(splits),
                '--output-root', str(self.out), '--runs', 'original', '--validate-only']
        with patch('sys.argv', argv), patch.object(cli, 'load_split_cases', side_effect=lambda p, s, **kw: groups[s]), \
             patch.object(cli, 'audit_cache', return_value={'issues': []}), \
             patch.object(cli, 'train_bemd_run', side_effect=AssertionError('No training')) as train, \
             patch.object(cli, 'evaluate_bemd_test', side_effect=AssertionError('No evaluation')) as evaluate:
            self.assertEqual(cli.main(), 0)
            train.assert_not_called()
            evaluate.assert_not_called()
        self.assertFalse(self.out.exists())


if __name__ == '__main__':
    unittest.main()

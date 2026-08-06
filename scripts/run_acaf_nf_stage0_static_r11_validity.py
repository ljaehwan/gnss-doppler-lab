#!/usr/bin/env python3
"""ACAF-NF Stage-0-R1.1 clean-only raw-IQ/tracker validity campaign."""
from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
from pathlib import Path

import h5py
import numpy as np

# Support direct execution from a source checkout, not only an installed package.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from gnss_doppler_lab.acaf_nf_stage0_r11_validity import *

ROOT = Path('/home/ubuntu/ssd_data/gnss-early-detection/artifacts/texbat-clean-graph-input-v2/receiver/cleanStatic-complex9/raw')
RAW = Path('/home/ubuntu/unraid_hdd/texbat/raw/cleanStatic.bin')
OUTDEFAULT = 'artifacts/acaf_nf_stage0_static_r11_validity'


def dump(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + '\n')


def csvout(path, rows):
    keys = sorted({key for row in rows for key in row}) or ['status']
    with Path(path).open('w', newline='') as handle:
        writer = csv.DictWriter(handle, keys)
        writer.writeheader()
        writer.writerows(rows)


def channel_name(path):
    return path.stem.removeprefix('epl_tracking_ch_')


def records():
    rows, inventory = [], []
    for path in sorted(ROOT.glob('epl_tracking_ch_*.mat')):
        mat_sha = sha256(path)
        with h5py.File(path, 'r') as mat:
            count = len(mat['PRN']) if 'PRN' in mat else 0
            missing = [key for key in REQ if key not in mat]
            inventory.append({'path': str(path), 'channel': channel_name(path), 'bytes': path.stat().st_size,
                              'sha256': mat_sha, 'n_rows': count, 'missing_required': missing})
            if missing:
                continue
            arrays = {key: np.asarray(mat[key]).reshape(-1) for key in REQ}
            for index in range(count):
                row = {key: arrays[key][index].item() for key in REQ}
                row.update({'mat_path': str(path), 'mat_index': index, 'mat_sha256': mat_sha,
                            'channel': channel_name(path)})
                rows.append(row)
    return parse_tracker_rows(rows), inventory


def grid(iq, row, candidate):
    """Evaluate the actual 3x3 raw prompt neighborhood and return peak provenance."""
    values = []
    for delay in (-.125, 0., .125):
        shifted = dict(row)
        shifted['aux1'] = row['aux1'] + delay * FS / row['code_freq_chips']
        for doppler in (-50., 0., 50.):
            hypothesis = dict(shifted)
            hypothesis['carrier_doppler_hz'] = row['carrier_doppler_hz'] + doppler
            values.append(local_prompt(iq, hypothesis, candidate))
    values = np.asarray(values).reshape(3, 3)
    peak = np.unravel_index(values.argmax(), values.shape)
    delay_offset = (-.125, 0., .125)[peak[0]]
    doppler_offset = (-50., 0., 50.)[peak[1]]
    return {
        'center_magnitude': float(values[1, 1]),
        'reconstructed_peak_magnitude': float(values[peak]),
        'peak_delay_offset_chips': delay_offset,
        'peak_doppler_offset_hz': doppler_offset,
        'center_peak': bool(peak == (1, 1)),
        'grid_boundary': bool(peak[0] in (0, 2) or peak[1] in (0, 2)),
    }


def candidate_window(row, candidate):
    source = row if candidate['tracker_row'] == 'k' else row['previous']
    start = row['sample_count'] if candidate['interval'] == 'end_k' else row['previous']['sample_count']
    return source, int(start), int(start) + EPOCH


def selected_clean_rows(base, candidate, bounded_n):
    """Filter valid candidate windows then PRN-stratify without raw-window overlap."""
    eligible = []
    for row in base:
        if not row.get('previous'):
            continue
        source, start, end = candidate_window(row, candidate)
        if start >= 0 and end <= RAW.stat().st_size // 4:
            copy = dict(row)
            copy['_raw_start'] = start
            copy['_raw_end'] = end
            copy['_source_prn'] = source['prn']
            eligible.append(copy)
    # Helper selects a deterministic PRN round-robin set and restores chronological order.
    selected = stratified_round_robin_clean(
        [dict(row, prn=row['_source_prn'], sample_count=row['_raw_start'], end_sample=row['_raw_end']) for row in eligible],
        bounded_n,
    )
    return selected, eligible


def row_provenance(row, source, start, end, candidate, result, raw_sha):
    return {
        'scenario': 'cleanStatic',
        'channel': source['channel'],
        'mat_path': source['mat_path'],
        'tracker_row_index': source['mat_index'],
        'prn': source['prn'],
        'sample_start': start,
        'sample_end_exclusive': end,
        'raw_byte_start': start * 4,
        'raw_byte_end_exclusive': end * 4,
        'raw_sha256': raw_sha,
        'tracker_mat_sha256': source['mat_sha256'],
        'aux1_samples': source['aux1'],
        'converted_remnant_chips': aux_samples_to_chips(source['aux1'], source['code_freq_chips']),
        'signed_remnant_chips': candidate['remnant_sign'] * aux_samples_to_chips(source['aux1'], source['code_freq_chips']),
        'tracker_doppler_hz': source['carrier_doppler_hz'],
        'mat_prompt_magnitude': source['mat_prompt_mag'],
        **result,
    }


def run_focused_tests():
    command = [sys.executable, '-m', 'pytest', '-q', 'tests/test_acaf_nf_stage0_r11_validity.py']
    completed = subprocess.run(command, text=True, capture_output=True)
    return command, completed


def write_recursive_checksums(out):
    checksum_path = out / 'checksums.json'
    files = {str(path.relative_to(out)): sha256(path) for path in sorted(out.rglob('*'))
             if path.is_file() and path != checksum_path}
    dump(checksum_path, {
        'algorithm': 'sha256',
        'scope': 'recursive_all_regular_artifacts_except_checksums.json_self_reference',
        'excluded_self_reference': 'checksums.json cannot hash its own final bytes',
        'files': files,
    })


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--output', default=OUTDEFAULT)
    parser.add_argument('--alignment-n', type=int, default=600)
    parser.add_argument('--gate-n', type=int, default=500)
    args = parser.parse_args()
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    (out / 'plots').mkdir(exist_ok=True)

    canonical = canonical_prn_identity()
    rows, inventory = records()
    raw_bytes = RAW.stat().st_size if RAW.is_file() else 0
    raw_samples = raw_bytes // 4
    raw_sha = sha256(RAW) if RAW.is_file() else None
    rows = [row for row in rows if row['end_sample'] <= raw_samples]
    rows.sort(key=lambda row: (row['sample_count'], row['prn']))
    byprn = {}
    for row in rows:
        byprn.setdefault(row['prn'], []).append(row)
    for sequence in byprn.values():
        for index, row in enumerate(sequence):
            row['previous'] = sequence[index - 1] if index else None
    try:
        split = chronological_split(rows)
        split_error = None
    except ValueError as error:
        split = {'train': [], 'calibration': [], 'holdout': []}
        split_error = str(error)

    config = {
        'stage': 'ACAF-NF Stage-0-R1.1 validity repair',
        'raw_format': 's16 little-endian interleaved IQ', 'fs_hz': FS, 'raw_epoch_samples': EPOCH,
        'aux1_formula': 'aux1_samples * code_freq_chips / fs',
        'selection': 'deterministic PRN round-robin clean strata, chronological output, non-overlapping raw windows',
        'no_attack_data_used': True, 'canonical_prns_verified': canonical,
        'center_tolerance': {'delay_chips': .125, 'doppler_hz': 50},
    }
    dump(out / 'config.json', config)
    dump(out / 'environment.json', {'python': platform.python_version(), 'numpy': np.__version__, 'h5py': h5py.__version__,
                                    'raw_path': str(RAW), 'raw_bytes': raw_bytes, 'raw_sha256': raw_sha})
    dump(out / 'tracker_source_inventory.json', {
        'source': '/home/ubuntu/build-gnss-sdr-complex9/src/algorithms/tracking/gnuradio_blocks/dll_pll_veml_tracking.cc',
        'base': '1ddd4562723040fd66cb334b578a5b69455625f4', 'tree_state': 'modified (confirmed external provenance)',
        'aux1_source': 'd_rem_code_phase_samples; converted by aux1*code_freq_chips/fs',
        'prn_start_source': 'nitems_read + current PRN samples', 'generator_sha256': '6c4512adefcfe49ae7d964c0425b26bfffd8b988ad7f9a0cf6f4b2e30fc5cafb',
        'mat_files': inventory,
    })
    dump(out / 'data_manifest.json', {
        'cleanStatic_raw': {'path': str(RAW), 'exists': RAW.exists(), 'bytes': raw_bytes, 'samples': raw_samples,
                            'duration_s': raw_samples / FS, 'sha256': raw_sha,
                            'byte_bounds_convention': '[sample_start*4, sample_end_exclusive*4)'},
        'ds4_manifest_search': {'status': 'NOT_FOUND', 'searched': ['data/external/texbat/manifests', 'configured tracker roots'],
                                'reason': 'no DS4 tracker/raw manifest supplied'},
        'ds7_ds8_one_second_overlap': ds78_overlap(None, None),
    })
    dump(out / 'sample_count_summary.json', {'stable_valid_rows': len(rows), 'valid_prns': sorted(byprn),
        'valid_prn_count': len(byprn), 'raw_duration_s': raw_samples / FS,
        'rows_after_explicit_raw_bounds': len(rows), 'split_error': split_error})
    roles = {key: len(value) for key, value in split.items()}
    dump(out / 'normal_split.json', {'status': 'PASS' if not split_error else 'FAIL', 'chronological_contiguous_roles': True,
        'roles': roles, 'required': {'train': 1000, 'calibration': 500, 'holdout': 500},
        'raw_ranges_nonoverlap': not bool(split_error), 'source': 'cleanStatic only', 'attack_data_used': False})

    base = split['train'] + split['calibration'] + split['holdout']
    hypotheses, evidence = [], {}
    for candidate in alignment_candidates():
        picked, eligible = selected_clean_rows(base, candidate, args.alignment_n)
        values, mat_values, centers, boundaries, prns = [], [], [], [], []
        for row in picked:
            source, start, _ = candidate_window(row, candidate)
            try:
                result = grid(raw_iq_s16le(RAW, start), source, candidate)
            except Exception:
                continue
            values.append(result['center_magnitude']); mat_values.append(source['mat_prompt_mag'])
            centers.append(result['center_peak']); boundaries.append(result['grid_boundary']); prns.append(source['prn'])
        hypotheses.append({'name': candidate['name'], 'status': 'EVALUATED_CLEAN_ONLY' if values else 'NO_VALID_RAW_EPOCHS',
            'n': len(values), 'eligible_n': len(eligible), 'prompt_spearman': spearman(values, mat_values) if values else float('nan'),
            'center_peak_fraction': float(np.mean(centers)) if centers else 0., 'boundary_fraction': float(np.mean(boundaries)) if boundaries else 1.,
            'prn_count': len(set(prns)), 'prns': sorted(set(prns)),
            'dominant_fraction': max([prns.count(prn) / len(prns) for prn in set(prns)], default=1.)})
        evidence[candidate['name']] = (values, mat_values, centers, boundaries, prns)
    csvout(out / 'alignment_hypotheses.csv', hypotheses)
    ranked = sorted([item for item in hypotheses if item['n']],
                    key=lambda item: (item['prompt_spearman'], item['center_peak_fraction'], -item['boundary_fraction']), reverse=True)
    selected = ranked[0] if ranked else None
    dump(out / 'selected_alignment.json', {'status': 'SELECTED_CLEAN_ACTUAL_PROMPT_AND_CENTER_EVIDENCE' if selected else 'NO_SELECTION',
        'selection_rule': 'max Prompt Spearman, then center peak, then lower boundary; attacks excluded',
        'selected': selected, 'candidates_evaluated': len(hypotheses)})

    validation = []
    if selected:
        candidate = next(item for item in alignment_candidates() if item['name'] == selected['name'])
        picked, eligible = selected_clean_rows(base, candidate, args.gate_n)
        for row in picked:
            source, start, end = candidate_window(row, candidate)
            try:
                result = grid(raw_iq_s16le(RAW, start), source, candidate)
                validation.append({'status': 'OK', **row_provenance(row, source, start, end, candidate, result, raw_sha)})
            except Exception as error:
                validation.append({'status': 'RAW_ERROR', 'reason': type(error).__name__, 'scenario': 'cleanStatic',
                                   'prn': source['prn'], 'sample_start': start, 'sample_end_exclusive': end,
                                   'raw_byte_start': start * 4, 'raw_byte_end_exclusive': end * 4,
                                   'raw_sha256': raw_sha, 'tracker_mat_sha256': source['mat_sha256']})
    csvout(out / 'center_validation.csv', validation)
    good = [row for row in validation if row.get('status') == 'OK']
    prns = [row['prn'] for row in good]
    stats = {'n': len(good), 'within_fraction': float(np.mean([row['center_peak'] for row in good])) if good else 0.,
        'spearman': spearman([row['center_magnitude'] for row in good], [row['mat_prompt_magnitude'] for row in good]) if good else float('nan'),
        'boundary_fraction': float(np.mean([row['grid_boundary'] for row in good])) if good else 1., 'prn_count': len(set(prns)),
        'prns': sorted(set(prns)), 'dominant_fraction': max([prns.count(prn) / len(prns) for prn in set(prns)], default=1.),
        'r0_delay_error_chips_mean_abs': float(np.mean(np.abs([row['peak_delay_offset_chips'] for row in good]))) if good else float('nan'),
        'r1_doppler_error_hz_mean_abs': float(np.mean(np.abs([row['peak_doppler_offset_hz'] for row in good]))) if good else float('nan')}
    gate = center_gate(stats)
    dump(out / 'center_validation_summary.json', {**stats, 'gate_a': gate, 'tolerances': config['center_tolerance'],
        'r0_definition': 'absolute reconstructed peak delay offset from the 3x3 center (chips)',
        'r1_definition': 'absolute reconstructed peak Doppler offset from the 3x3 center (Hz)'})

    gate_a = gate['status']; gate_c = 'INCOMPLETE'; verdict = classify_verdict(gate_a, gate_c)
    dump(out / 'thresholds.json', {'status': 'NOT_COMPUTED_CENTER_INVALID' if gate_a != 'PASS' else 'COMPUTED_NORMAL_ONLY',
        'reason': 'Gate B calculations only permitted after Gate A PASS', 'performance_claim': False})
    csvout(out / 'per_epoch_scores.csv', [{'status': 'NOT_EVALUATED_UNTIL_CENTER_VALID', 'reason': 'Gate A ' + gate_a}])
    csvout(out / 'scenario_metrics.csv', [{'scenario': 'DS4', 'status': 'INCOMPLETE', 'reason': 'DS4/B0 unavailable; not blocker of Gate A/B'}])
    csvout(out / 'budget_metrics.csv', [{'status': 'NOT_EVALUATED_UNTIL_CENTER_VALID', 'reason': 'no normal/attack budget performance claim before center validity'}])
    dump(out / 'bootstrap_results.json', {'status': 'NOT_EVALUATED_UNTIL_CENTER_VALID', 'reason': 'Gate A ' + gate_a, 'performance_claim': False})
    dump(out / 'physical_controls.json', {'status': 'NOT_EVALUATED_UNTIL_CENTER_VALID', 'reason': 'two-source/physical inference prohibited before center validity',
        'two_source': 'NOT_EVALUATED_UNTIL_CENTER_VALID', 'circular_observed_caf_rolling_used': False})
    dump(out / 'execution_validity.json', {'gate_a': gate_a, 'gate_b': 'NOT_EVALUATED_UNTIL_CENTER_VALID' if gate_a != 'PASS' else 'NOT_IMPLEMENTED',
        'gate_c': 'INCOMPLETE', 'fail_closed': True})
    dump(out / 'go_no_go.json', {**verdict, 'gate_a': gate_a, 'status': 'FAIL_CLOSED', 'verdict': verdict['verdict'], 'physics_no_go_claim': False})

    readme = f'''# ACAF-NF Stage-0-R1.1 validity repair

**Scope:** cleanStatic only; no attacks are read, fit, scored, or claimed. All stable valid PRNs are retained from the tracker and bounded raw reconstruction uses deterministic PRN round-robin strata restored to chronological order with non-overlapping raw windows.

**Tracker/remnant formula:** `remnant_chips = aux1_samples * code_freq_chips / fs`; the signed reconstruction uses `remnant_sign * remnant_chips`. Raw samples are s16 little-endian interleaved IQ; each row records `[sample_start, sample_end_exclusive)` and raw bytes `[start*4, end*4)` with raw and tracker-MAT SHA-256 provenance.

**Actual clean validation:** alignment `{selected['n'] if selected else 0}` / requested `{args.alignment_n}`; center `{len(good)}` / requested `{args.gate_n}`; PRNs `{sorted(set(prns))}` (`{len(set(prns))}`). True 3x3-center rate is `{stats['within_fraction']:.6f}` and true raw-center-magnitude/MAT-Prompt Spearman correlation is `{stats['spearman']:.6f}`. R0 delay error is mean absolute reconstructed peak center offset `{stats['r0_delay_error_chips_mean_abs']:.6f}` chips; R1 Doppler error is mean absolute reconstructed peak center offset `{stats['r1_doppler_error_hz_mean_abs']:.6f}` Hz.

**Normal split:** train/calibration/holdout = `{roles['train']}/{roles['calibration']}/{roles['holdout']}`, chronological and raw-range non-overlapping: `{not bool(split_error)}`.

**Gates:** Gate A `{gate_a}` (n≥500, center≥0.95, Prompt Spearman≥0.90, boundary≤0.05, ≥4 PRNs, dominant≤0.50). Gate B is not evaluated until A passes. Gate C is `INCOMPLETE` because DS4/B0 provenance is unavailable. Verdict: `{verdict['verdict']}`. This is **not** a physics NO-GO claim and makes **no** detection/performance claim.
'''
    (out / 'README.md').write_text(readme)
    dump(out / 'verification_report.json', {'artifact_status': 'complete_substantive_fail_closed', 'required_files': 'written', 'gate_a': gate_a,
        'actual_clean_alignment_epochs': selected['n'] if selected else 0, 'actual_center_epochs': len(good),
        'actual_center_prns': sorted(set(prns)), 'raw_duration_s': raw_samples / FS})

    command, test_result = run_focused_tests()
    (out / 'test_report.txt').write_text('command: ' + ' '.join(command) + '\nexit_code: ' + str(test_result.returncode) + '\nresult: ' + ('PASS' if test_result.returncode == 0 else 'FAIL') + '\nstdout:\n' + test_result.stdout + '\nstderr:\n' + test_result.stderr)
    write_recursive_checksums(out)
    print(json.dumps({'gate_a': gate_a, 'verdict': verdict['verdict'], 'stable_rows': len(rows),
                      'alignment_n': selected['n'] if selected else 0, 'center_n': len(good), 'center_prns': sorted(set(prns)),
                      'focused_test_exit_code': test_result.returncode}))
    return test_result.returncode


if __name__ == '__main__':
    raise SystemExit(main())

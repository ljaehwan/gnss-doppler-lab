"""Actual cleanStatic holdout control generation and Full rescoring."""
from __future__ import annotations

import numpy as np

from .gcspo_clean import AggregatedClean, EPOCH_S, EPOCH_SAMPLES, residual_table
from .gcspo_control_grid import generate_control_grid
from .gcspo_controls import ControlContext
from .gcspo_core import content_seed
from .gcspo_full import GeometryCache, role_full_terms_from_z, score_full_terms


def _slice(data, mask):
    return AggregatedClean(epoch=data.epoch[mask], prn=data.prn[mask], channel=data.channel[mask],
                           segment=data.segment[mask], q=data.q[mask], sample_min=data.sample_min[mask],
                           sample_max=data.sample_max[mask], epsilons=data.epsilons,
                           source_files=data.source_files)


def _complete_cube(data, start_s, end_s):
    first, final = round(start_s / EPOCH_S), round(end_s / EPOCH_S)
    epochs = np.arange(first, final, dtype=np.int64)
    mask = (data.epoch >= first) & (data.epoch < final)
    candidates = sorted(set(map(int, data.prn[mask])))
    continuous = [prn for prn in candidates if np.array_equal(np.sort(data.epoch[mask & (data.prn == prn)]), epochs)]
    lookup = {(int(epoch), int(prn)): index for index, (epoch, prn) in enumerate(zip(data.epoch, data.prn))
              if first <= epoch < final and prn in continuous}
    prns, channels, segments = [], [], []
    for prn in continuous:
        indices = [lookup[(int(epoch), prn)] for epoch in epochs]
        identities = {(int(data.channel[index]), int(data.segment[index])) for index in indices}
        if len(identities) == 1:
            channel, segment = identities.pop(); prns.append(prn); channels.append(channel); segments.append(segment)
    if len(prns) < 4: raise ValueError("control block has fewer than four continuous stable identities")
    q = np.stack([[data.q[lookup[(int(epoch), prn)]] for epoch in epochs] for prn in prns])
    selected = np.asarray([lookup[(int(epoch), prn)] for epoch in epochs for prn in prns], np.int64)
    block = _slice(data, selected)
    return block, np.asarray(prns, np.int64), epochs, q, np.asarray(channels), np.asarray(segments)


def _residual_cube(block, model, whitener, start_s, end_s, prns, epochs):
    residual_epochs, residual_prns, residuals, _ = residual_table(block, model, whitener, start_s, end_s)
    lookup = {(int(epoch), int(prn)): value for epoch, prn, value in zip(residual_epochs, residual_prns, residuals)}
    cube = np.zeros((len(prns), len(epochs), 10), dtype=np.float64)
    for pi, prn in enumerate(prns):
        for ti, epoch in enumerate(epochs):
            if (int(epoch), int(prn)) in lookup: cube[pi, ti] = lookup[(int(epoch), int(prn))]
    return cube


def run_clean_control_evidence(clean, geometry, *, smoothness, threshold):
    data, model, whitener, gamma = clean["data"], clean["model"], clean["whitener"], clean["gamma"]
    validated = {"code_error_chips", "pll_phase_error_cycles", "carrier_doppler_hz", "code_frequency_offset_chips_s"}
    train_epochs, train_prns, train_residuals, _ = residual_table(data, model, whitener, 30., 140.)
    robust = {}
    for prn in sorted(set(map(int, train_prns))):
        values = train_residuals[train_prns == prn]
        center = np.median(values, axis=0)
        robust[prn] = 1.4826 * np.median(np.abs(values - center), axis=0)
    source_blocks = []
    for source_index in range(11):
        start = 30. + 10. * source_index
        block, prns, epochs, _, _, _ = _complete_cube(data, start, start + 10.)
        source_blocks.append((prns, _residual_cube(block, model, whitener, start, start + 10., prns, epochs)))

    contexts = []
    for block_id in range(12):
        block_start = 350. + 10. * block_id
        block, prns, epochs, q, channels, segments = _complete_cube(data, block_start, block_start + 10.)
        epsilon = np.asarray([data.epsilons[int(prn)] for prn in prns], float)
        norm = np.linalg.norm(q[:, :, :6], axis=2)
        if np.any(norm >= 1.) or np.any(norm <= 0.): raise ValueError("cannot reconstruct raw complex control inputs")
        raw = q[:, :, :6] * epsilon[:, None, None] / (1. - norm[:, :, None])
        baseline_residual = _residual_cube(block, model, whitener, block_start, block_start + 10., prns, epochs)
        robust_scale = np.stack([robust[int(prn)] for prn in prns])

        def factory(control_id, level, *, _prns=prns, _epochs=epochs, _raw=raw, _q=q,
                    _residual=baseline_residual, _channels=channels, _segments=segments,
                    _epsilon=epsilon, _robust=robust_scale, _block_id=block_id, _start=block_start):
            seed = content_seed(control_id, "cleanStatic", "holdout", _block_id, level, "BLOCK")
            source_index = seed % len(source_blocks); source_prns, source_cube = source_blocks[source_index]
            source = source_cube
            return ControlContext(prns=_prns, times_s=_epochs * EPOCH_S, raw_complex=_raw,
                                  other_q=_q[:, :, 6:], epsilon_by_prn=_epsilon,
                                  residual=_residual, source_residual=source,
                                  cn0=np.zeros((_prns.size, _epochs.size)), robust_scale=_robust,
                                  channels=_channels, segments=_segments, source_block_index=source_index, source_prns=source_prns)
        contexts.append((block_id, block_start, factory))

    score_cache = {}
    def scorer(result):
        key = (tuple(map(int, result.prns)), result.q.tobytes() if result.stage == "raw_q" else result.residual.tobytes())
        if key in score_cache: return score_cache[key]
        epochs = np.rint(result.times_s / EPOCH_S).astype(np.int64)
        if len(result.prns) < 4: return []
        if result.stage == "raw_q":
            flat_epoch = np.repeat(epochs, len(result.prns)); flat_prn = np.tile(result.prns, len(epochs))
            flat_q = np.transpose(result.q, (1, 0, 2)).reshape(-1, 10)
            flat_channel = np.tile(result.channels, len(epochs)); flat_segment = np.tile(result.segments, len(epochs))
            transformed = AggregatedClean(flat_epoch, flat_prn, flat_channel, flat_segment, flat_q,
                                          flat_epoch * EPOCH_SAMPLES, (flat_epoch + 1) * EPOCH_SAMPLES - 1,
                                          data.epsilons, ("clean-control",))
            residual_epochs, residual_prns, _, z = residual_table(transformed, model, whitener,
                                                                   result.times_s[0], result.times_s[-1] + EPOCH_S)
        else:
            residual_epochs = np.repeat(epochs[model.lags:], len(result.prns))
            residual_prns = np.tile(result.prns, len(epochs) - model.lags)
            raw_residual = np.transpose(result.residual[:, model.lags:], (1, 0, 2)).reshape(-1, 10)
            z = whitener.transform(raw_residual)
        cache = GeometryCache(geometry["ephemerides"], geometry["receiver_ecef"], validated)
        terms = role_full_terms_from_z(residual_epochs, residual_prns, z, model, gamma, cache,
                                       result.times_s[0], result.times_s[-1] + EPOCH_S)
        values = [row["score"] for row in score_full_terms(terms, smoothness)]
        score_cache[key] = values
        return values

    report = generate_control_grid(contexts, scenario="cleanStatic", phase="holdout",
                                   var_coefficients=model.coefficients, threshold=threshold, scorer=scorer)
    report["source_blocks"] = [{"index": index, "start_s": 30. + 10. * index,
                                "end_s": 40. + 10. * index} for index in range(11)]
    report["holdout_blocks"] = [{"index": index, "start_s": 350. + 10. * index,
                                 "end_s": 360. + 10. * index} for index in range(12)]
    return report

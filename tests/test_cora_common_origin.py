import numpy as np
import pytest

from gnss_doppler_lab.cora_common_origin import (
    TANGENT_PROJECTOR,
    condition_tokens,
    fit_shared_conditioner,
    phase_surrogate,
    score_token_block,
    shared_emitter_likelihood,
    temporal_desynchronize,
)
from gnss_doppler_lab.cora_cross_cumulant import (
    brute_force_cross_cumulant,
    cross_cumulant_matrix,
    fourth_cross_cumulant_kstat,
    permute_matrix,
)


def _complex_normal(rng, shape):
    return (rng.normal(size=shape) + 1j * rng.normal(size=shape)) / np.sqrt(2.0)


def test_tangent_projector_removes_constant_delay_and_doppler_directions():
    delay, doppler = np.meshgrid((-0.25, 0.0, 0.25), (-25.0, 0.0, 25.0), indexing="ij")
    tangent = np.column_stack((np.ones(9), delay.ravel(), doppler.ravel()))
    assert np.allclose(TANGENT_PROJECTOR, TANGENT_PROJECTOR.T)
    assert np.allclose(TANGENT_PROJECTOR @ TANGENT_PROJECTOR, TANGENT_PROJECTOR)
    assert np.linalg.norm(TANGENT_PROJECTOR @ tangent) < 1e-12


def test_unbiased_kstat_is_zero_for_independent_gaussian_in_expectation():
    rng = np.random.default_rng(1103)
    estimates = []
    for _ in range(600):
        x = _complex_normal(rng, 32)
        y = _complex_normal(rng, 32)
        estimates.append(fourth_cross_cumulant_kstat(x, y))
    assert abs(np.mean(estimates)) < 0.02


def test_gaussian_common_component_is_cancelled_by_all_pair_partitions():
    rng = np.random.default_rng(1104)
    estimates = []
    for _ in range(500):
        shared = _complex_normal(rng, 48)
        x = shared + 0.3 * _complex_normal(rng, 48)
        y = (0.4 + 0.7j) * shared + 0.3 * _complex_normal(rng, 48)
        estimates.append(fourth_cross_cumulant_kstat(x, y))
    assert abs(np.mean(estimates)) < 0.06


def test_shared_nongaussian_latent_separates_from_independent_matched_marginals():
    rng = np.random.default_rng(1105)
    shared_values = []
    independent_values = []
    for _ in range(300):
        shared = rng.laplace(size=64).astype(complex)
        shared_values.append(fourth_cross_cumulant_kstat(
            shared + 0.2 * _complex_normal(rng, 64),
            shared + 0.2 * _complex_normal(rng, 64),
        ))
        left = rng.laplace(size=64).astype(complex)
        right = rng.laplace(size=64).astype(complex)
        independent_values.append(fourth_cross_cumulant_kstat(
            left + 0.2 * _complex_normal(rng, 64),
            right + 0.2 * _complex_normal(rng, 64),
        ))
    assert np.mean(shared_values) > 1.0
    assert abs(np.mean(independent_values)) < 0.15


def test_kstat_converges_to_independent_plug_in_relation():
    rng = np.random.default_rng(1106)
    shared = rng.laplace(size=200_000)
    x = shared + 0.5 * rng.normal(size=len(shared))
    y = 0.8 * shared + 0.5 * rng.normal(size=len(shared))
    assert fourth_cross_cumulant_kstat(x, y) == pytest.approx(
        brute_force_cross_cumulant(x, y), rel=4e-4
    )


def test_matrix_symmetry_permutation_and_variable_prn_count():
    rng = np.random.default_rng(1107)
    tokens = _complex_normal(rng, (48, 6, 3))
    latent = rng.laplace(size=(48, 1, 1))
    tokens += latent * np.asarray([1, .8, .6, .4, .2, .1])[None, :, None]
    matrix = cross_cumulant_matrix(tokens)
    order = np.asarray([4, 0, 5, 2, 1, 3])
    reordered = cross_cumulant_matrix(tokens[:, order])
    assert np.allclose(matrix, matrix.T)
    assert np.allclose(np.diag(matrix), 0.0)
    assert np.allclose(reordered, permute_matrix(matrix, order))
    assert cross_cumulant_matrix(tokens[:, :4]).shape == (4, 4)


def test_likelihood_and_score_are_prn_permutation_invariant():
    rng = np.random.default_rng(1108)
    tokens = _complex_normal(rng, (64, 5, 4))
    tokens += rng.laplace(size=(64, 1, 1))
    first, matrix = score_token_block(tokens, null_variance=0.05)
    order = np.asarray([2, 4, 0, 3, 1])
    second, reordered = score_token_block(tokens[:, order], null_variance=0.05)
    assert second.score == pytest.approx(first.score)
    assert second.rank1_strength == pytest.approx(first.rank1_strength)
    assert np.allclose(reordered, permute_matrix(matrix, order))


def test_clean_only_shared_conditioner_removes_context_without_prn_identity():
    rng = np.random.default_rng(1109)
    epochs, prns, projections = 80, 5, 3
    context = rng.normal(size=(epochs, prns, 2))
    beta = np.asarray([[1 + .3j, -.4j, .2], [.7, -.2j, .1], [-.3j, .4, .8]])
    design = np.concatenate((np.ones((epochs, prns, 1)), context), axis=2)
    tokens = np.einsum("epc,ck->epk", design, beta) + .05 * _complex_normal(rng, (epochs, prns, projections))
    model = fit_shared_conditioner(tokens, context)
    innovations = condition_tokens(tokens, context, model)
    flat_context = context.reshape(-1, 2)
    flat_innovations = innovations.reshape(-1, projections)
    for c in range(2):
        for k in range(projections):
            assert abs(np.corrcoef(flat_context[:, c], flat_innovations[:, k].real)[0, 1]) < .05


def test_temporal_and_phase_destruction_preserve_shape_and_basic_marginals():
    rng = np.random.default_rng(1110)
    tokens = _complex_normal(rng, (128, 5, 3))
    shifted = temporal_desynchronize(tokens, np.arange(5))
    assert shifted.shape == tokens.shape
    for prn in range(5):
        assert np.allclose(np.sort(np.abs(shifted[:, prn]), axis=0), np.sort(np.abs(tokens[:, prn]), axis=0))
    surrogate = phase_surrogate(tokens, seed=44)
    assert surrogate.shape == tokens.shape
    assert np.allclose(np.sort(np.abs(surrogate), axis=0), np.sort(np.abs(tokens), axis=0))


def test_likelihood_rejects_too_few_prns():
    with pytest.raises(ValueError, match=">=4"):
        shared_emitter_likelihood(np.zeros((3, 3)), null_variance=1.0)

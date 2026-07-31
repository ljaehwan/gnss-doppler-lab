import io
import numpy as np
import pytest
import torch

from gnss_doppler_lab.gcmr_model import (
    CleanReferenceScoreCalibrator,
    GcmrNet,
    RobustGcmrScaler,
    collate_gcmr_events,
    gcmr_loss,
    masked_reconstruction_mse,
    event_reconstruction_errors,
)
from gnss_doppler_lab.gcmr_relations import GcmrPairRelationEvent


def event(pairs=3, offset=0.0):
    observations = np.arange(pairs * 10, dtype=np.float32).reshape(pairs, 10) / 10 + offset
    mask = np.ones((pairs, 10), dtype=bool)
    conditions = np.arange(pairs * 8, dtype=np.float32).reshape(pairs, 8) / 7 + offset
    return GcmrPairRelationEvent(0.0, 1.0, np.stack((np.arange(1, pairs + 1), np.arange(2, pairs + 2)), 1), observations, mask, conditions)


def fitted_model(batch):
    torch.manual_seed(7)
    model = GcmrNet(pair_hidden=32, event_hidden=32, latent_dim=16)
    model.fit_scaler(batch["observations"], batch["observation_mask"], batch["conditions"], batch["pair_mask"])
    model.eval()
    return model


def test_pair_permutation_equivariance_and_latent_invariance():
    batch = collate_gcmr_events([event(4)])
    model = fitted_model(batch)
    permutation = torch.tensor([2, 0, 3, 1])
    with torch.no_grad():
        recon, latent = model(**batch)
        permuted = {key: value[:, permutation] if value.ndim >= 2 and value.shape[1] == 4 else value for key, value in batch.items()}
        recon_p, latent_p = model(**permuted)
    assert torch.equal(latent, latent_p)
    assert torch.allclose(recon[:, permutation], recon_p, atol=1e-7, rtol=0)


def test_padding_leaves_real_output_latent_and_loss_unchanged():
    one = collate_gcmr_events([event(2)])
    padded = collate_gcmr_events([event(2), event(5, 1.0)])
    model = fitted_model(padded)
    with torch.no_grad():
        r1, z1 = model(**one)
        rp, zp = model(**{key: value[:1] for key, value in padded.items()})
    assert torch.equal(z1, zp)
    assert torch.allclose(r1, rp[:, :2], atol=1e-7, rtol=0)
    assert torch.equal(gcmr_loss(r1, one["observations"], one["observation_mask"], one["pair_mask"], observation_scale=model.scaler.observation_scale),
                       gcmr_loss(rp, padded["observations"][:1], padded["observation_mask"][:1], padded["pair_mask"][:1], observation_scale=model.scaler.observation_scale))


def test_empty_and_all_masked_events_are_rejected():
    with pytest.raises(ValueError, match="pair"):
        collate_gcmr_events([event(0)])
    bad = event(2)
    object.__setattr__(bad, "observation_mask", np.zeros((2, 10), dtype=bool))
    with pytest.raises(ValueError, match="observed"):
        collate_gcmr_events([bad])


def test_robust_scaler_ignores_masked_and_padded_extremes():
    obs = torch.ones(2, 2, 10)
    obs[0, 1] = 1e30
    obs_mask = torch.ones_like(obs, dtype=torch.bool)
    obs_mask[0, 1] = False
    conditions = torch.ones(2, 2, 8)
    conditions[1, 1] = -1e30
    pair_mask = torch.tensor([[True, True], [True, False]])
    a = RobustGcmrScaler().fit(obs, obs_mask, conditions, pair_mask)
    obs[0, 1] = -1e30
    conditions[1, 1] = 1e30
    b = RobustGcmrScaler().fit(obs, obs_mask, conditions, pair_mask)
    assert torch.equal(a.observation_center, b.observation_center)
    assert torch.equal(a.observation_scale, b.observation_scale)
    assert torch.equal(a.condition_center, b.condition_center)
    assert torch.equal(a.condition_scale, b.condition_scale)


def test_decoder_is_geometry_conditioned_not_target_conditioned():
    batch = collate_gcmr_events([event(3)])
    model = fitted_model(batch)
    with torch.no_grad():
        _, latent = model(**batch)
        c1 = model.scaler.transform_conditions(batch["conditions"])
        c2 = c1.clone(); c2[:, 0] += 3
        r1 = model.decode(latent, c1)
        r2 = model.decode(latent, c2)
    assert not torch.allclose(r1[:, 0], r2[:, 0])


def test_masked_reconstruction_loss_ignores_invalid_extreme_target():
    target = torch.zeros(1, 2, 10)
    recon = torch.ones_like(target)
    mask = torch.ones_like(target, dtype=torch.bool)
    mask[0, 1, 3] = False
    pair_mask = torch.tensor([[True, True]])
    before = masked_reconstruction_mse(recon, target, mask, pair_mask, observation_scale=torch.ones(10))
    target[0, 1, 3] = 1e30
    after = masked_reconstruction_mse(recon, target, mask, pair_mask, observation_scale=torch.ones(10))
    assert torch.equal(before, after)


def test_clean_reference_calibrator_handles_singular_covariance_and_outlier():
    clean_latent = np.stack([np.linspace(0, 1, 8) * x for x in np.linspace(.9, 1.1, 20)])
    clean_errors = np.linspace(.09, .11, 20)
    calibrator = CleanReferenceScoreCalibrator(shrinkage=0.1).fit(clean_errors, clean_latent)
    clean_score = calibrator.score(clean_errors, clean_latent)
    outlier_score = calibrator.score(np.array([2.0]), np.full((1, 8), 10.0))[0]
    assert np.isfinite(clean_score).all() and np.isfinite(outlier_score)
    assert outlier_score > clean_score.max() + 5


def test_state_dict_save_load_is_deterministic():
    batch = collate_gcmr_events([event(3)])
    model = fitted_model(batch)
    with torch.no_grad(): expected = model(**batch)
    buffer = io.BytesIO(); torch.save(model.state_dict(), buffer); buffer.seek(0)
    restored = GcmrNet(pair_hidden=32, event_hidden=32, latent_dim=16)
    restored.load_state_dict(torch.load(buffer, weights_only=True)); restored.eval()
    with torch.no_grad(): actual = restored(**batch)
    assert torch.equal(expected[0], actual[0]); assert torch.equal(expected[1], actual[1])


def test_standardized_residual_loss_balances_channel_units_and_masking():
    scaler = RobustGcmrScaler()
    scaler.observation_scale.copy_(torch.tensor([2., 5.] + [1.] * 8)); scaler.fitted.fill_(True)
    target = torch.zeros(1, 1, 10); reconstruction = target.clone()
    reconstruction[0, 0, 0] = 2.; reconstruction[0, 0, 1] = 5.
    mask = torch.zeros_like(target, dtype=torch.bool); mask[0, 0, :2] = True
    pairs = torch.ones(1, 1, dtype=torch.bool)
    both = masked_reconstruction_mse(reconstruction, target, mask, pairs, observation_scale=scaler.observation_scale)
    first = mask.clone(); first[..., 1] = False
    one = masked_reconstruction_mse(reconstruction, target, first, pairs, observation_scale=scaler.observation_scale)
    assert both.item() == pytest.approx(1.) and one.item() == pytest.approx(1.)
    assert event_reconstruction_errors(reconstruction, target, mask, pairs, observation_scale=scaler.observation_scale).item() == pytest.approx(1.)

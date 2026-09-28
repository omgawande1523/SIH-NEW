import numpy as np
import xarray as xr

from sih import config as C
from sih.products import _headline, alert_probabilities, pick_members, weighted_quantile


def _ens(n=50, seed=0):
    rng = np.random.default_rng(seed)
    lat, lon = np.arange(10, 22.5, 1.5), np.arange(80, 92.5, 1.5)
    tp6 = rng.gamma(1.0, 10.0, (n, 1, len(lat), len(lon))).astype("float32")
    return xr.Dataset({"tp6": (("member", "lead", "lat", "lon"), tp6)},
                      coords={"member": np.arange(n), "lead": [96], "lat": lat, "lon": lon})


def test_member_weights_sum_to_one_and_keep_the_wettest_members():
    ens = _ens()
    members, w = pick_members(ens, 96, [10, 80, 22, 92], 24)
    assert len(members) == len(set(members)) == 24
    assert abs(sum(w) - 1) < 1e-9
    wettest = int(np.argmax(ens.tp6.isel(lead=0).max(("lat", "lon")).values))
    assert wettest in members


def test_small_ensemble_is_used_whole():
    members, w = pick_members(_ens(n=20), 96, [10, 80, 22, 92], 24)
    assert sorted(members) == list(range(20)) and np.allclose(w, 1 / 20)


def test_importance_weights_give_an_unbiased_exceedance_probability():
    ens = _ens(seed=3)
    box = [10, 80, 22, 92]
    peak = ens.tp6.isel(lead=0).max(("lat", "lon")).values
    truth = (peak >= 45).mean()
    members, w = pick_members(ens, 96, box, 24)
    est = sum(wi for m, wi in zip(members, w) if peak[m] >= 45)
    assert abs(est - truth) < 0.1


def test_weighted_quantile_matches_numpy_for_equal_weights():
    x = np.random.default_rng(1).normal(size=(101, 3))
    w = np.full(101, 1 / 101)
    assert np.allclose(weighted_quantile(x, w, 0.5), np.median(x, axis=0))


def test_alert_probability_uses_the_weights():
    lat, lon = np.arange(15, 17, 0.25), np.arange(84, 86, 0.25)
    wet = np.full((len(lat), len(lon)), 60.0, "float32")
    dry = np.zeros_like(wet)
    _, _, probs, _, _ = alert_probabilities(lat, lon, [wet, dry], [0.2, 0.8], radius_km=C.ALERT_RADIUS_KM)
    assert np.allclose(probs[-1], 0.2)  # P(severe) = weight of the wet sample


def _core(lead, severe, se=0.05, level="none"):
    return {"lead": lead, "level": level, "prob": {"severe": severe}, "prob_se": {"severe": se}}


def test_headline_takes_the_highest_p_severe():
    assert _headline([_core(60, 0.10), _core(96, 0.26), _core(132, 0.12)])["lead"] == 96


def test_headline_ties_go_to_the_earliest_lead():
    assert _headline([_core(60, 0.24), _core(96, 0.26), _core(132, 0.25)])["lead"] == 60


def test_headline_never_promotes_the_level():
    h = _headline([_core(60, 0.10, level="low"), _core(96, 0.30, level="none")])
    assert h["lead"] == 96 and h["level"] == "none"

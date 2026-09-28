import numpy as np

from sih.stage1.efi import P, efi, sot

rng = np.random.default_rng(0)
CLIM = rng.normal(300.0, 2.0, (3000, 4, 4))
Q = np.quantile(CLIM, P, axis=0)


# The integral is evaluated on 101 climate quantiles with the singular end points left out,
# so its extremes are +-0.91 rather than the analytic +-1.


def test_efi_saturates_when_every_member_beats_the_climate_maximum():
    ens = np.full((50, 4, 4), CLIM.max() + 5.0)
    assert (efi(ens, Q, "t2m") > 0.9).all()


def test_efi_is_symmetric_below_the_climate_minimum():
    hi = efi(np.full((50, 4, 4), CLIM.max() + 5.0), Q, "t2m")
    lo = efi(np.full((50, 4, 4), CLIM.min() - 5.0), Q, "t2m")
    assert np.allclose(lo, -hi)


def test_efi_is_near_zero_for_a_climatological_ensemble():
    ens = rng.normal(300.0, 2.0, (200, 4, 4))
    assert np.abs(efi(ens, Q, "t2m")).max() < 0.15


def test_sot_is_positive_only_when_the_tail_goes_beyond_q99():
    assert (sot(np.full((50, 4, 4), Q[99] + 3.0), Q) > 0).all()
    assert (sot(rng.normal(300.0, 2.0, (50, 4, 4)), Q) < 0).all()

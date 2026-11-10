import numpy as np
import pytest

from waterbalance import balance as bal
from waterbalance import et0, soil, units


def test_unit_conversions():
    assert units.k_to_c(273.15) == pytest.approx(0.0)
    assert units.pa_to_kpa(101_325) == pytest.approx(101.325)
    assert units.j_to_mj(15e6) == pytest.approx(15.0)
    assert units.m_to_mm(0.0042) == pytest.approx(4.2)
    assert units.era5_evaporation_to_mm(-0.003) == pytest.approx(3.0)       # sign flipped
    assert units.wind_to_2m(1.0, z=10) == pytest.approx(0.748, abs=1e-3)     # FAO-56 factor
    assert units.olm_oc_to_om_pct(2.0) == pytest.approx(10 / 10 * 1.724)     # 10 g/kg OC
    assert units.theta_to_mm(0.3, 0.6) == pytest.approx(180.0)


def test_et0_fao56_example_18():
    # FAO-56 Example 18 (Brussels, 6 July): ET0 = 3.9 mm/day
    v = et0.et0_pm(tmax_c=21.5, tmin_c=12.3, ea_kpa=1.409, u2=2.078, rn_mj=13.28, p_kpa=100.1)
    assert float(v) == pytest.approx(3.9, abs=0.1)


def test_saxton_rawls_loam():
    # Saxton & Rawls (2006) Table: loam (40 % sand, 20 % clay, 2.5 % OM) WP 12, FC 28, SAT 46 %
    wp, fc, sat = soil.saxton_rawls(0.40, 0.20, 2.5)
    assert float(wp) == pytest.approx(0.12, abs=0.02)
    assert float(fc) == pytest.approx(0.28, abs=0.02)
    assert float(sat) == pytest.approx(0.46, abs=0.03)


def test_profile_integration():
    depths = [0, 10, 30, 60, 100, 200]
    const = np.full((6, 2, 2), 0.3)
    assert soil.profile_to_mm(const, depths, 0.6) == pytest.approx(np.full((2, 2), 180.0))
    lin = np.array([0.1, 0.2, 0.2, 0.2, 0.2, 0.2])
    assert float(soil.profile_to_mm(lin, depths, 0.1)) == pytest.approx(15.0)   # (0.1+0.2)/2*100
    with pytest.raises(ValueError):
        soil.profile_to_mm(const, depths, 2.5)


def test_hydrologic_group_and_cn():
    g = soil.hydrologic_group(np.array([0.9, 0.4, 0.3, 0.1]), np.array([0.05, 0.15, 0.3, 0.5]))
    assert g.tolist() == [1, 2, 3, 4]
    cn = soil.curve_number(g, {"A": 67, "B": 78, "C": 85, "D": 89})
    assert cn.tolist() == [67, 78, 85, 89]


def test_scs_runoff():
    assert float(bal.scs_runoff(50.0, 80.0)) == pytest.approx(13.8, abs=0.1)
    assert float(bal.scs_runoff(5.0, 80.0)) == 0.0


def _bucket(P, ET0, irrigation="none", p=0.5, s0=None):
    h = w = 3
    fc, wp, sat = np.full((h, w), 180.0), np.full((h, w), 80.0), np.full((h, w), 280.0)
    cn = np.full((h, w), 78.0)
    s0 = fc if s0 is None else np.full((h, w), s0)
    nt = P.shape[0]
    return bal.run_bucket(P, ET0, fc, wp, sat, cn, s0, kc=1.0, p=p, tau=0.5,
                          irrigation=irrigation, periods=[("all", 0, nt), ("last", nt - 14, nt)])


@pytest.mark.parametrize("irrigation", ["none", "auto"])
def test_bucket_closes(irrigation):
    rng = np.random.default_rng(1)
    P = (rng.gamma(0.4, 20.0, (120, 3, 3)) * (rng.random((120, 3, 3)) < 0.4)).astype(np.float32)
    ET0 = rng.uniform(2, 6, (120, 3, 3)).astype(np.float32)
    _, res = _bucket(P, ET0, irrigation)
    for r in res.values():
        assert np.abs(r["residual"]).max() < 1e-3          # P + I - R - ET - D = dS
    if irrigation == "none":
        assert res["all"]["I"].max() == 0


def test_drying_stops_at_wilting_point():
    P = np.zeros((200, 3, 3), np.float32)
    ET0 = np.full((200, 3, 3), 6.0, np.float32)
    s_end, res = _bucket(P, ET0)
    f = res["last"]["f_end"]
    assert (f >= 0).all() and (f < 0.05).all()
    assert (s_end >= 80.0 - 1e-4).all()
    assert (bal.classify(f, 0.5) == 1).all()                 # severe
    assert res["last"]["stress_days"].min() == 14


def test_heavy_rain_gives_surplus():
    P = np.zeros((20, 3, 3), np.float32)
    P[-3:] = 120.0
    ET0 = np.full((20, 3, 3), 3.0, np.float32)
    _, res = _bucket(P, ET0)
    f = res["last"]["f_end"]
    assert (f > 1.0).all()
    assert (bal.classify(f, 0.5) == 4).all()


def test_auto_irrigation_avoids_stress():
    P = np.zeros((60, 3, 3), np.float32)
    ET0 = np.full((60, 3, 3), 5.0, np.float32)
    _, res = _bucket(P, ET0, irrigation="auto")
    assert res["all"]["I"].min() > 0
    assert (res["last"]["f_min"] > 0.4).all()                # never far below 1 - p


def test_classify_thresholds():
    f = np.array([0.1, 0.3, 0.6, 0.95, 1.2, np.nan])
    assert bal.classify(f, p=0.5).tolist() == [1, 2, 3, 3, 4, -1]
    # rice p = 0.2 -> stress starts below 0.8
    assert bal.classify(np.array([0.7, 0.85]), p=0.2).tolist() == [2, 3]
    assert bal.classify(np.array([0.7]), p=0.5, valid=np.array([False])).tolist() == [-1]

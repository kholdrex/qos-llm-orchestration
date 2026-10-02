import pytest

from qos_orchestration.policy import TARGETS, NetState, Profile, Request, predict_latency, qos_aware, quality_cost


def profile(quality, decode, price_out=0.0):
    return Profile(
        q={"qa": quality},
        n_out={"qa": 11},
        ttft0=0.1,
        prefill=0.0,
        decode=decode,
        bytes_per_tok=100,
        price_out=price_out,
    )


PROFILES = {"local": profile(0.5, 0.01), "edge": profile(0.7, 0.05), "cloud": profile(0.9, 0.02, price_out=1e-6)}
REQUEST = Request(cls="qa", n_in=10, s_up=500)


def net(rtt=0.05, down=0):
    return NetState(
        rtt={"local": 0.0, "edge": 0.002, "cloud": rtt},
        b_up={m: 0 for m in TARGETS},
        b_down={"local": 0, "edge": 0, "cloud": down},
    )


def test_predicted_latency_components():
    assert predict_latency(PROFILES["local"], REQUEST, net(), "local") == pytest.approx(0.1 + 10 * 0.01)
    assert predict_latency(PROFILES["cloud"], REQUEST, net(), "cloud") == pytest.approx(0.05 + 0.1 + 10 * 0.02)


def test_download_bound_when_bandwidth_is_low():
    # 11 tokens * 100 bytes over 8 kbit/s = 1.1 s, longer than 10 * 0.02 s of generation
    assert predict_latency(PROFILES["cloud"], REQUEST, net(down=8_000), "cloud") == pytest.approx(0.05 + 0.1 + 1.1)


def test_cheapest_feasible_target_is_chosen():
    tau_q, tau_l = {"qa": 0.6}, {"qa": 1.0}
    assert qos_aware(REQUEST, PROFILES, net(), tau_q, tau_l) == "edge"


def test_deadline_excludes_slow_target():
    tau_q, tau_l = {"qa": 0.6}, {"qa": 0.4}
    assert qos_aware(REQUEST, PROFILES, net(), tau_q, tau_l) == "cloud"


def test_fallback_maximises_utility_when_nothing_is_feasible():
    tau_q, tau_l = {"qa": 0.95}, {"qa": 0.2}
    assert qos_aware(REQUEST, PROFILES, net(), tau_q, tau_l) == "local"


def test_quality_cost_ignores_latency():
    assert quality_cost(REQUEST, PROFILES, net(rtt=10.0), tau_q={"qa": 0.8}) == "cloud"


def test_first_token_bytes_count_against_bandwidth():
    one_token = Request(cls="qa", n_in=0, s_up=0)
    prof = Profile(q={"qa": 1.0}, n_out={"qa": 1}, ttft0=0.1, prefill=0.0, decode=0.02, bytes_per_tok=100)
    # 100 bytes over 800 bit/s take 1 s even though there is nothing left to decode
    assert predict_latency(prof, one_token, net(rtt=0.1, down=800), "cloud") == pytest.approx(0.1 + 0.1 + 1.0)

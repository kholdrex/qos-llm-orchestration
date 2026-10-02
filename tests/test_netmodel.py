import pytest

from qos_orchestration.netmodel import INITIAL_WINDOW, round_trips, t_net, tx_simple, tx_slowstart


def test_serialisation_and_unlimited_bandwidth():
    assert tx_simple(125_000, 1e6) == pytest.approx(1.0)
    assert tx_simple(125_000, 0) == 0.0


def test_round_trips():
    assert round_trips() == 1
    assert round_trips(new_conn=True) == 2
    assert round_trips(new_conn=True, tls=True) == 3


def test_slow_start_adds_idle_round_trips():
    rtt, bw = 0.1, 100e6
    size = 3 * INITIAL_WINDOW  # needs one window doubling before the data is covered
    assert tx_slowstart(size, bw, rtt) == pytest.approx(tx_simple(size, bw) + rtt - tx_simple(INITIAL_WINDOW, bw))


def test_slow_start_irrelevant_for_small_transfers():
    assert tx_slowstart(1_000, 10e6, 0.2) == pytest.approx(tx_simple(1_000, 10e6))


def test_t_net_warm_connection():
    assert t_net(0.05, 1_000, 4_000, 8e6, 8e6, slowstart=False) == pytest.approx(0.05 + 0.001 + 0.004)

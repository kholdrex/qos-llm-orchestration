"""Network part of the end-to-end latency, eqs. (2)-(3) of the paper.

    T_net = k * RTT + T_tx(S_up, B_up) + T_tx(S_down, B_down)

k = 1 on a warm keep-alive connection, 2 when a new TCP connection is opened, 3 with a TLS 1.3 handshake.
T_tx is either plain serialisation 8 S / B, or serialisation plus the idle time of TCP slow start while the
congestion window grows from the initial window to the bandwidth-delay product.
Bandwidth is in bit/s, sizes in bytes, times in seconds; a bandwidth of 0 means "not limiting".
"""

MSS = 1448
INITIAL_WINDOW = 10 * MSS


def tx_simple(size, bandwidth, rtt=None):
    return 0.0 if not bandwidth else 8.0 * size / bandwidth


def tx_slowstart(size, bandwidth, rtt, initial_window=INITIAL_WINDOW):
    total = tx_simple(size, bandwidth)
    bdp = float("inf") if not bandwidth else bandwidth * rtt / 8.0
    window, sent = initial_window, 0.0
    while sent + window < size and window < bdp:
        total += max(0.0, rtt - tx_simple(window, bandwidth))
        sent += window
        window *= 2
    return total


def round_trips(new_conn=False, tls=False):
    return 1 + int(new_conn) + int(new_conn and tls)


def t_net(rtt, s_up, s_down, b_up, b_down, new_conn=False, tls=False, slowstart=True):
    tx = tx_slowstart if slowstart else tx_simple
    return round_trips(new_conn, tls) * rtt + tx(s_up, b_up, rtt) + tx(s_down, b_down, rtt)

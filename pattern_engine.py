"""
pattern_engine.py
Phân tích mẫu hình giá toàn diện:
  - Classic: H&S, Double Top/Bottom, Cup & Handle, Rounding
  - Continuation: Triangle, Wedge, Flag, Pennant, Rectangle, Channel
  - Candlestick: Doji, Hammer, Engulfing, Morning/Evening Star, ...
  - Smart Money: Spring, Upthrust, Accumulation Box
Thuật toán: ZigZag peak/trough + Linear Regression + Touch Counting
"""
import numpy as np
import pandas as pd
from scipy.signal import argrelextrema
from scipy.stats import linregress


# ════════════════════════════════════════════════════════════════════
# PEAK / TROUGH DETECTION (ZigZag)
# ════════════════════════════════════════════════════════════════════

def find_peaks_troughs(close, order=5, min_pct=0.02):
    """
    Tìm đỉnh (peaks) và đáy (troughs) có ý nghĩa.
    order: số nến mỗi bên phải thấp/cao hơn
    min_pct: thay đổi tối thiểu để tính là đỉnh/đáy thực sự
    """
    arr = np.array(close)
    peak_idx  = argrelextrema(arr, np.greater_equal, order=order)[0]
    trough_idx= argrelextrema(arr, np.less_equal,    order=order)[0]

    # Lọc theo biên độ tối thiểu
    def filter_by_pct(indices, is_peak):
        valid = []
        for i in indices:
            left  = arr[max(0, i-order):i]
            right = arr[i+1:min(len(arr), i+order+1)]
            if len(left) == 0 or len(right) == 0:
                continue
            ref = arr[i]
            if is_peak:
                if ref > left.max() * (1 - min_pct) and ref > right.max() * (1 - min_pct):
                    valid.append(i)
            else:
                if ref < left.min() * (1 + min_pct) and ref < right.min() * (1 + min_pct):
                    valid.append(i)
        return np.array(valid)

    peaks   = filter_by_pct(peak_idx,   is_peak=True)
    troughs = filter_by_pct(trough_idx, is_peak=False)

    return peaks, troughs


def fit_trendline(x_indices, y_values):
    """Linear regression → trả về (slope, intercept, r2)."""
    if len(x_indices) < 2:
        return 0, 0, 0
    slope, intercept, r, _, _ = linregress(x_indices, y_values)
    return slope, intercept, r**2


def trendline_at(slope, intercept, x):
    return slope * x + intercept


def count_touches(arr, slope, intercept, indices, tolerance=0.015):
    """Đếm số lần giá chạm trendline (trong ngưỡng tolerance %)."""
    touches = 0
    for i in indices:
        expected = trendline_at(slope, intercept, i)
        if abs(arr[i] - expected) / (expected + 1e-9) <= tolerance:
            touches += 1
    return touches


# ════════════════════════════════════════════════════════════════════
# CLASSIC PATTERNS
# ════════════════════════════════════════════════════════════════════

def detect_head_and_shoulders(df, order=5):
    """Head & Shoulders và Inverse H&S."""
    close  = df['close'].astype(float).values
    volume = df['volume'].astype(float).values
    n      = len(close)
    results= []

    peaks,   _ = find_peaks_troughs(close, order=order)
    _, troughs = find_peaks_troughs(close, order=order)

    # Cần ít nhất 3 đỉnh gần nhau
    for i in range(len(peaks) - 2):
        p1, p2, p3 = peaks[i], peaks[i+1], peaks[i+2]
        if p3 - p1 < 20:
            continue
        v1, v2, v3 = close[p1], close[p2], close[p3]

        # H&S điều kiện: đỉnh giữa cao nhất, 2 vai gần bằng nhau
        if (v2 > v1 and v2 > v3                       # head cao hơn
                and abs(v1 - v3) / v2 < 0.08           # 2 vai gần bằng
                and abs(p2 - p1 - (p3 - p2)) / (p3 - p1) < 0.4):  # đối xứng

            # Tìm neckline từ 2 troughs giữa p1–p2 và p2–p3
            tr_mid1 = [t for t in troughs if p1 < t < p2]
            tr_mid2 = [t for t in troughs if p2 < t < p3]
            if tr_mid1 and tr_mid2:
                n1, n2 = tr_mid1[-1], tr_mid2[0]
                nk_slope, nk_inter, _ = fit_trendline([n1, n2], [close[n1], close[n2]])
                neckline_now = trendline_at(nk_slope, nk_inter, n - 1)

                # Volume: giảm dần từ LS → H → RS
                vol_ls = volume[max(0,p1-3):p1+1].mean()
                vol_h  = volume[max(0,p2-3):p2+1].mean()
                vol_confirmed = vol_h < vol_ls * 1.1

                target = neckline_now - (v2 - neckline_now)
                results.append({
                    'pattern':   'Head & Shoulders',
                    'type':      'bearish',
                    'signal':    'SELL',
                    'confidence': 0.80 if vol_confirmed else 0.60,
                    'points':    {'LS':(p1,v1),'Head':(p2,v2),'RS':(p3,v3),
                                  'N1':(n1,close[n1]),'N2':(n2,close[n2])},
                    'neckline':  (nk_slope, nk_inter),
                    'target':    round(target, 0),
                    'broken':    close[-1] < neckline_now,
                    'description': f"H&S: Head={v2:,.0f} | Neckline={neckline_now:,.0f} | Target={target:,.0f}",
                })

        # Inverse H&S
        if (v2 < v1 and v2 < v3
                and abs(v1 - v3) / (v2 + 1e-9) < 0.08
                and abs(p2 - p1 - (p3 - p2)) / (p3 - p1) < 0.4):

            pk_mid1 = [t for t in peaks if p1 < t < p2]
            pk_mid2 = [t for t in peaks if p2 < t < p3]
            if pk_mid1 and pk_mid2:
                n1, n2 = pk_mid1[-1], pk_mid2[0]
                nk_slope, nk_inter, _ = fit_trendline([n1, n2], [close[n1], close[n2]])
                neckline_now = trendline_at(nk_slope, nk_inter, n - 1)
                target = neckline_now + (neckline_now - v2)
                results.append({
                    'pattern':    'Inverse H&S',
                    'type':       'bullish',
                    'signal':     'BUY',
                    'confidence': 0.80,
                    'points':     {'LS':(p1,v1),'Head':(p2,v2),'RS':(p3,v3),
                                   'N1':(n1,close[n1]),'N2':(n2,close[n2])},
                    'neckline':   (nk_slope, nk_inter),
                    'target':     round(target, 0),
                    'broken':     close[-1] > neckline_now,
                    'description': f"IH&S: Head={v2:,.0f} | Neckline={neckline_now:,.0f} | Target={target:,.0f}",
                })

    return results[-1:] if results else []   # trả về mẫu hình gần nhất


def detect_double_top_bottom(df, order=5, tolerance=0.03, min_gap=10):
    """Double Top và Double Bottom."""
    close  = df['close'].astype(float).values
    n      = len(close)
    results= []

    peaks,   _ = find_peaks_troughs(close, order=order)
    _, troughs = find_peaks_troughs(close, order=order)

    # Double Top
    for i in range(len(peaks) - 1):
        p1, p2 = peaks[i], peaks[i + 1]
        if p2 - p1 < min_gap:
            continue
        v1, v2 = close[p1], close[p2]
        if abs(v1 - v2) / ((v1 + v2) / 2) < tolerance:
            tr_between = [t for t in troughs if p1 < t < p2]
            if tr_between:
                valley = tr_between[np.argmin([close[t] for t in tr_between])]
                target = close[valley] - (max(v1, v2) - close[valley])
                results.append({
                    'pattern':    'Double Top',
                    'type':       'bearish',
                    'signal':     'SELL',
                    'confidence': 0.75,
                    'points':     {'T1':(p1,v1),'T2':(p2,v2),'Valley':(valley,close[valley])},
                    'neckline':   (0, close[valley]),
                    'target':     round(target, 0),
                    'broken':     close[-1] < close[valley],
                    'description': f"Double Top: T1={v1:,.0f} T2={v2:,.0f} | Support={close[valley]:,.0f} | Target={target:,.0f}",
                })

    # Double Bottom
    for i in range(len(troughs) - 1):
        t1, t2 = troughs[i], troughs[i + 1]
        if t2 - t1 < min_gap:
            continue
        v1, v2 = close[t1], close[t2]
        if abs(v1 - v2) / ((v1 + v2) / 2) < tolerance:
            pk_between = [p for p in peaks if t1 < p < t2]
            if pk_between:
                peak_b = pk_between[np.argmax([close[p] for p in pk_between])]
                target = close[peak_b] + (close[peak_b] - min(v1, v2))
                results.append({
                    'pattern':    'Double Bottom',
                    'type':       'bullish',
                    'signal':     'BUY',
                    'confidence': 0.75,
                    'points':     {'B1':(t1,v1),'B2':(t2,v2),'Peak':(peak_b,close[peak_b])},
                    'neckline':   (0, close[peak_b]),
                    'target':     round(target, 0),
                    'broken':     close[-1] > close[peak_b],
                    'description': f"Double Bottom: B1={v1:,.0f} B2={v2:,.0f} | Resist={close[peak_b]:,.0f} | Target={target:,.0f}",
                })

    return results[-1:] if results else []


# ════════════════════════════════════════════════════════════════════
# CONTINUATION PATTERNS
# ════════════════════════════════════════════════════════════════════

def detect_triangles(df, order=4, min_touches=2, lookback=60):
    """Symmetrical, Ascending, Descending Triangle."""
    close  = df['close'].astype(float)
    window = close.iloc[-lookback:].values
    n      = len(window)
    x      = np.arange(n)
    results= []

    peaks,   _ = find_peaks_troughs(pd.Series(window), order=order)
    _, troughs = find_peaks_troughs(pd.Series(window), order=order)

    if len(peaks) < 2 or len(troughs) < 2:
        return results

    # Fit trendlines
    resist_slope, resist_inter, resist_r2 = fit_trendline(peaks, window[peaks])
    support_slope, support_inter, support_r2 = fit_trendline(troughs, window[troughs])

    # Touch counts
    all_idx = np.arange(n)
    resist_touches = count_touches(window, resist_slope, resist_inter, peaks)
    support_touches= count_touches(window, support_slope, support_inter, troughs)

    if resist_touches < min_touches or support_touches < min_touches:
        return results

    # Converging = triangle
    if abs(resist_slope - support_slope) < 0.01 * window.mean() / n:
        return results  # parallel = channel, not triangle

    # Symmetrical: both converging
    if resist_slope < -0.001 and support_slope > 0.001:
        # Calculate apex
        apex_x = (support_inter - resist_inter) / (resist_slope - support_slope)
        target_up   = trendline_at(resist_slope, resist_inter, apex_x) * 1.10
        target_down = trendline_at(support_slope, support_inter, apex_x) * 0.90
        results.append({
            'pattern':    'Symmetrical Triangle',
            'type':       'neutral',
            'signal':     'WATCH',
            'confidence': min(0.5 + (resist_touches + support_touches) * 0.05, 0.85),
            'trendlines': {
                'resistance': (resist_slope, resist_inter),
                'support':    (support_slope, support_inter),
            },
            'touches':    {'resist': resist_touches, 'support': support_touches},
            'apex_x':     int(apex_x) if apex_x > 0 else n + 10,
            'target_up':  round(target_up, 0),
            'target_down':round(target_down, 0),
            'lookback_offset': len(close) - lookback,
            'description': f"Symmetrical Triangle | Touches: {resist_touches}R/{support_touches}S | Apex ~{int(max(0,apex_x-n))} nến nữa",
        })

    # Ascending Triangle: flat resistance, rising support
    elif abs(resist_slope) < 0.0005 * window.mean() / n and support_slope > 0.001:
        flat_resist = np.mean(window[peaks])
        target = flat_resist + (flat_resist - window[troughs].min())
        results.append({
            'pattern':    'Ascending Triangle',
            'type':       'bullish',
            'signal':     'BUY',
            'confidence': 0.75,
            'trendlines': {
                'resistance': (0, flat_resist),
                'support':    (support_slope, support_inter),
            },
            'touches':    {'resist': resist_touches, 'support': support_touches},
            'target_up':  round(target, 0),
            'lookback_offset': len(close) - lookback,
            'description': f"Ascending Triangle | Resistance={flat_resist:,.0f} | Target={target:,.0f}",
        })

    # Descending Triangle: falling resistance, flat support
    elif abs(support_slope) < 0.0005 * window.mean() / n and resist_slope < -0.001:
        flat_support = np.mean(window[troughs])
        target = flat_support - (window[peaks].max() - flat_support)
        results.append({
            'pattern':    'Descending Triangle',
            'type':       'bearish',
            'signal':     'SELL',
            'confidence': 0.70,
            'trendlines': {
                'resistance': (resist_slope, resist_inter),
                'support':    (0, flat_support),
            },
            'touches':    {'resist': resist_touches, 'support': support_touches},
            'target_down':round(target, 0),
            'lookback_offset': len(close) - lookback,
            'description': f"Descending Triangle | Support={flat_support:,.0f} | Target={target:,.0f}",
        })

    return results


def detect_wedges(df, order=4, lookback=60):
    """Rising Wedge (bearish) và Falling Wedge (bullish)."""
    close  = df['close'].astype(float)
    window = close.iloc[-lookback:].values
    n      = len(window)
    results= []

    peaks,   _ = find_peaks_troughs(pd.Series(window), order=order)
    _, troughs = find_peaks_troughs(pd.Series(window), order=order)

    if len(peaks) < 2 or len(troughs) < 2:
        return results

    rs, ri, rr2 = fit_trendline(peaks,   window[peaks])
    ss, si, sr2 = fit_trendline(troughs, window[troughs])

    # Cả hai hướng cùng chiều nhưng converging = wedge
    same_dir  = (rs > 0 and ss > 0) or (rs < 0 and ss < 0)
    converging= abs(rs - ss) > 0.0002 * window.mean() / n

    if not (same_dir and converging):
        return results

    # Rising Wedge: cả hai đi lên nhưng resist thoải hơn support
    if rs > 0 and ss > 0 and ss > rs:
        target = trendline_at(ss, si, n) * 0.92
        results.append({
            'pattern':    'Rising Wedge',
            'type':       'bearish',
            'signal':     'SELL',
            'confidence': 0.70,
            'trendlines': {'resistance':(rs,ri), 'support':(ss,si)},
            'target_down':round(target, 0),
            'lookback_offset': len(close) - lookback,
            'description': f"Rising Wedge (Bearish) | Breakout xuống kỳ vọng | Target={target:,.0f}",
        })

    # Falling Wedge: cả hai đi xuống nhưng support thoải hơn resist
    elif rs < 0 and ss < 0 and abs(ss) < abs(rs):
        target = trendline_at(rs, ri, n) * 1.08
        results.append({
            'pattern':    'Falling Wedge',
            'type':       'bullish',
            'signal':     'BUY',
            'confidence': 0.72,
            'trendlines': {'resistance':(rs,ri), 'support':(ss,si)},
            'target_up':  round(target, 0),
            'lookback_offset': len(close) - lookback,
            'description': f"Falling Wedge (Bullish) | Breakout lên kỳ vọng | Target={target:,.0f}",
        })

    return results


def detect_flags_pennants(df, order=3, lookback=40):
    """Bull Flag, Bear Flag, Pennant."""
    close  = df['close'].astype(float).values
    volume = df['volume'].astype(float).values
    n      = len(close)
    results= []

    # Tìm flagpole: biến động mạnh trong 10-20 nến gần nhất
    pole_window = 15
    if n < pole_window + lookback:
        return results

    pre_flag = close[-(pole_window + lookback): -lookback]
    flag_zone= close[-lookback:]
    flag_vol = volume[-lookback:]

    pole_move = (pre_flag[-1] - pre_flag[0]) / pre_flag[0]

    # Bull Flag: pole tăng mạnh, flag consolidate nhẹ
    if pole_move > 0.08:
        flag_range = (flag_zone.max() - flag_zone.min()) / flag_zone.mean()
        flag_slope = linregress(np.arange(len(flag_zone)), flag_zone)[0]
        vol_decline= flag_vol.mean() < volume[-(pole_window+lookback):-lookback].mean() * 0.7

        if flag_range < 0.06 and flag_slope < 0 and vol_decline:
            target = flag_zone[-1] + (pre_flag[-1] - pre_flag[0])
            results.append({
                'pattern':    'Bull Flag',
                'type':       'bullish',
                'signal':     'BUY',
                'confidence': 0.75 if vol_decline else 0.60,
                'target_up':  round(target, 0),
                'pole_move':  round(pole_move * 100, 1),
                'description': f"Bull Flag | Pole +{pole_move:.1%} | Volume decline={vol_decline} | Target={target:,.0f}",
            })
        elif flag_range < 0.04:
            target = flag_zone[-1] + (pre_flag[-1] - pre_flag[0])
            results.append({
                'pattern':    'Pennant (Bullish)',
                'type':       'bullish',
                'signal':     'BUY',
                'confidence': 0.70,
                'target_up':  round(target, 0),
                'description': f"Bullish Pennant | Pole +{pole_move:.1%} | Target={target:,.0f}",
            })

    # Bear Flag
    elif pole_move < -0.08:
        flag_slope = linregress(np.arange(len(flag_zone)), flag_zone)[0]
        flag_range = (flag_zone.max() - flag_zone.min()) / flag_zone.mean()
        vol_decline= flag_vol.mean() < volume[-(pole_window+lookback):-lookback].mean() * 0.7

        if flag_range < 0.06 and flag_slope > 0 and vol_decline:
            target = flag_zone[-1] + (pre_flag[-1] - pre_flag[0])
            results.append({
                'pattern':    'Bear Flag',
                'type':       'bearish',
                'signal':     'SELL',
                'confidence': 0.72,
                'target_down':round(target, 0),
                'pole_move':  round(pole_move * 100, 1),
                'description': f"Bear Flag | Pole {pole_move:.1%} | Target={target:,.0f}",
            })

    return results


def detect_channel(df, order=4, lookback=80):
    """Ascending / Descending / Horizontal Channel."""
    close  = df['close'].astype(float)
    window = close.iloc[-lookback:].values
    n      = len(window)
    results= []

    peaks,   _ = find_peaks_troughs(pd.Series(window), order=order)
    _, troughs = find_peaks_troughs(pd.Series(window), order=order)

    if len(peaks) < 2 or len(troughs) < 2:
        return results

    rs, ri, _ = fit_trendline(peaks,   window[peaks])
    ss, si, _ = fit_trendline(troughs, window[troughs])

    slope_diff = abs(rs - ss)
    parallel   = slope_diff < 0.0003 * window.mean() / n
    if not parallel:
        return results

    resist_now = trendline_at(rs, ri, n - 1)
    support_now= trendline_at(ss, si, n - 1)
    channel_h  = resist_now - support_now
    last_price = window[-1]
    pos_in_ch  = (last_price - support_now) / (channel_h + 1e-9)

    if rs > 0.001:
        label = 'Ascending Channel'
        btype = 'bullish'
        signal= 'BUY' if pos_in_ch < 0.3 else 'HOLD'
    elif rs < -0.001:
        label = 'Descending Channel'
        btype = 'bearish'
        signal= 'SELL' if pos_in_ch > 0.7 else 'HOLD'
    else:
        label = 'Horizontal Channel'
        btype = 'neutral'
        signal= 'BUY' if pos_in_ch < 0.2 else ('SELL' if pos_in_ch > 0.8 else 'HOLD')

    results.append({
        'pattern':    label,
        'type':       btype,
        'signal':     signal,
        'confidence': 0.70,
        'trendlines': {'resistance':(rs,ri), 'support':(ss,si)},
        'pos_in_channel': round(pos_in_ch, 2),
        'support_now':    round(support_now, 0),
        'resist_now':     round(resist_now, 0),
        'lookback_offset': len(close) - lookback,
        'description': (f"{label} | Support={support_now:,.0f} | Resist={resist_now:,.0f} "
                        f"| Giá ở {pos_in_ch:.0%} kênh"),
    })

    return results


def detect_cup_handle(df, order=5, lookback=120):
    """Cup & Handle (Bullish Continuation)."""
    close  = df['close'].astype(float).values
    n      = len(close)
    results= []
    if n < lookback:
        return results

    window = close[-lookback:]
    _, troughs = find_peaks_troughs(pd.Series(window), order=order)

    if len(troughs) < 1:
        return results

    # Cup: đáy nằm ở giữa, 2 đầu (rim) gần bằng nhau
    deepest = troughs[np.argmin(window[troughs])]
    left_rim  = window[:deepest].max()
    right_rim = window[deepest:].max()
    cup_depth = (max(left_rim, right_rim) - window[deepest]) / max(left_rim, right_rim)

    if 0.10 < cup_depth < 0.40 and abs(left_rim - right_rim) / max(left_rim, right_rim) < 0.10:
        # Handle: consolidation nhỏ sau right rim
        handle = window[int(deepest + (len(window)-deepest)*0.6):]
        handle_range = (handle.max() - handle.min()) / handle.mean() if len(handle) > 3 else 1

        if handle_range < 0.06:
            target = right_rim + (right_rim - window[deepest])
            results.append({
                'pattern':    'Cup & Handle',
                'type':       'bullish',
                'signal':     'BUY',
                'confidence': 0.78,
                'cup_depth':  round(cup_depth * 100, 1),
                'target_up':  round(target, 0),
                'rim_level':  round(right_rim, 0),
                'lookback_offset': n - lookback,
                'description': f"Cup & Handle | Depth={cup_depth:.0%} | Rim={right_rim:,.0f} | Target={target:,.0f}",
            })

    return results


# ════════════════════════════════════════════════════════════════════
# CANDLESTICK PATTERNS
# ════════════════════════════════════════════════════════════════════

def detect_candlestick_patterns(df, lookback=5):
    """
    Phát hiện các mẫu nến Nhật trong lookback nến gần nhất.
    """
    o = df['open'].astype(float)
    h = df['high'].astype(float)
    l = df['low'].astype(float)
    c = df['close'].astype(float)
    v = df['volume'].astype(float)

    results = []
    n       = len(df)

    # ── Hàm tiện ích ──────────────────────────────────────────────
    def body(i):    return abs(c.iloc[i] - o.iloc[i])
    def rng(i):     return h.iloc[i] - l.iloc[i]
    def upper(i):   return h.iloc[i] - max(c.iloc[i], o.iloc[i])
    def lower(i):   return min(c.iloc[i], o.iloc[i]) - l.iloc[i]
    def is_bull(i): return c.iloc[i] > o.iloc[i]
    def is_bear(i): return c.iloc[i] < o.iloc[i]
    def avg_body(start, end): return np.mean([body(i) for i in range(start, end)])

    last = n - 1

    # ── Doji ──────────────────────────────────────────────────────
    if rng(last) > 0 and body(last) / rng(last) < 0.1:
        results.append({'name':'Doji','type':'neutral','signal':'WATCH',
            'confidence':0.55,'bar':last,
            'desc':'Doji: thị trường do dự – cần xác nhận hướng đi'})

    # ── Hammer / Hanging Man ──────────────────────────────────────
    if (lower(last) > body(last) * 2
            and upper(last) < body(last) * 0.5
            and rng(last) > 0):
        if c.iloc[max(0,last-5):last].mean() < c.iloc[last]:
            results.append({'name':'Hammer','type':'bullish','signal':'BUY',
                'confidence':0.68,'bar':last,
                'desc':'Hammer: đuôi dài dưới = phe mua chiếm ưu thế'})
        else:
            results.append({'name':'Hanging Man','type':'bearish','signal':'SELL',
                'confidence':0.62,'bar':last,
                'desc':'Hanging Man: cảnh báo đảo chiều giảm'})

    # ── Shooting Star / Inverted Hammer ──────────────────────────
    if (upper(last) > body(last) * 2
            and lower(last) < body(last) * 0.5
            and rng(last) > 0):
        if c.iloc[max(0,last-5):last].mean() > c.iloc[last]:
            results.append({'name':'Shooting Star','type':'bearish','signal':'SELL',
                'confidence':0.68,'bar':last,
                'desc':'Shooting Star: từ chối vùng giá cao = phe bán mạnh'})
        else:
            results.append({'name':'Inverted Hammer','type':'bullish','signal':'BUY',
                'confidence':0.58,'bar':last,
                'desc':'Inverted Hammer: tiềm năng đảo chiều tăng'})

    # ── Engulfing ─────────────────────────────────────────────────
    if n >= 2:
        prev = last - 1
        if (is_bear(prev) and is_bull(last)
                and c.iloc[last] > o.iloc[prev]
                and o.iloc[last] < c.iloc[prev]
                and body(last) > body(prev)):
            results.append({'name':'Bullish Engulfing','type':'bullish','signal':'BUY',
                'confidence':0.75,'bar':last,
                'desc':'Bullish Engulfing: nến xanh nuốt nến đỏ – đảo chiều tăng mạnh'})

        if (is_bull(prev) and is_bear(last)
                and o.iloc[last] > c.iloc[prev]
                and c.iloc[last] < o.iloc[prev]
                and body(last) > body(prev)):
            results.append({'name':'Bearish Engulfing','type':'bearish','signal':'SELL',
                'confidence':0.75,'bar':last,
                'desc':'Bearish Engulfing: nến đỏ nuốt nến xanh – đảo chiều giảm mạnh'})

    # ── Morning Star / Evening Star ───────────────────────────────
    if n >= 3:
        p2, p1, p0 = last-2, last-1, last
        if (is_bear(p2)
                and body(p1) < avg_body(max(0,p2-5), p2) * 0.4
                and is_bull(p0)
                and c.iloc[p0] > (o.iloc[p2] + c.iloc[p2]) / 2):
            results.append({'name':'Morning Star','type':'bullish','signal':'BUY',
                'confidence':0.78,'bar':last,
                'desc':'Morning Star: 3 nến đảo chiều tăng – tín hiệu mua mạnh'})

        if (is_bull(p2)
                and body(p1) < avg_body(max(0,p2-5), p2) * 0.4
                and is_bear(p0)
                and c.iloc[p0] < (o.iloc[p2] + c.iloc[p2]) / 2):
            results.append({'name':'Evening Star','type':'bearish','signal':'SELL',
                'confidence':0.78,'bar':last,
                'desc':'Evening Star: 3 nến đảo chiều giảm – tín hiệu bán mạnh'})

    # ── Three White Soldiers / Three Black Crows ──────────────────
    if n >= 3:
        i1, i2, i3 = last-2, last-1, last
        avg_b = avg_body(max(0,last-10), last-2)
        if (all(is_bull(i) for i in [i1,i2,i3])
                and all(body(i) > avg_b * 0.8 for i in [i1,i2,i3])
                and c.iloc[i2] > c.iloc[i1] and c.iloc[i3] > c.iloc[i2]):
            results.append({'name':'Three White Soldiers','type':'bullish','signal':'BUY',
                'confidence':0.80,'bar':last,
                'desc':'Three White Soldiers: 3 nến xanh liên tiếp = xu hướng tăng mạnh'})

        if (all(is_bear(i) for i in [i1,i2,i3])
                and all(body(i) > avg_b * 0.8 for i in [i1,i2,i3])
                and c.iloc[i2] < c.iloc[i1] and c.iloc[i3] < c.iloc[i2]):
            results.append({'name':'Three Black Crows','type':'bearish','signal':'SELL',
                'confidence':0.80,'bar':last,
                'desc':'Three Black Crows: 3 nến đỏ liên tiếp = xu hướng giảm mạnh'})

    # ── Volume-confirmed Breakout candle ──────────────────────────
    vol_ma20 = v.rolling(20).mean().iloc[-1]
    if (is_bull(last)
            and body(last) > avg_body(max(0,last-20), last) * 1.5
            and v.iloc[-1] > vol_ma20 * 1.8):
        results.append({'name':'Volume Breakout Candle','type':'bullish','signal':'BUY',
            'confidence':0.82,'bar':last,
            'desc':'Nến bứt phá volume lớn: tổ chức tham gia – xác nhận breakout'})

    return results


# ════════════════════════════════════════════════════════════════════
# SMART MONEY PATTERNS
# ════════════════════════════════════════════════════════════════════

def detect_smart_money_patterns(df):
    """Spring, Upthrust, Accumulation Box (Wyckoff-based)."""
    close  = df['close'].astype(float).values
    volume = df['volume'].astype(float).values
    low    = df['low'].astype(float).values
    high   = df['high'].astype(float).values
    n      = len(close)
    results= []

    if n < 30:
        return results

    vol_ma = np.convolve(volume, np.ones(20)/20, mode='valid')
    support_20 = np.min(low[-20:])
    resist_20  = np.max(high[-20:])

    # Spring: giá chọc xuống dưới support nhưng đóng cửa trên
    if (low[-1] < support_20 * 0.99
            and close[-1] > support_20
            and volume[-1] < vol_ma[-1] * 1.2):
        results.append({
            'name':'Spring',
            'type':'bullish',
            'signal':'STRONG_BUY',
            'confidence':0.82,
            'description':f"Spring: Giá test dưới support {support_20:,.0f} với volume thấp → tổ chức không bán",
        })

    # Upthrust: giá chọc lên trên kháng cự nhưng đóng cửa dưới
    if (high[-1] > resist_20 * 1.01
            and close[-1] < resist_20
            and volume[-1] > vol_ma[-1] * 1.3):
        results.append({
            'name':'Upthrust',
            'type':'bearish',
            'signal':'SELL',
            'confidence':0.78,
            'description':f"Upthrust: Giá fake break kháng cự {resist_20:,.0f} với volume cao → tổ chức phân phối",
        })

    # Accumulation Box: sideways, volume giảm dần
    price_range = (resist_20 - support_20) / np.mean(close[-20:])
    vol_trend   = np.mean(volume[-5:]) / (np.mean(volume[-20:]) + 1e-9)
    if price_range < 0.08 and vol_trend < 0.7:
        results.append({
            'name':'Accumulation Box',
            'type':'bullish',
            'signal':'ACCUMULATION',
            'confidence':0.72,
            'description':f"Accumulation Box: Biên độ {price_range:.1%} | Volume dry-up {vol_trend:.0%} → đang tích lũy",
        })

    return results


# ════════════════════════════════════════════════════════════════════
# MASTER SCANNER
# ════════════════════════════════════════════════════════════════════

def scan_all_patterns(df, fast=False):
    """
    Quét toàn bộ mẫu hình. fast=True bỏ qua các pattern nặng.
    Returns: dict với danh sách patterns theo nhóm + summary
    """
    all_patterns = {
        'classic':      [],
        'continuation': [],
        'candlestick':  [],
        'smart_money':  [],
    }

    if len(df) < 30:
        return all_patterns

    try: all_patterns['classic']     += detect_head_and_shoulders(df)
    except: pass
    try: all_patterns['classic']     += detect_double_top_bottom(df)
    except: pass
    if not fast:
        try: all_patterns['classic'] += detect_cup_handle(df)
        except: pass

    try: all_patterns['continuation']+= detect_triangles(df)
    except: pass
    try: all_patterns['continuation']+= detect_wedges(df)
    except: pass
    try: all_patterns['continuation']+= detect_flags_pennants(df)
    except: pass
    try: all_patterns['continuation']+= detect_channel(df)
    except: pass

    try: all_patterns['candlestick'] += detect_candlestick_patterns(df)
    except: pass
    try: all_patterns['smart_money'] += detect_smart_money_patterns(df)
    except: pass

    # Tổng hợp tín hiệu
    all_pats = (all_patterns['classic'] + all_patterns['continuation'] +
                all_patterns['candlestick'] + all_patterns['smart_money'])

    buy_score  = sum(p.get('confidence',0.5) for p in all_pats if p.get('type') == 'bullish')
    sell_score = sum(p.get('confidence',0.5) for p in all_pats if p.get('type') == 'bearish')
    total      = buy_score + sell_score + 1e-9

    all_patterns['summary'] = {
        'total_patterns':  len(all_pats),
        'bullish_count':   len([p for p in all_pats if p.get('type') == 'bullish']),
        'bearish_count':   len([p for p in all_pats if p.get('type') == 'bearish']),
        'buy_score':       round(buy_score, 2),
        'sell_score':      round(sell_score, 2),
        'bias':            'BULLISH' if buy_score > sell_score * 1.2
                           else ('BEARISH' if sell_score > buy_score * 1.2 else 'NEUTRAL'),
        'bias_strength':   round(abs(buy_score - sell_score) / total, 2),
        'top_patterns':    sorted(all_pats, key=lambda x: x.get('confidence',0), reverse=True)[:3],
    }

    return all_patterns


# ════════════════════════════════════════════════════════════════════
# VISUALIZATION
# ════════════════════════════════════════════════════════════════════

def plot_patterns(df, patterns_dict, symbol='', figsize=(14, 8)):
    """Vẽ biểu đồ giá với mẫu hình được highlight."""
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    import matplotlib.dates as mdates

    close = df['close'].astype(float)
    high  = df['high'].astype(float)
    low   = df['low'].astype(float)
    idx   = np.arange(len(close))
    dates = df.index

    fig, ax = plt.subplots(figsize=figsize, facecolor='#0e1117')
    ax.set_facecolor('#0e1117')
    ax.tick_params(colors='#9ca3af', labelsize=8)
    ax.spines[:].set_color('#374151')

    # Price line
    ax.plot(dates, close, color='#60a5fa', lw=1.3, zorder=3)

    COLOR_MAP = {'bullish':'#22c55e', 'bearish':'#ef4444', 'neutral':'#f59e0b'}

    # Vẽ trendlines cho các pattern có trendlines
    all_pats = (patterns_dict.get('classic',[]) +
                patterns_dict.get('continuation',[]))

    for pat in all_pats:
        col = COLOR_MAP.get(pat.get('type','neutral'), '#6b7280')
        tl  = pat.get('trendlines', {})
        offset = pat.get('lookback_offset', 0)

        for line_name, (slope, inter) in tl.items():
            start_x = offset
            end_x   = len(close) - 1
            y_start = trendline_at(slope, inter, start_x - offset)
            y_end   = trendline_at(slope, inter, end_x - offset)
            ls = '--' if line_name == 'resistance' else '-.'
            ax.plot([dates[start_x], dates[end_x]], [y_start, y_end],
                    color=col, lw=1.2, ls=ls, alpha=0.85, zorder=4)

        # Neckline
        if 'neckline' in pat:
            nk = pat['neckline']
            nk_y = trendline_at(nk[0], nk[1], len(close) - 1)
            ax.axhline(nk_y, color=col, lw=1.0, ls=':', alpha=0.7)
            ax.text(dates[-1], nk_y, f" Neckline {nk_y:,.0f}", color=col,
                    fontsize=7, va='center', ha='left')

        # Target line
        for tgt_key in ('target_up', 'target_down', 'target'):
            tgt = pat.get(tgt_key)
            if tgt:
                ax.axhline(tgt, color=col, lw=0.8, ls='--', alpha=0.5)
                ax.text(dates[len(dates)//2], tgt, f" Target {tgt:,.0f}",
                        color=col, fontsize=7, va='bottom')

        # Pattern label
        label_x = dates[min(pat.get('lookback_offset', 0) + 5, len(dates)-1)]
        label_y = float(close.iloc[min(pat.get('lookback_offset',0)+5, len(close)-1)])
        ax.annotate(pat.get('pattern',''), xy=(label_x, label_y),
                    fontsize=7, color=col,
                    bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e2a3a', edgecolor=col, alpha=0.8))

    # Candlestick patterns markers
    for pat in patterns_dict.get('candlestick', []):
        bar_i = pat.get('bar', len(close)-1)
        col   = COLOR_MAP.get(pat.get('type','neutral'), '#6b7280')
        marker= '^' if pat.get('type') == 'bullish' else ('v' if pat.get('type') == 'bearish' else 'o')
        y_val = float(low.iloc[bar_i]) * 0.99 if pat.get('type') == 'bullish' else float(high.iloc[bar_i]) * 1.01
        ax.scatter(dates[bar_i], y_val, marker=marker, color=col, s=80, zorder=5)
        ax.annotate(pat['name'], xy=(dates[bar_i], y_val),
                    xytext=(0, 10 if pat.get('type')=='bullish' else -10),
                    textcoords='offset points', fontsize=6, color=col, ha='center')

    # Smart money markers
    for pat in patterns_dict.get('smart_money', []):
        col = '#a78bfa' if pat.get('type') == 'bullish' else '#f87171'
        ax.axhspan(float(low.iloc[-20:].min()) * 0.995,
                   float(low.iloc[-20:].min()) * 1.005,
                   alpha=0.2, color=col)
        ax.text(dates[-10], float(close.iloc[-1]),
                f" {pat['name']}", color=col, fontsize=7,
                bbox=dict(boxstyle='round', facecolor='#1e2a3a', edgecolor=col, alpha=0.7))

    ax.set_title(f"{symbol} — Pattern Analysis", color='white', fontsize=12, pad=8)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f'{x:,.0f}'))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=2))
    plt.setp(ax.xaxis.get_majorticklabels(), rotation=30, ha='right')

    # Legend
    legend_patches = [
        mpatches.Patch(color='#22c55e', label='Bullish Pattern'),
        mpatches.Patch(color='#ef4444', label='Bearish Pattern'),
        mpatches.Patch(color='#f59e0b', label='Neutral Pattern'),
        mpatches.Patch(color='#a78bfa', label='Smart Money'),
    ]
    ax.legend(handles=legend_patches, loc='upper left', fontsize=7,
              facecolor='#1f2937', labelcolor='white', framealpha=0.8)

    plt.tight_layout()
    return fig

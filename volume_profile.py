"""
volume_profile.py
Volume Profile: POC, VAL, VAH, HVN, LVN + Cumulative Delta + Absorption Signal
"""
import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════
# VOLUME PROFILE
# ════════════════════════════════════════════════════════════════════════

def compute_volume_profile(ohlcv_df: pd.DataFrame, bins: int = 50,
                            value_area_pct: float = 0.70,
                            lookback: int = 60) -> dict:
    """
    Tính Volume Profile cho lookback sessions gần nhất.
    Returns: POC, VAH, VAL, HVN, LVN, profile DataFrame
    """
    if ohlcv_df is None or len(ohlcv_df) < 10:
        return {}

    df = ohlcv_df.iloc[-lookback:].copy()
    close  = df['close'].astype(float)
    high   = df['high'].astype(float)
    low    = df['low'].astype(float)
    volume = df['volume'].astype(float)

    price_min = float(low.min())
    price_max = float(high.max())
    if price_max <= price_min:
        return {}

    # Tạo price bins
    edges      = np.linspace(price_min, price_max, bins + 1)
    bin_centers= (edges[:-1] + edges[1:]) / 2
    vol_per_bin= np.zeros(bins)

    # Phân phối volume vào bins theo OHLC
    for _, row in df.iterrows():
        lo, hi, vol = float(row['low']), float(row['high']), float(row['volume'])
        rng = hi - lo
        if rng == 0:
            idx = np.searchsorted(edges, lo, side='right') - 1
            idx = max(0, min(idx, bins - 1))
            vol_per_bin[idx] += vol
        else:
            for b in range(bins):
                b_lo, b_hi = edges[b], edges[b + 1]
                overlap = max(0, min(hi, b_hi) - max(lo, b_lo))
                if overlap > 0:
                    vol_per_bin[b] += vol * (overlap / rng)

    total_vol = vol_per_bin.sum()
    if total_vol == 0:
        return {}

    # POC = bin có volume cao nhất
    poc_idx   = int(np.argmax(vol_per_bin))
    poc_price = float(bin_centers[poc_idx])

    # Value Area: 70% volume quanh POC
    sorted_idx   = np.argsort(vol_per_bin)[::-1]
    cumvol       = 0
    va_indices   = []
    for i in sorted_idx:
        if cumvol >= total_vol * value_area_pct:
            break
        va_indices.append(i)
        cumvol += vol_per_bin[i]

    va_indices   = sorted(va_indices)
    val_price    = float(bin_centers[va_indices[0]])
    vah_price    = float(bin_centers[va_indices[-1]])

    # HVN / LVN
    vol_mean  = vol_per_bin.mean()
    hvn_bins  = [(bin_centers[i], vol_per_bin[i]) for i in range(bins) if vol_per_bin[i] > vol_mean * 1.5]
    lvn_bins  = [(bin_centers[i], vol_per_bin[i]) for i in range(bins) if vol_per_bin[i] < vol_mean * 0.5]

    current_price = float(close.iloc[-1])

    # Vị trí giá trong VP
    if vah_price > val_price:
        pos_in_va = (current_price - val_price) / (vah_price - val_price)
    else:
        pos_in_va = 0.5

    # Tín hiệu
    signals   = []
    bias      = 'NEUTRAL'

    if current_price > vah_price:
        signals.append('Giá trên VAH → đang trong uptrend, có thể pullback về VAH')
        bias = 'BULLISH'
    elif current_price < val_price:
        signals.append('Giá dưới VAL → yếu, có thể test POC')
        bias = 'BEARISH'
    elif abs(current_price - poc_price) / poc_price < 0.02:
        signals.append('Giá tại POC → vùng cân bằng, chờ breakout')
    elif current_price > poc_price:
        signals.append('Giá trên POC → bias tăng trong VA')
        bias = 'BULLISH'
    else:
        signals.append('Giá dưới POC → bias giảm trong VA')
        bias = 'BEARISH'

    # LVN gần nhất phía trên = kháng cự mỏng (dễ break)
    lvn_above = [p for p, v in lvn_bins if p > current_price]
    lvn_below = [p for p, v in lvn_bins if p < current_price]
    hvn_above = [p for p, v in hvn_bins if p > current_price]
    hvn_below = [p for p, v in hvn_bins if p < current_price]

    return {
        'poc':         round(poc_price, 0),
        'val':         round(val_price, 0),
        'vah':         round(vah_price, 0),
        'current':     round(current_price, 0),
        'pos_in_va':   round(pos_in_va, 2),
        'bias':        bias,
        'signals':     signals,
        'lvn_above':   [round(p, 0) for p in lvn_above[:3]],
        'lvn_below':   [round(p, 0) for p in lvn_below[-3:]],
        'hvn_above':   [round(p, 0) for p in hvn_above[:3]],
        'hvn_below':   [round(p, 0) for p in hvn_below[-3:]],
        'profile_bins':     bin_centers.tolist(),
        'profile_volumes':  vol_per_bin.tolist(),
        'lookback_sessions':lookback,
    }


# ════════════════════════════════════════════════════════════════════════
# CUMULATIVE DELTA
# ════════════════════════════════════════════════════════════════════════

def compute_cumulative_delta(ohlcv_df: pd.DataFrame) -> dict:
    """
    Delta proxy = Vol_green - Vol_red (ước lượng mua/bán chủ động).
    Cumulative Delta divergence → hidden accumulation/distribution.
    """
    if ohlcv_df is None or len(ohlcv_df) < 10:
        return {}

    close  = ohlcv_df['close'].astype(float)
    open_  = ohlcv_df['open'].astype(float)
    volume = ohlcv_df['volume'].astype(float)
    high   = ohlcv_df['high'].astype(float)
    low    = ohlcv_df['low'].astype(float)

    # Delta proxy: bullish nến = mua, bearish = bán
    body   = close - open_
    delta  = np.where(body >= 0, volume, -volume)
    cum_delta = pd.Series(delta, index=ohlcv_df.index).cumsum()

    last_cd  = float(cum_delta.iloc[-1])
    cd_5d    = float(cum_delta.iloc[-1] - cum_delta.iloc[-6]) if len(cum_delta) >= 6 else 0
    cd_20d   = float(cum_delta.iloc[-1] - cum_delta.iloc[-21]) if len(cum_delta) >= 21 else 0

    # Divergence detection (20-day window)
    window  = min(20, len(close))
    c_win   = close.iloc[-window:]
    cd_win  = cum_delta.iloc[-window:]

    # Price tạo đáy mới nhưng Delta không tạo đáy mới → bullish divergence
    price_lo = float(c_win.iloc[-1]) < float(c_win.min()) * 1.005
    cd_lo    = float(cd_win.iloc[-1]) > float(cd_win.min()) * 0.95

    # Price tạo đỉnh mới nhưng Delta không tạo đỉnh mới → bearish divergence
    price_hi = float(c_win.iloc[-1]) > float(c_win.max()) * 0.995
    cd_hi    = float(cd_win.iloc[-1]) < float(cd_win.max()) * 1.05

    divergence = None
    if price_lo and cd_lo:
        divergence = 'BULLISH'
    elif price_hi and not cd_hi:
        divergence = 'BEARISH'

    # Daily delta (3 ngày gần nhất)
    recent_deltas = [float(d) for d in delta[-3:]]
    consecutive_buy  = all(d > 0 for d in recent_deltas)
    consecutive_sell = all(d < 0 for d in recent_deltas)

    signals = []
    if consecutive_buy:
        signals.append('Delta dương liên tiếp 3 phiên → phe mua kiểm soát')
    if consecutive_sell:
        signals.append('Delta âm liên tiếp 3 phiên → phe bán áp đảo')
    if divergence == 'BULLISH':
        signals.append('Bullish Divergence: Giá đáy thấp hơn, Delta đáy cao hơn → hidden accumulation')
    if divergence == 'BEARISH':
        signals.append('Bearish Divergence: Giá đỉnh cao hơn, Delta đỉnh thấp hơn → distribution')

    return {
        'cum_delta_last':  round(last_cd, 0),
        'delta_5d':        round(cd_5d, 0),
        'delta_20d':       round(cd_20d, 0),
        'divergence':      divergence,
        'consecutive_buy': consecutive_buy,
        'consecutive_sell':consecutive_sell,
        'recent_deltas':   [round(d, 0) for d in recent_deltas],
        'signals':         signals,
        'cum_delta_series':cum_delta.tolist(),
    }


# ════════════════════════════════════════════════════════════════════════
# ABSORPTION SIGNAL
# ════════════════════════════════════════════════════════════════════════

def detect_absorption(ohlcv_df: pd.DataFrame, lookback: int = 5) -> dict:
    """
    Absorption = Volume cao + thân nến nhỏ + bóng dài tại support/resistance.
    Smart money đang gom/bán mà không đẩy giá.
    """
    if ohlcv_df is None or len(ohlcv_df) < 20:
        return {}

    df = ohlcv_df.copy()
    close  = df['close'].astype(float)
    open_  = df['open'].astype(float)
    high   = df['high'].astype(float)
    low    = df['low'].astype(float)
    volume = df['volume'].astype(float)

    avg_vol  = volume.rolling(20).mean()
    avg_body = (close - open_).abs().rolling(20).mean()

    results = []

    for i in range(-lookback, 0):
        idx  = len(df) + i
        if idx < 20: continue

        vol_ratio  = float(volume.iloc[i] / (avg_vol.iloc[i] + 1e-9))
        body_ratio = float(abs(close.iloc[i] - open_.iloc[i]) / (avg_body.iloc[i] + 1e-9))
        shadow_up  = float(high.iloc[i] - max(close.iloc[i], open_.iloc[i]))
        shadow_dn  = float(min(close.iloc[i], open_.iloc[i]) - low.iloc[i])
        total_rng  = float(high.iloc[i] - low.iloc[i]) + 1e-9

        # Absorption: vol cao + body nhỏ
        if vol_ratio > 1.5 and body_ratio < 0.5:
            signal_type = None

            # Absorption tại support (bóng dưới dài)
            if shadow_dn / total_rng > 0.4:
                signal_type = 'SUPPORT_ABSORPTION'
                desc = f'Hấp thụ tại hỗ trợ: Vol {vol_ratio:.1f}x TB, thân nhỏ, bóng dưới dài → Smart money gom'

            # Absorption tại resistance (bóng trên dài)
            elif shadow_up / total_rng > 0.4:
                signal_type = 'RESISTANCE_ABSORPTION'
                desc = f'Hấp thụ tại kháng cự: Vol {vol_ratio:.1f}x TB, thân nhỏ, bóng trên dài → Smart money phân phối'

            if signal_type:
                results.append({
                    'bar_offset':  i,
                    'date':        str(df.index[i])[:10] if hasattr(df.index[i], '__str__') else '',
                    'type':        signal_type,
                    'vol_ratio':   round(vol_ratio, 2),
                    'body_ratio':  round(body_ratio, 2),
                    'description': desc,
                    'price':       float(close.iloc[i]),
                    'is_bullish':  signal_type == 'SUPPORT_ABSORPTION',
                })

    return {
        'absorptions': results,
        'has_support_absorption':    any(r['type'] == 'SUPPORT_ABSORPTION'    for r in results),
        'has_resistance_absorption': any(r['type'] == 'RESISTANCE_ABSORPTION' for r in results),
        'count': len(results),
    }


# ════════════════════════════════════════════════════════════════════════
# SMART MONEY CHECKLIST
# ════════════════════════════════════════════════════════════════════════

def smart_money_checklist(ohlcv_df, vp: dict, cum_delta: dict,
                           absorption: dict, flow_summary: dict = {}) -> dict:
    """
    4-point SMC checklist từ MASTER_PROMPT.
    ≥3/4: High confidence signal
    """
    checks = {}

    # 1. Volume >1.5x TB khi breakout
    if not ohlcv_df.empty:
        vol = ohlcv_df['volume'].astype(float)
        close = ohlcv_df['close'].astype(float)
        vol_ratio = float(vol.iloc[-1] / (vol.rolling(20).mean().iloc[-1] + 1e-9))
        is_breakout = float(close.iloc[-1]) > float(ohlcv_df['high'].astype(float).rolling(20).max().iloc[-2])
        checks['volume_breakout'] = vol_ratio > 1.5 and is_breakout

    # 2. Delta dương liên tiếp 3 phiên
    checks['delta_consecutive'] = cum_delta.get('consecutive_buy', False)

    # 3. Absorption tại key level
    checks['absorption_at_key'] = absorption.get('has_support_absorption', False)

    # 4. Foreign/Block flow đồng thuận
    ff_net = float(flow_summary.get('foreign_net_30d', 0) or 0)
    checks['foreign_flow_bullish'] = ff_net > 0

    passed = sum(1 for v in checks.values() if v)
    confidence = 'HIGH' if passed >= 3 else ('MEDIUM' if passed >= 2 else 'LOW')

    return {
        'checks':     checks,
        'passed':     passed,
        'confidence': confidence,
        'signal':     'STRONG_BUY' if passed >= 3 else ('BUY' if passed >= 2 else 'WATCH'),
        'description': f"SMC Checklist: {passed}/4 điều kiện — {confidence} confidence",
    }


# ════════════════════════════════════════════════════════════════════════
# FORMAT REPORT
# ════════════════════════════════════════════════════════════════════════

def format_vp_report(vp: dict) -> str:
    if not vp:
        return "Volume Profile: Không đủ dữ liệu"
    lines = [
        "=== VOLUME PROFILE ===",
        f"POC:  {vp.get('poc',0):,.0f}  ← Vùng giá trị cao nhất",
        f"VAH:  {vp.get('vah',0):,.0f}  ← Biên trên Value Area (70%)",
        f"VAL:  {vp.get('val',0):,.0f}  ← Biên dưới Value Area (70%)",
        f"Giá hiện tại: {vp.get('current',0):,.0f}  ({vp.get('bias','')})",
        f"Vị trí trong VA: {vp.get('pos_in_va',0):.0%}",
    ]
    if vp.get('lvn_above'):
        lines.append(f"LVN phía trên (vùng mỏng, dễ break): {vp['lvn_above']}")
    if vp.get('hvn_below'):
        lines.append(f"HVN phía dưới (hỗ trợ mạnh): {vp['hvn_below']}")
    for s in vp.get('signals', []):
        lines.append(f"  → {s}")
    return "\n".join(lines)


def format_smc_report(smc: dict, cum_delta: dict, absorption: dict) -> str:
    lines = ["=== SMART MONEY CONCEPTS ==="]
    lines.append(f"Checklist: {smc.get('passed',0)}/4  —  {smc.get('confidence','')} confidence")
    for k, v in smc.get('checks', {}).items():
        lines.append(f"  {'✅' if v else '❌'} {k.replace('_',' ').title()}")

    if cum_delta.get('divergence'):
        lines.append(f"\nCumulative Delta: {cum_delta['divergence']} DIVERGENCE")
    if cum_delta.get('signals'):
        for s in cum_delta['signals']:
            lines.append(f"  → {s}")

    if absorption.get('count', 0) > 0:
        lines.append(f"\nAbsorption: {absorption['count']} tín hiệu")
        for a in absorption.get('absorptions', []):
            lines.append(f"  → {a['description']}")

    return "\n".join(lines)

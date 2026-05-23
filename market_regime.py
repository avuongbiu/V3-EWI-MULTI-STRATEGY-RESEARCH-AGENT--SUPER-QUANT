"""
market_regime.py
Market Regime AI: Bull / Bear / Sideway / Crash Risk
Dựa trên: VNIndex, Breadth, Liquidity, Foreign Flow, Macro
"""
import numpy as np
import pandas as pd

REGIMES = {
    'BULL':        {'label': '🟢 Bull Market',    'strategy': 'momentum – tăng tỷ trọng',   'weight': 1.2},
    'BULL_WEAK':   {'label': '🟡 Bull yếu',       'strategy': 'selective – chọn lọc',       'weight': 1.0},
    'SIDEWAY':     {'label': '🟡 Sideway',         'strategy': 'swing – mua hỗ trợ bán kháng cự', 'weight': 0.8},
    'BEAR':        {'label': '🔴 Bear Market',     'strategy': 'defensive – giảm tỷ trọng', 'weight': 0.4},
    'CRASH_RISK':  {'label': '🚨 Crash Risk',      'strategy': 'cash – thoát toàn bộ',      'weight': 0.0},
}


def classify_market_regime(vnindex_df, breadth_df=None, macro_dict=None):
    """
    Phân loại Market Regime từ VNIndex + breadth + macro.
    Returns: dict với regime, score, signals, strategy
    """
    if vnindex_df is None or vnindex_df.empty or len(vnindex_df) < 50:
        return _regime_result('SIDEWAY', 0.5, ['Không đủ dữ liệu VNIndex'])

    close = vnindex_df['close'].astype(float) if 'close' in vnindex_df.columns else vnindex_df.iloc[:, 0].astype(float)

    # ── Trend indicators ──────────────────────────────────────────
    ma20  = close.rolling(20).mean().iloc[-1]
    ma50  = close.rolling(50).mean().iloc[-1]
    ma200 = close.rolling(200).mean().iloc[-1] if len(close) >= 200 else ma50
    last  = float(close.iloc[-1])

    ret_1m  = float(close.pct_change(20).iloc[-1])
    ret_3m  = float(close.pct_change(60).iloc[-1]) if len(close) >= 60 else ret_1m
    ret_6m  = float(close.pct_change(120).iloc[-1]) if len(close) >= 120 else ret_1m

    above_ma20  = last > ma20
    above_ma50  = last > ma50
    above_ma200 = last > ma200
    ma20_above_ma50 = ma20 > ma50

    # ── Volatility ────────────────────────────────────────────────
    vol_20 = float(close.pct_change().rolling(20).std().iloc[-1]) * np.sqrt(252)
    drawdown = float((close / close.rolling(252).max() - 1).iloc[-1]) if len(close) >= 252 else 0

    signals  = []
    score    = 0.0   # positive = bull, negative = bear

    # MA structure
    if above_ma200 and above_ma50 and above_ma20:
        score += 0.3; signals.append('Giá > MA20 > MA50 > MA200 ✓')
    elif above_ma50:
        score += 0.1; signals.append('Giá > MA50')
    elif not above_ma200:
        score -= 0.3; signals.append('Giá < MA200 – xu hướng dài hạn yếu')

    # Momentum
    if ret_3m > 0.08: score += 0.2; signals.append(f'Momentum 3M: {ret_3m:+.1%}')
    elif ret_3m > 0:   score += 0.1
    elif ret_3m < -0.15: score -= 0.3; signals.append(f'Giảm mạnh 3M: {ret_3m:+.1%}')
    else:              score -= 0.1

    # Volatility
    if vol_20 > 0.40:
        score -= 0.2; signals.append(f'Volatility cao: {vol_20:.0%} annualized')
    if drawdown < -0.20:
        score -= 0.3; signals.append(f'Drawdown: {drawdown:.1%} từ đỉnh 1 năm')

    # Market breadth (nếu có)
    if breadth_df is not None and not breadth_df.empty:
        adv_col = next((c for c in breadth_df.columns if 'advanc' in c.lower() or 'tang' in c.lower()), None)
        dec_col = next((c for c in breadth_df.columns if 'declin' in c.lower() or 'giam' in c.lower()), None)
        if adv_col and dec_col:
            adv = float(breadth_df[adv_col].iloc[-1])
            dec = float(breadth_df[dec_col].iloc[-1])
            adv_ratio = adv / (adv + dec + 1e-9)
            if adv_ratio > 0.65: score += 0.1; signals.append(f'Breadth tốt: {adv_ratio:.0%} mã tăng')
            elif adv_ratio < 0.35: score -= 0.1; signals.append(f'Breadth yếu: {adv_ratio:.0%} mã tăng')

    # Macro context (nếu có)
    if macro_dict:
        liquidity_signal = macro_dict.get('liquidity_signal', 0)
        if liquidity_signal > 0: score += 0.1; signals.append('Thanh khoản vĩ mô hỗ trợ')
        elif liquidity_signal < 0: score -= 0.1; signals.append('Thanh khoản vĩ mô thắt chặt')

    # ── Classify ──────────────────────────────────────────────────
    if score >= 0.5:
        regime = 'BULL'
    elif score >= 0.2:
        regime = 'BULL_WEAK'
    elif score >= -0.1:
        regime = 'SIDEWAY'
    elif score >= -0.4:
        regime = 'BEAR'
    else:
        regime = 'CRASH_RISK'

    # Crash override: drawdown cực lớn + vol cao
    if drawdown < -0.30 and vol_20 > 0.50:
        regime = 'CRASH_RISK'
        signals.append('🚨 Crash conditions: drawdown sâu + volatility cao')

    confidence = min(abs(score) + 0.3, 0.95)

    return _regime_result(regime, confidence, signals,
                          ret_1m=ret_1m, ret_3m=ret_3m,
                          ma20=ma20, ma50=ma50, last_price=last)


def _regime_result(regime, confidence, signals, **kwargs):
    info = REGIMES.get(regime, REGIMES['SIDEWAY'])
    return {
        'regime':     regime,
        'label':      info['label'],
        'strategy':   info['strategy'],
        'weight':     info['weight'],
        'confidence': round(confidence, 2),
        'signals':    signals,
        **kwargs,
    }


def format_regime_report(r):
    lines = [
        "=== MARKET REGIME ===",
        f"Regime: {r['label']}",
        f"Strategy: {r['strategy']}",
        f"Confidence: {r['confidence']:.0%}",
        f"Allocation weight: {r['weight']:.1f}x",
    ]
    if r.get('ret_3m') is not None:
        lines.append(f"VNIndex 1M: {r.get('ret_1m',0):+.1%} | 3M: {r.get('ret_3m',0):+.1%}")
    lines.append("Signals:")
    for s in r['signals']:
        lines.append(f"  • {s}")
    return "\n".join(lines)


# ── Sector Rotation Map ───────────────────────────────────────────────
SECTOR_ROTATION = {
    'credit_growth':   ['Ngân hàng', 'Chứng khoán', 'Bất động sản'],
    'public_invest':   ['Xây dựng', 'Vật liệu xây dựng', 'Đá, Cát'],
    'power_shortage':  ['Điện', 'Năng lượng tái tạo'],
    'export_boom':     ['Dệt may', 'Thủy sản', 'Điện tử', 'Gỗ'],
    'consumption':     ['Bán lẻ', 'Tiêu dùng', 'Thực phẩm'],
    'rate_cut':        ['Bất động sản', 'Ngân hàng', 'Tiện ích'],
    'usd_strong':      ['Xuất khẩu', 'Dầu khí'],
    'gold_up':         ['Vàng', 'Tài nguyên'],
}


def get_sector_rotation_signal(macro_context):
    """
    Gợi ý sector hưởng lợi dựa trên macro context.
    macro_context: dict với keys từ SECTOR_ROTATION
    """
    recommendations = []
    for signal, sectors in SECTOR_ROTATION.items():
        if macro_context.get(signal, False):
            recommendations.append({
                'trigger': signal,
                'sectors': sectors,
            })
    return recommendations

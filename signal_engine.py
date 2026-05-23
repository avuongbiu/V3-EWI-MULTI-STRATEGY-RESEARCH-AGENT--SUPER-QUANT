"""
signal_engine.py
Signal Engine: BUY / STRONG_BUY / BREAKOUT / ACCUMULATION / HOLD / SELL
Logic: Wyckoff + Smart Money + Alpha Score + Regime + Macro
"""
import numpy as np

SIGNALS = {
    'STRONG_BUY':    {'label': '🚀 STRONG BUY',    'score': 1.0},
    'BREAKOUT':      {'label': '⚡ BREAKOUT',        'score': 0.9},
    'BUY':           {'label': '🟢 BUY',            'score': 0.7},
    'ACCUMULATION':  {'label': '🔵 ACCUMULATION',   'score': 0.5},
    'HOLD':          {'label': '🟡 HOLD',           'score': 0.0},
    'REDUCE':        {'label': '🟠 REDUCE',         'score': -0.5},
    'SELL':          {'label': '🔴 SELL',           'score': -1.0},
}


def generate_signal(alpha_scores, wyckoff_result, regime_result, fundamental=None):
    """
    Tổng hợp tất cả engine → tín hiệu cuối.
    """
    reasons    = []
    score      = 0.0
    conditions = []

    composite  = alpha_scores.get('composite', 0)
    momentum   = alpha_scores.get('momentum', 0)
    volume_sc  = alpha_scores.get('volume', 0)
    smart_money= alpha_scores.get('smart_money', 0)
    trend_sc   = alpha_scores.get('trend', 0)

    regime     = regime_result.get('regime', 'SIDEWAY')
    regime_w   = regime_result.get('weight', 0.8)

    wyckoff_ph = wyckoff_result.get('phase', 'UNKNOWN')
    wyckoff_c  = wyckoff_result.get('confidence', 0)
    is_buy_z   = wyckoff_result.get('is_buy_zone', False)
    is_sell_z  = wyckoff_result.get('is_sell_zone', False)

    # ── Logic: STRONG BUY ─────────────────────────────────────────
    # Breakout + Volume + Smart Money + Macro support
    strong_buy_conds = [
        alpha_scores.get('trend', 0) > 0.5,               # breakout confirmed
        volume_sc > 0.5,                                    # volume spike
        smart_money > 0.3,                                  # tổ chức mua
        regime in ('BULL', 'BULL_WEAK'),                    # macro support
        wyckoff_ph in ('D', 'E'),                           # đúng pha Wyckoff
    ]
    if sum(strong_buy_conds) >= 4:
        score = 0.95
        conditions.append('STRONG_BUY')
        reasons.append('✅ Breakout + Volume spike + Smart money + Macro bull + Wyckoff D/E')

    # ── Logic: BREAKOUT ───────────────────────────────────────────
    elif (alpha_scores.get('trend', 0) > 0.5
          and volume_sc > 0.3
          and composite > 0.4):
        score = 0.85
        conditions.append('BREAKOUT')
        reasons.append('✅ Breakout với volume xác nhận')
        if smart_money > 0: reasons.append('  + Smart money hỗ trợ')

    # ── Logic: ACCUMULATION → BUY ────────────────────────────────
    elif (wyckoff_ph in ('B', 'C')
          and wyckoff_c > 0.5
          and smart_money >= 0
          and volume_sc > -0.2
          and regime in ('BULL', 'BULL_WEAK', 'SIDEWAY')):
        if smart_money > 0.3:
            score = 0.75
            conditions.append('BUY')
            reasons.append('✅ Wyckoff B/C + Smart money accumulation')
        else:
            score = 0.50
            conditions.append('ACCUMULATION')
            reasons.append('🔵 Wyckoff B/C – đang tích lũy, chờ xác nhận')

    # ── Logic: BUY thông thường ───────────────────────────────────
    elif composite > 0.3 and regime in ('BULL', 'BULL_WEAK'):
        score = 0.65
        conditions.append('BUY')
        reasons.append('✅ Composite alpha tốt + Regime bull')

    # ── Logic: SELL ───────────────────────────────────────────────
    elif (is_sell_z
          or regime == 'CRASH_RISK'
          or (composite < -0.4 and momentum < -0.3)):
        score = -0.9
        conditions.append('SELL')
        if regime == 'CRASH_RISK': reasons.append('🚨 Crash Risk – thoát toàn bộ')
        if is_sell_z: reasons.append('⚠ Wyckoff Distribution – phân phối')
        if composite < -0.4: reasons.append('📉 Alpha composite rất tiêu cực')

    # ── Logic: REDUCE ─────────────────────────────────────────────
    elif (regime == 'BEAR'
          or (composite < -0.2 and momentum < 0)):
        score = -0.5
        conditions.append('REDUCE')
        reasons.append('⚠ Bear market / Momentum yếu – giảm tỷ trọng')

    # ── Default: HOLD ─────────────────────────────────────────────
    else:
        score = composite * 0.5
        conditions.append('HOLD')
        reasons.append('Tín hiệu trung tính – giữ nguyên')

    # ── Điều chỉnh theo regime weight ────────────────────────────
    final_score = score * regime_w
    signal_key  = _score_to_signal(final_score)

    # ── Fundamental filter ────────────────────────────────────────
    if fundamental:
        pe = fundamental.get('pe', None)
        roe = fundamental.get('roe', None)
        if pe and pe > 30 and signal_key in ('STRONG_BUY', 'BUY'):
            reasons.append(f'⚠ PE cao ({pe:.1f}x) – cẩn thận định giá')
        if roe and roe > 15:
            reasons.append(f'✅ ROE tốt: {roe:.1f}%')

    return {
        'signal':      signal_key,
        'label':       SIGNALS[signal_key]['label'],
        'score':       round(final_score, 3),
        'raw_score':   round(score, 3),
        'reasons':     reasons,
        'conditions':  conditions,
        'alpha_detail': alpha_scores,
        'wyckoff_phase': wyckoff_ph,
        'regime':      regime,
    }


def _score_to_signal(score):
    if score >= 0.85:   return 'STRONG_BUY'
    elif score >= 0.70: return 'BREAKOUT'
    elif score >= 0.50: return 'BUY'
    elif score >= 0.20: return 'ACCUMULATION'
    elif score >= -0.3: return 'HOLD'
    elif score >= -0.6: return 'REDUCE'
    else:               return 'SELL'


def format_signal_report(sig):
    lines = [
        "=== SIGNAL ENGINE ===",
        f"Tín hiệu: {sig['label']}",
        f"Score: {sig['score']:+.3f} (raw: {sig['raw_score']:+.3f})",
        f"Wyckoff Phase: {sig['wyckoff_phase']} | Regime: {sig['regime']}",
        "Lý do:",
    ]
    for r in sig['reasons']:
        lines.append(f"  {r}")
    a = sig.get('alpha_detail', {})
    if a:
        lines.append(f"Alpha: momentum={a.get('momentum',0):+.2f}  "
                     f"volume={a.get('volume',0):+.2f}  "
                     f"smart_money={a.get('smart_money',0):+.2f}  "
                     f"trend={a.get('trend',0):+.2f}")
    return "\n".join(lines)

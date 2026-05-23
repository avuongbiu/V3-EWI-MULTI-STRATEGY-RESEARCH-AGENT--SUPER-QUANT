"""
wyckoff.py
Tự động phát hiện Wyckoff Phase A → E
Logic: Volume + Price Action + Support/Resistance
"""
import numpy as np
import pandas as pd


PHASES = {
    'A': 'Dừng giảm (Stopping Action)',
    'B': 'Tích lũy (Building Cause)',
    'C': 'Spring / Test',
    'D': 'Markup bắt đầu',
    'E': 'Xu hướng tăng mạnh',
    'DISTRIBUTION': 'Phân phối (Chuẩn bị giảm)',
    'UNKNOWN': 'Chưa xác định',
}


def detect_wyckoff_phase(df):
    """
    Phát hiện Wyckoff phase từ OHLCV + features.
    Returns: dict với phase, confidence, signals, description
    """
    if len(df) < 50:
        return _result('UNKNOWN', 0.0, [], 'Chưa đủ dữ liệu (cần >= 50 ngày)')

    close  = df['close'].astype(float)
    volume = df['volume'].astype(float)
    high   = df['high'].astype(float)
    low    = df['low'].astype(float)

    # Tính các chỉ số cơ bản
    vol_ma20 = volume.rolling(20).mean()
    vol_ratio = volume / vol_ma20

    price_trend_20 = (close.iloc[-1] - close.iloc[-20]) / close.iloc[-20]
    price_trend_60 = (close.iloc[-1] - close.iloc[-60]) / close.iloc[-60] if len(df) >= 60 else 0

    # Range ngắn hạn so với trung hạn
    range_5  = high.rolling(5).max().iloc[-1]  - low.rolling(5).min().iloc[-1]
    range_20 = high.rolling(20).max().iloc[-1] - low.rolling(20).min().iloc[-1]
    range_60 = high.rolling(60).max().iloc[-1] - low.rolling(60).min().iloc[-1] if len(df) >= 60 else range_20

    tightness  = range_5 / (range_20 + 1e-9)         # < 0.3 = tight range
    volatility_ratio = range_20 / (range_60 + 1e-9)  # < 0.5 = contracting

    # Volume behavior
    recent_vol_trend = vol_ratio.iloc[-10:].mean()
    vol_declining    = vol_ratio.iloc[-5:].mean() < 0.6
    vol_expanding    = vol_ratio.iloc[-5:].mean() > 1.5
    vol_climax       = vol_ratio.iloc[-3:].max() > 3.0

    # Giá so với vùng support/resistance
    support_20  = low.rolling(20).min().iloc[-1]
    resist_20   = high.rolling(20).max().iloc[-1]
    price_pos   = (close.iloc[-1] - support_20) / (resist_20 - support_20 + 1e-9)

    # MA trend
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma_bull = float(ma20.iloc[-1]) > float(ma50.iloc[-1]) if len(df) >= 50 else None

    signals = []

    # ── PHASE E: Trend mạnh ──────────────────────────────────────
    if (price_trend_20 > 0.05
            and price_trend_60 > 0.10
            and ma_bull is True
            and not vol_climax
            and price_pos > 0.6):
        score = 0.7
        if vol_expanding: score += 0.15; signals.append('Volume expansion xác nhận trend')
        if tightness < 0.4: signals.append('Pullback nhẹ, trend vẫn tốt')
        return _result('E', min(score, 0.95), signals, PHASES['E'])

    # ── PHASE D: Markup bắt đầu ──────────────────────────────────
    if (price_trend_20 > 0.02
            and price_pos > 0.5
            and vol_expanding
            and volatility_ratio < 0.8):
        score = 0.65
        signals.append('Giá bứt phá khỏi vùng tích lũy')
        signals.append('Volume tăng xác nhận markup')
        if price_trend_60 < 0.05: score += 0.1; signals.append('Break from long base')
        return _result('D', min(score, 0.90), signals, PHASES['D'])

    # ── PHASE C: Spring / Test ────────────────────────────────────
    if (price_pos < 0.2
            and tightness < 0.35
            and vol_declining
            and abs(price_trend_20) < 0.03):
        score = 0.60
        signals.append('Giá test lại vùng support với volume thấp (Spring/Test)')
        signals.append('Volume dry-up → tổ chức không bán nữa')
        if vol_ratio.iloc[-1] < 0.5: score += 0.15; signals.append('Ultra-low volume = Last Point of Support')
        return _result('C', min(score, 0.90), signals, PHASES['C'])

    # ── PHASE B: Tích lũy ────────────────────────────────────────
    if (abs(price_trend_60) < 0.08
            and volatility_ratio < 0.7
            and 0.2 < price_pos < 0.8):
        score = 0.55
        signals.append('Giá đi ngang trong biên độ hẹp (Cause building)')
        if recent_vol_trend < 0.8:
            score += 0.1
            signals.append('Volume giảm dần trong sideways → accumulation')
        if tightness < 0.4:
            score += 0.1
            signals.append('Range thu hẹp → compression')
        return _result('B', min(score, 0.85), signals, PHASES['B'])

    # ── PHASE A: Dừng giảm ───────────────────────────────────────
    if (price_trend_60 < -0.10
            and vol_climax
            and price_pos < 0.3):
        score = 0.60
        signals.append('Volume climax sau đà giảm mạnh (Selling Climax)')
        signals.append('Dấu hiệu dừng giảm - Phase A')
        return _result('A', score, signals, PHASES['A'])

    # ── DISTRIBUTION ──────────────────────────────────────────────
    if (price_trend_20 < -0.03
            and price_pos > 0.7
            and vol_climax):
        signals.append('Volume climax gần đỉnh → dấu hiệu phân phối')
        return _result('DISTRIBUTION', 0.60, signals, PHASES['DISTRIBUTION'])

    # Không đủ tín hiệu rõ
    dominant = 'B' if abs(price_trend_20) < 0.03 else ('D' if price_trend_20 > 0 else 'A')
    return _result(dominant, 0.30, ['Tín hiệu chưa rõ ràng'], PHASES.get(dominant, ''))


def _result(phase, confidence, signals, description):
    return {
        'phase':       phase,
        'confidence':  round(confidence, 2),
        'signals':     signals,
        'description': description,
        'label':       f'Phase {phase}: {description}',
        'is_buy_zone': phase in ('B', 'C', 'D'),
        'is_sell_zone': phase in ('DISTRIBUTION', 'E') and confidence > 0.7,
    }


def format_wyckoff_report(result):
    lines = [
        f"=== WYCKOFF ANALYSIS ===",
        f"Phase: {result['phase']} ({result['description']})",
        f"Confidence: {result['confidence']:.0%}",
    ]
    if result['signals']:
        lines.append("Signals:")
        for s in result['signals']:
            lines.append(f"  • {s}")
    if result['is_buy_zone']:
        lines.append("→ VÙNG MUA (Phase B/C/D)")
    if result['is_sell_zone']:
        lines.append("→ CẢNH BÁO BÁN (Distribution / Phase E cuối)")
    return "\n".join(lines)

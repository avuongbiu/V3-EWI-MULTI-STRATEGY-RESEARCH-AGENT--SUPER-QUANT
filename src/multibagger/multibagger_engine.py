"""
multibagger_engine.py - VNSE Optimized Version
Adapted for Vietnam Stock Market (HOSE/HNX/UPCoM)
Ref: Yartseva (2025) + Local Market Dynamics (Foreign Flow, Liquidity, Dividends)
"""
import numpy as np
import pandas as pd
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MultibaggerConfig:
    # --- VALUE & FCF (Yartseva Core) ---
    min_fcf_p: float = 0.04       # FCF Yield ≥ 4%
    min_dividend_yield: float = 0.02 # Cổ tức ≥ 2% (An toàn cho VNSE)
    
    # --- GROWTH & QUALITY ---
    min_roe: float = 0.15         # ROE ≥ 15% (Tiêu chuẩn cao hơn VN)
    min_roa: float = 0.08         # ROA ≥ 8%
    min_revenue_growth: float = 0.10 # Tăng trưởng doanh thu ≥ 10%
    
    # --- VN SPECIFICS ---
    min_avg_volume: int = 100000  # Khối lượng khớp lệnh TB ≥ 100k (Tránh cổ phiếu chết)
    max_foreign_ownership: float = 0.49 # Ưu tiên room ngoại còn trống
    
    # --- TIMING ---
    max_price_range_12m: float = 0.40 # Mua khi giá chưa tăng quá 40% đỉnh 12T
    
    # --- SCORING ---
    buy_threshold: int = 6
    hold_threshold: int = 4

def safe_get(data, *keys, default=0.0):
    for k in keys:
        if isinstance(data, dict):
            data = data.get(k)
        else: return default
    return float(data) if data is not None else default

def compute_yartseva_features(symbol, quote, fundamental, ohlcv=None, macro_context={}):
    """Tính toán các yếu tố đặc thù cho VNSE"""
    feat = {'symbol': symbol, 'timestamp': datetime.now().isoformat()}
    
    # 1. LIQUIDITY CHECK (VN Specific)
    vol = safe_get(quote, 'volume', 'vol', default=0)
    feat['avg_volume'] = vol
    feat['is_liquid'] = vol >= 100000 # Filter nhanh

    # 2. VALUE: FCF/P & Dividend
    mcap = safe_get(quote, 'market_cap', 'marketCap', default=0)
    price = safe_get(quote, 'price', 'currentPrice', default=0)
    
    # Ưu tiên OCF nếu có, nếu không dùng Net Profit * 0.8 (Proxy FCF)
    net_profit = safe_get(fundamental, 'net_profit', 'net_profit_latest', default=0)
    ocf = safe_get(fundamental, 'cash_flow', 'operating_cash_flow', default=0)
    
    fcf_proxy = ocf if ocf > 0 else net_profit * 0.8
    fcf_p = fcf_proxy / mcap if mcap > 0 else 0
    
    dividend_yield = safe_get(fundamental, 'dividend_yield', default=0)
    
    feat['fcf_p'] = fcf_p
    feat['dividend_yield'] = dividend_yield

    # 3. FOREIGN OWNERSHIP (VN Catalyst)
    foreign_pct = safe_get(fundamental, 'foreign_ownership', 'foreign_ownership_ratio', default=0)
    # Room còn lại = 49% (hoặc 30% tùy ngành) - foreign_pct
    room_left = max(0, 0.49 - foreign_pct)
    feat['foreign_pct'] = foreign_pct
    feat['foreign_room'] = room_left

    # 4. PROFITABILITY (ROE/ROA)
    roe = safe_get(fundamental, 'roe', 'ROE', default=0)
    roa = safe_get(fundamental, 'roa', 'ROA', default=0)
    feat['roe'] = roe
    feat['roa'] = roa

    # 5. GROWTH
    rev_growth = safe_get(fundamental, 'revenue_growth', 'revenue_growth_yoy', default=0)
    feat['revenue_growth'] = rev_growth

    # 6. TIMING (Price Range)
    h52 = safe_get(quote, 'price_52w_high', 'high52Week', default=0)
    l52 = safe_get(quote, 'price_52w_low', 'low52Week', default=0)
    
    pr = 0.5
    if h52 > l52 and price > 0:
        pr = (price - l52) / (h52 - l52)
    feat['price_range_12m'] = pr

    # 7. MACRO (Interest Rate Environment)
    rate_trend = macro_context.get('rate_trend', 'stable')
    feat['rate_up_dummy'] = 1 if rate_trend == 'rising' else 0
    
    return feat

def calculate_vn_multibagger_score(features, cfg=None):
    if cfg is None: cfg = MultibaggerConfig()
    
    score = 0
    breakdown = {}
    penalty = 0

    # ── FILTER: Thanh khoản (Bắt buộc) ────────────────────────────
    if not features.get('is_liquid', False):
        return {'symbol': features['symbol'], 'score': 0, 'signal': 'AVOID', 
                'reason': 'Thanh khoản thấp (Rủi ro thanh khoản)', 'breakdown': {}}

    # 1. FCF Yield (+2 điểm)
    if features['fcf_p'] >= cfg.min_fcf_p:
        score += 2
        breakdown['fcf_p'] = {'pass': True, 'pts': 2}
    else:
        breakdown['fcf_p'] = {'pass': False, 'pts': 0}

    # 2. Dividend Bonus (+1 điểm - Đặc thù VN)
    if features['dividend_yield'] >= cfg.min_dividend_yield:
        score += 1
        breakdown['dividend'] = {'pass': True, 'pts': 1}
    else:
        breakdown['dividend'] = {'pass': False, 'pts': 0}

    # 3. Foreign Room Bonus (+1 điểm - Catalyst tăng giá)
    if features['foreign_room'] > 0.10: # Còn room > 10%
        score += 1
        breakdown['foreign_room'] = {'pass': True, 'pts': 1}
    else:
        breakdown['foreign_room'] = {'pass': False, 'pts': 0}

    # 4. ROE/ROA Quality (+1 điểm)
    if features['roe'] >= cfg.min_roe or features['roa'] >= cfg.min_roa:
        score += 1
        breakdown['profitability'] = {'pass': True, 'pts': 1}
    else:
        breakdown['profitability'] = {'pass': False, 'pts': 0}

    # 5. Growth (+1 điểm)
    if features['revenue_growth'] >= cfg.min_revenue_growth:
        score += 1
        breakdown['growth'] = {'pass': True, 'pts': 1}
    else:
        breakdown['growth'] = {'pass': False, 'pts': 0}

    # 6. Timing / Entry (+1 điểm)
    if features['price_range_12m'] <= cfg.max_price_range_12m:
        score += 1
        breakdown['timing'] = {'pass': True, 'pts': 1}
    else:
        breakdown['timing'] = {'pass': False, 'pts': 0}

    # ── PENALTY: Macro ───────────────────────────────────────────
    if features['rate_up_dummy'] == 1:
        penalty += 1 # Trừ điểm nếu lãi suất tăng

    final_score = score - penalty
    
    # Signal Logic
    signal = 'BUY' if final_score >= cfg.buy_threshold else ('HOLD' if final_score >= cfg.hold_threshold else 'AVOID')
    
    # Entry Zone Calculation
    low = features.get('price_52w_low', 0)
    entry = f"{low*1.05:,.0f} - {low*1.15:,.0f}" if low > 0 else "N/A"

    return {
        'symbol': features['symbol'],
        'score': final_score,
        'signal': signal,
        'entry_zone': entry,
        'breakdown': breakdown,
        'features': features
    }
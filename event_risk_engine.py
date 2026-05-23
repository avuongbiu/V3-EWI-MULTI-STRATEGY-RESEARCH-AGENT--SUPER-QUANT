"""
event_risk_engine.py
EWI Framework - Event Risk Filter
Event Risk Score = Σ(Group1-4) + Macro Adjustment + Commodity Adjustment
Score 0-4: Normal | 5-6: Reduce | 7-8: No new buy | ≥9: Exit all
"""
import numpy as np
import pandas as pd

# ── Action table ────────────────────────────────────────────────────────
ERS_ACTION_TABLE = {
    (0, 4):  {'color':'🟢','action':'Giao dịch bình thường','max_position':0.20,'stoploss':0.06},
    (5, 6):  {'color':'🟡','action':'Giảm tỷ trọng 30-50%, SL chặt','max_position':0.12,'stoploss':0.035},
    (7, 8):  {'color':'🟠','action':'Không mua mới, giảm 50-70% vị thế','max_position':0.04,'stoploss':0.025},
    (9, 99): {'color':'🔴','action':'Thoát toàn bộ, chỉ quan sát','max_position':0.00,'stoploss':None},
}

def get_ers_action(score):
    for (lo, hi), info in ERS_ACTION_TABLE.items():
        if lo <= score <= hi:
            return info
    return ERS_ACTION_TABLE[(9, 99)]


# ════════════════════════════════════════════════════════════════════════
# GROUP SCORING
# ════════════════════════════════════════════════════════════════════════

def score_group1_governance(fundamental: dict, news_text: str = '') -> dict:
    """
    Nhóm 1: Quản trị & Tài chính (Max 9 điểm)
    Cần: ratio data từ vnstock_data Fundamental
    """
    score = 0
    details = []

    # Guidance LNST điều chỉnh >15% YoY
    rev_growth = fundamental.get('revenue_growth', 0) or 0
    if abs(float(rev_growth)) > 15:
        score += 3
        details.append(f'⚠ Guidance LNST thay đổi >{rev_growth:.1f}% YoY: +3')

    # Nợ vay/VCSH >80%
    de = float(fundamental.get('debt_to_equity', 0) or 0)
    if de > 0.8:
        score += 3
        details.append(f'⚠ Nợ/Vốn = {de:.2f} (>80%): +3')

    # ROE yếu (proxy cho sức khoẻ tài chính)
    roe = float(fundamental.get('roe', 0) or 0)
    if roe < 5:
        score += 1
        details.append(f'⚠ ROE thấp = {roe:.1f}%: +1')

    # Kiểm tra news cho audit warning
    audit_keywords = ['ngoại trừ','kiểm toán','nhấn mạnh','audit','material weakness']
    if any(kw in news_text.lower() for kw in audit_keywords):
        score += 3
        details.append('⚠ Phát hiện rủi ro kiểm toán trong tin tức: +3')

    return {'group': 1, 'name': 'Quản trị & Tài chính', 'score': min(score, 9), 'max': 9, 'details': details}


def score_group2_capital_structure(fundamental: dict, news_text: str = '') -> dict:
    """
    Nhóm 2: Cấu trúc vốn & Pha loãng (Max 9 điểm)
    """
    score = 0
    details = []

    # Từ khoá pha loãng trong news
    dilution_keywords = ['phát hành','esop','chào bán','tăng vốn','trái phiếu chuyển đổi']
    bond_keywords     = ['đáo hạn','bond','trái phiếu']

    if any(kw in news_text.lower() for kw in dilution_keywords):
        score += 3
        details.append('⚠ Phát hiện kế hoạch phát hành CP/ESOP/chào bán: +3')

    if any(kw in news_text.lower() for kw in bond_keywords):
        score += 3
        details.append('⚠ Trái phiếu sắp đáo hạn: +3')

    # PE quá cao = định giá đắt, rủi ro điều chỉnh
    pe = float(fundamental.get('pe', 0) or 0)
    if pe > 25:
        score += 1
        details.append(f'⚠ PE = {pe:.1f}x (cao, rủi ro điều chỉnh): +1')

    return {'group': 2, 'name': 'Cấu trúc vốn & Pha loãng', 'score': min(score, 9), 'max': 9, 'details': details}


def score_group3_legal_policy(news_text: str = '', flow_summary: dict = {}) -> dict:
    """
    Nhóm 3: Pháp lý & Chính sách Ngành (Max 9 điểm)
    """
    score = 0
    details = []

    # Rủi ro pháp lý
    legal_kw  = ['thanh tra','kiểm tra','điều tra','phạt','vi phạm','khởi tố','cưỡng chế']
    policy_kw = ['siết','hạn chế','quy định mới','tăng thuế','phí mới','kiểm soát']
    insider_kw= ['cổ đông lớn bán','nội bộ bán','đăng ký bán']

    if any(kw in news_text.lower() for kw in legal_kw):
        score += 3
        details.append('⚠ Phát hiện rủi ro pháp lý/thanh tra: +3')

    if any(kw in news_text.lower() for kw in policy_kw):
        score += 3
        details.append('⚠ Ngành đang bị siết chính sách: +3')

    if any(kw in news_text.lower() for kw in insider_kw):
        score += 3
        details.append('⚠ Cổ đông nội bộ đăng ký bán: +3')

    return {'group': 3, 'name': 'Pháp lý & Chính sách', 'score': min(score, 9), 'max': 9, 'details': details}


def score_group4_cashflow_liquidity(ohlcv_df: pd.DataFrame, flow_summary: dict = {}) -> dict:
    """
    Nhóm 4: Dòng tiền & Thanh khoản (Max 3 điểm)
    Tính từ OHLCV + foreign flow data
    """
    score = 0
    details = []

    if ohlcv_df is None or ohlcv_df.empty or len(ohlcv_df) < 5:
        return {'group': 4, 'name': 'Dòng tiền & Thanh khoản', 'score': 0, 'max': 3, 'details': ['Không đủ dữ liệu']}

    close  = ohlcv_df['close'].astype(float)
    volume = ohlcv_df['volume'].astype(float)

    # Volume giảm >50% khi giá giảm (5 ngày gần nhất)
    recent_vol  = volume.iloc[-5:].mean()
    avg_vol_20  = volume.rolling(20).mean().iloc[-1]
    price_down  = close.iloc[-1] < close.iloc[-5]

    if price_down and recent_vol < avg_vol_20 * 0.5:
        score += 1
        details.append(f'⚠ Volume giảm {(1 - recent_vol/avg_vol_20):.0%} khi giá giảm: +1')

    # Foreign/Block sell liên tiếp
    ff_net_5d = flow_summary.get('foreign_net_5d', 0) or flow_summary.get('ff_net_5d', 0) or 0
    if float(ff_net_5d) < -500_000:
        score += 1
        details.append(f'⚠ Foreign sell ròng mạnh 5 ngày: {ff_net_5d:,.0f} cp: +1')

    # Giá phá hỗ trợ (20-day low)
    support_20 = ohlcv_df['low'].astype(float).rolling(20).min().iloc[-1]
    if close.iloc[-1] < support_20 * 1.005:
        score += 1
        details.append(f'⚠ Giá phá hỗ trợ 20 ngày ({support_20:,.0f}): +1')

    return {'group': 4, 'name': 'Dòng tiền & Thanh khoản', 'score': min(score, 3), 'max': 3, 'details': details}


# ════════════════════════════════════════════════════════════════════════
# MACRO & COMMODITY ADJUSTMENT
# ════════════════════════════════════════════════════════════════════════

MACRO_RULES_REDUCE = [
    {'key': 'fed_cut_dovish',        'desc': 'Fed cắt lãi + dovish guidance',              'delta': -1},
    {'key': 'china_stimulus_large',  'desc': 'Trung Quốc kích thích >500B USD',            'delta': -1},
    {'key': 'vn_disbursement_30pct', 'desc': 'VN giải ngân đầu tư công >30% kế hoạch Q2', 'delta': -1},
    {'key': 'foreign_inflow_2000b',  'desc': 'Ngoại mua ròng >2,000B VND/tuần',           'delta': -1},
    {'key': 'export_growth_10pct',   'desc': 'Xuất khẩu tăng >10% YoY 2 tháng liên tiếp', 'delta': -1},
]

MACRO_RULES_RAISE = [
    {'key': 'fed_cut_prob_low',      'desc': 'Xác suất Fed cắt <30%',                      'delta': +1},
    {'key': 'china_pmi_weak',        'desc': 'PMI Trung Quốc <48 liên tiếp 2 tháng',       'delta': +1},
    {'key': 'oil_above_100',         'desc': 'Brent dầu >100 USD kéo dài 2 tuần',          'delta': +2},
    {'key': 'vn_cpi_above_5p5',      'desc': 'CPI Việt Nam >5.5% YoY',                    'delta': +2},
    {'key': 'foreign_sell_3weeks',   'desc': 'Ngoại bán ròng >1,000B VND/tuần x3',        'delta': +1},
]

def compute_macro_adjustment(macro_context: dict) -> tuple[int, list]:
    """
    Tính macro adjustment từ dict context.
    macro_context keys = key trong MACRO_RULES_*
    """
    total_adj = 0
    details   = []

    for rule in MACRO_RULES_REDUCE:
        if macro_context.get(rule['key'], False):
            total_adj += rule['delta']
            details.append(f"✅ {rule['desc']}: {rule['delta']:+d}")

    for rule in MACRO_RULES_RAISE:
        if macro_context.get(rule['key'], False):
            total_adj += rule['delta']
            details.append(f"❌ {rule['desc']}: {rule['delta']:+d}")

    return total_adj, details


def compute_scenario_probability(ers_score: int, macro_context: dict = {}) -> dict:
    """
    Tính xác suất 3 kịch bản theo ERS score.
    Base: 55% ± Bear/Bull triggers
    """
    bear_triggers = sum(1 for r in MACRO_RULES_RAISE if macro_context.get(r['key'], False))
    bull_conf     = sum(1 for r in MACRO_RULES_REDUCE if macro_context.get(r['key'], False))

    # Công thức từ MASTER_PROMPT
    base = 0.55
    if bear_triggers >= 3:  base -= 0.10
    if bull_conf >= 3:      base += 0.10
    # ERS cao → giảm bull, tăng bear
    base -= (ers_score / 30) * 0.15
    base = max(0.25, min(0.75, base))

    bull = max(0.10, 0.25 + (bull_conf - bear_triggers) * 0.05)
    bear = max(0.10, 1 - base - bull)

    # Normalize
    total = base + bull + bear
    return {
        'base_case': round(base / total, 2),
        'bull_case': round(bull / total, 2),
        'bear_case': round(bear / total, 2),
        'bull_triggers': bull_conf,
        'bear_triggers': bear_triggers,
    }


# ════════════════════════════════════════════════════════════════════════
# MASTER ERS CALCULATOR
# ════════════════════════════════════════════════════════════════════════

def compute_ers(symbol: str, fundamental: dict, ohlcv_df,
                flow_summary: dict = {}, news_text: str = '',
                macro_context: dict = {}) -> dict:
    """
    Tính Event Risk Score đầy đủ.
    Returns: dict với score, action, position_max, groups, scenario
    """
    g1 = score_group1_governance(fundamental, news_text)
    g2 = score_group2_capital_structure(fundamental, news_text)
    g3 = score_group3_legal_policy(news_text, flow_summary)
    g4 = score_group4_cashflow_liquidity(ohlcv_df, flow_summary)

    base_score = g1['score'] + g2['score'] + g3['score'] + g4['score']

    macro_adj, macro_details = compute_macro_adjustment(macro_context)
    total_score = max(0, base_score + macro_adj)

    action_info = get_ers_action(total_score)
    scenario    = compute_scenario_probability(total_score, macro_context)

    return {
        'symbol':        symbol,
        'ers_score':     total_score,
        'base_score':    base_score,
        'macro_adj':     macro_adj,
        'color':         action_info['color'],
        'action':        action_info['action'],
        'max_position':  action_info['max_position'],
        'stoploss':      action_info['stoploss'],
        'groups':        [g1, g2, g3, g4],
        'macro_details': macro_details,
        'scenario':      scenario,
        'is_blocked':    total_score >= 9,
        'is_caution':    7 <= total_score <= 8,
    }


def format_ers_report(ers: dict) -> str:
    lines = [
        f"=== EVENT RISK SCORE: {ers['symbol']} ===",
        f"ERS Score: {ers['ers_score']}/30  {ers['color']}",
        f"Action: {ers['action']}",
        f"Max Position: {ers['max_position']:.0%}  |  Stop Loss: {(ers['stoploss'] or 0):.0%}",
        "",
        "Chi tiết nhóm điểm:",
    ]
    for g in ers['groups']:
        lines.append(f"  Nhóm {g['group']} {g['name']}: {g['score']}/{g['max']}")
        for d in g['details']:
            lines.append(f"    {d}")

    if ers['macro_details']:
        lines += ["", "Điều chỉnh Macro/Commodity:"]
        for d in ers['macro_details']:
            lines.append(f"  {d}")

    sc = ers['scenario']
    lines += [
        "",
        "Xác suất kịch bản:",
        f"  🟢 Bull:  {sc['bull_case']:.0%}",
        f"  🟡 Base:  {sc['base_case']:.0%}",
        f"  🔴 Bear:  {sc['bear_case']:.0%}",
    ]
    return "\n".join(lines)

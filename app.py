import io
import json
import tempfile
import os
import time
from datetime import datetime, timedelta
import numpy as np
import pandas as pd
pd.set_option('mode.chained_assignment', None)
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.gridspec as gridspec
import streamlit as st
import yaml
import seaborn as sns
from ai_agents import build_agents_from_config, extract_json, meta_ensemble
from pipeline import SYSTEM_PROMPT
from pathlib import Path
from pipeline import run_full_pipeline, load_config
from ai_agents import AGENT_PRESETS, MODEL_CATALOGUE
from data_fetcher import DataFetcher
from feature_engine import build_features, compute_alpha_factors, compute_alpha_score
from wyckoff import detect_wyckoff_phase, format_wyckoff_report
from market_regime import classify_market_regime, format_regime_report, get_sector_rotation_signal
from signal_engine import generate_signal, format_signal_report
from pattern_engine import scan_all_patterns, plot_patterns
from portfolio_engine import (compute_portfolio_allocation, compute_portfolio_risk,
                              format_portfolio_report, REGIME_ALLOCATION)
from src.multibagger.ui import render_multibagger_tab

# ── Page config ─────────────────────────────────────────────────────────
st.set_page_config(page_title="Super AI Quant VN", page_icon="🧠",
                   layout="wide", initial_sidebar_state="expanded")

# ── CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
.metric-card{background:#1e2a3a;border-radius:10px;padding:12px 16px;margin:4px 0}
.signal-buy{color:#22c55e;font-weight:bold;font-size:1.1em}
.signal-sell{color:#ef4444;font-weight:bold;font-size:1.1em}
.signal-hold{color:#f59e0b;font-weight:bold}
.signal-accum{color:#60a5fa;font-weight:bold}
.phase-badge{background:#1e3a5f;border-radius:6px;padding:2px 8px;font-size:0.85em}
</style>
""", unsafe_allow_html=True)

# ── Constants ────────────────────────────────────────────────────────────
SOURCES = ['VCI', 'TCBS', 'MSN']
VN30 = ['ACB','BCM','BID','BVH','CTG','FPT','GAS','GVR','HDB','HPG',
'MBB','MSN','MWG','PLX','POW','SAB','SHB','SSB','SSI','STB',
'TCB','TPB','VCB','VHM','VIB','VIC','VJC','VNM','VPB','VRE']
POPULAR = ['VCB','TCB','MBB','HPG','VNM','FPT','MWG','VIC','VHM','ACB','STB','SSI']
MIDCAP  = ['DGC','PNJ','KDH','NLG','VCI','BSI','HCM','EVF','GEX','PHR']
SIGNAL_COLOR = {
    'STRONG_BUY':'#22c55e','BREAKOUT':'#10b981','BUY':'#34d399',
    'ACCUMULATION':'#60a5fa','HOLD':'#f59e0b','REDUCE':'#fb923c','SELL':'#ef4444'
}
SIGNAL_ICON = {
    'STRONG_BUY':'🚀','BREAKOUT':'⚡','BUY':'🟢',
    'ACCUMULATION':'🔵','HOLD':'🟡','REDUCE':'🟠','SELL':'🔴'
}

# ═══════════════════════════════════════════════════════════════
# [MULTIBAGGER] Load config YAML + Fallback an toàn
# ═══════════════════════════════════════════════════════════════
@st.cache_data(ttl=300)
def load_mb_config():
    """Load cấu hình Multibagger từ config/multibagger.yaml"""
    default = {
        "scoring": {"min_fcf_p": 0.04, "min_roa": 0.08, "max_price_range_12m": 0.30, "min_bm_ratio": 0.40, "max_log_tev": 20.0},
        "polling": {"quote_interval": 3.0, "fundamentals_interval": 300.0},
        "cache": {"ttl_quote": 30, "ttl_fundamental": 3600, "max_entries": 500},
        "ui": {"default_watchlist": ["VCB","HPG","FPT","REE","DXG"], "show_rationale": True, "export_format": "csv"}
    }
    try:
        cfg_path = Path("config/multibagger.yaml")
        if cfg_path.exists():
            with open(cfg_path, "r", encoding="utf-8") as f:
                user_cfg = yaml.safe_load(f)
            return {**default, **user_cfg} if isinstance(user_cfg, dict) else default
    except Exception as e:
        st.warning(f"⚠️ Lỗi load config multibagger: {e}. Đang dùng mặc định.")
    return default

MB_CONFIG = load_mb_config()

def safe_df(df):
    """Ép toàn bộ DataFrame về kiểu str-safe để tránh PyArrow lỗi."""
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].apply(
                lambda v: f"{v:,.0f}" if isinstance(v, (float, int)) and not isinstance(v, bool)
                else (str(v) if v is not None and str(v) != 'nan' else "")
            )
    return out

def sig_badge(action):
    icon = SIGNAL_ICON.get(str(action).upper(), '')
    return f"{icon} {action}"

# ═══════════════════════════════════════════════════════════════
# SIDEBAR
# ═══════════════════════════════════════════════════════════════
with st.sidebar:
    st.markdown("## 🧠 Super AI Quant VN")
    st.caption("Level 6 · Renaissance + Two Sigma + Bridgewater")
    st.divider()
    st.markdown("### 📡 Nguồn & Thời gian")
    source = st.selectbox("Nguồn dữ liệu:", SOURCES)
    period = st.radio("Kỳ phân tích:", ['6T','1N','2N','3N','Tuỳ chọn'], index=1, horizontal=True)
    period_days = {'6T':180,'1N':365,'2N':730,'3N':1095}
    if period == 'Tuỳ chọn':
        start_date = st.date_input("Từ ngày:", value=datetime.today()-timedelta(days=365))
        start_str  = start_date.strftime('%Y-%m-%d')
    else:
        start_str = (datetime.today()-timedelta(days=period_days[period])).strftime('%Y-%m-%d')

    st.divider()
    st.markdown("### 📋 Universe")
    preset = st.radio("Danh sách:", ['VN30','Phổ biến','MidCap','Tuỳ chọn'], index=1)
    pools  = {'VN30':VN30,'Phổ biến':POPULAR,'MidCap':MIDCAP}
    pool   = pools.get(preset, VN30+POPULAR)
    default_syms = pools.get(preset, POPULAR)[:8] if preset != 'Tuỳ chọn' else ['VCB','TCB','HPG']

    selected = st.multiselect("Mã cổ phiếu:", options=pool, default=default_syms, max_selections=20)
    custom   = st.text_input("Thêm mã (phẩy):", placeholder="PNJ,DGC,KDH")
    if custom:
        selected = list(dict.fromkeys(selected + [s.strip().upper() for s in custom.split(',') if s.strip()]))

    st.divider()
    st.markdown("### 🤖 AI Agents")
    try:
        _cfg = load_config()
        all_agent_cfgs = _cfg['ai_agents'].get('agents', [])
        all_agent_names= [a['name'] for a in all_agent_cfgs]
        default_enabled= _cfg['ai_agents'].get('enabled', all_agent_names[:5])
    except Exception:
        all_agent_cfgs  = []
        all_agent_names = ['DeepSeek','ChatGPT','Claude','Groq-Llama','Qwen']
        default_enabled = all_agent_names

    preset_name = st.selectbox("Preset:", ['Custom'] + list(AGENT_PRESETS.keys()))
    if preset_name != 'Custom':
        sel_agent_cfgs = [{'name': a[0], 'provider': a[1], 'model': a[2]} for a in AGENT_PRESETS[preset_name]]
        sel_agents = [a['name'] for a in sel_agent_cfgs]
        st.caption(f"{len(sel_agents)} agents: {', '.join(sel_agents)}")
    else:
        sel_agents = st.multiselect("Chọn agents:", all_agent_names, default=[n for n in default_enabled if n in all_agent_names])
        sel_agent_cfgs = [a for a in all_agent_cfgs if a['name'] in sel_agents]

    with st.expander("➕ Thêm agent tuỳ chỉnh"):
        c_name     = st.text_input("Tên agent:", placeholder="Gemini-Pro")
        c_provider = st.selectbox("Provider:", ['openrouter','groq','openai','deepseek','anthropic','qwen'])
        c_model    = st.text_input("Model:", placeholder="openrouter/google/gemini-2.5-flash-preview")
        if st.button("Thêm") and c_name and c_model:
            new_cfg = {'name':c_name,'provider':c_provider,'model':c_model,'temperature':0.3}
            if new_cfg not in sel_agent_cfgs:
                sel_agent_cfgs.append(new_cfg)
                sel_agents.append(c_name)
                st.success(f"Đã thêm {c_name}")

    st.divider()
    st.markdown("### 🎯 Bộ lọc")
    min_score = st.slider("Score tối thiểu:", -1.0, 1.0, 0.0, 0.05)
    signal_filter = st.multiselect("Chỉ hiện tín hiệu:",
        ['STRONG_BUY','BREAKOUT','BUY','ACCUMULATION','HOLD','REDUCE','SELL'],
        default=['STRONG_BUY','BREAKOUT','BUY','ACCUMULATION'])
    wyckoff_filter = st.multiselect("Wyckoff Phase:",
        ['A','B','C','D','E','DISTRIBUTION','UNKNOWN'], default=[])

    st.divider()
    st.markdown("### 💼 Portfolio")
    port_value = st.number_input("Giá trị (triệu VND):", min_value=10, value=1000, step=100) * 1_000_000
    risk_pref  = st.select_slider("Khẩu vị rủi ro:", ['Thận trọng','Cân bằng','Tăng trưởng'], value='Cân bằng')
    risk_mult  = {'Thận trọng':0.6,'Cân bằng':1.0,'Tăng trưởng':1.4}[risk_pref]

    st.divider()
    st.markdown("### 📊 Biểu đồ")
    show_ma  = st.checkbox("MA20/MA50", value=True)
    show_bb  = st.checkbox("Bollinger Bands", value=True)
    show_vol = st.checkbox("Volume", value=True)
    show_macd= st.checkbox("MACD", value=True)

    st.caption(f"🕐 {datetime.now().strftime('%H:%M %d/%m/%Y')}")

# ═══════════════════════════════════════════════════════════════
# HEADER
# ═══════════════════════════════════════════════════════════════
st.markdown("# 🧠 Super AI Quant VN — Level 6")
st.caption(f"{source} · từ {start_str} · {len(selected)} mã · {len(sel_agents)} AI Agents · Portfolio {port_value/1e9:.1f}B VND")
if not selected:
    st.warning("Chưa chọn mã nào.")
    st.stop()

# ═══════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════
tab_market, tab_chart, tab_pattern, tab_ai, tab_portfolio, tab_report, tab_agent, tab_multibagger = st.tabs([
    "🌐 Thị trường", "📊 Biểu đồ", "🔍 Mẫu hình", "🤖 Phân tích AI",
    "💼 Portfolio", "📄 Báo cáo", "🧠 Auto Research", "🚀 Multibagger Agent"
])

# ─────────────────────────────────────────────────────────────────────────────
# TAB: THỊ TRƯỜNG — NÂNG CẤP (INDEX / FLOW / BREADTH / SECTOR / STRATEGY)
# ─────────────────────────────────────────────────────────────────────────────
with tab_market:
    col_btn1, col_btn2 = st.columns([1, 4])
    with col_btn1:
        refresh_market = st.button("🔄 Cập nhật", use_container_width=True)
        if refresh_market:
            with st.spinner("Đang tải dữ liệu thị trường..."):
                try:
                    f = DataFetcher(source=source, start_date=start_str)
                    ov = f.get_market_overview()
                    vi_df = f.get_index_ohlcv('VNINDEX')
                    regime_r = classify_market_regime(vi_df)
                    # sector rotation signal (fallback safe)
                    try:
                        _tmp_cfg = load_config()
                        sector_signals = get_sector_rotation_signal(_tmp_cfg.get('macro_context', {}))
                    except Exception:
                        sector_signals = {}
                    st.session_state.update({
                        'market_ov': ov,
                        'regime': regime_r,
                        'vnindex_df': vi_df,
                        'sector_signals': sector_signals
                    })
                    st.success("✅ Cập nhật thành công!")
                except Exception as e:
                    st.error(f"Lỗi: {e}")
                    import traceback
                    st.code(traceback.format_exc())

    # --- load from session (lazy) ---
    ov = st.session_state.get('market_ov', None)
    regime_state = st.session_state.get('regime', {})
    vnindex_df = st.session_state.get('vnindex_df', pd.DataFrame())
    sector_signals = st.session_state.get('sector_signals', {})

    # If missing, try a lightweight fetch (no crash)
    if ov is None:
        try:
            f_tmp = DataFetcher(source=source, start_date=start_str)
            ov = f_tmp.get_market_overview() or {}
            st.session_state['market_ov'] = ov
            if 'vnindex_df' not in st.session_state:
                try:
                    vi = f_tmp.get_index_ohlcv('VNINDEX')
                    st.session_state['vnindex_df'] = vi
                    st.session_state['regime'] = classify_market_regime(vi)
                    vnindex_df = vi
                    regime_state = st.session_state['regime']
                except Exception:
                    pass
        except Exception:
            ov = {}
            st.session_state['market_ov'] = ov

    # Normalize ov
    if isinstance(ov, pd.DataFrame):
        ov = {}

    # --- Top banner: regime + quick strategy summary ---
    col_banner_left, col_banner_right = st.columns([3, 1])
    with col_banner_left:
        if regime_state and isinstance(regime_state, dict):
            label = regime_state.get('label', 'Unknown')
            conf = regime_state.get('confidence', 0.0)
            strat = regime_state.get('strategy', 'N/A')
            color = {'BULL':'#22c55e','BULL_WEAK':'#86efac','SIDEWAY':'#f59e0b',
                     'BEAR':'#ef4444','CRASH_RISK':'#7f1d1d'}.get(regime_state.get('regime',''),'#6b7280')
            st.markdown(f"""<div style="background:{color}22;border-left:4px solid {color};
                padding:12px 16px;border-radius:8px;margin-bottom:8px">
                <b style="color:{color};font-size:1.15em">{label}</b> &nbsp;·&nbsp;
                Strategy: <b>{strat}</b> &nbsp;·&nbsp;
                Confidence: <b>{conf:.0%}</b>
            </div>""", unsafe_allow_html=True)
        else:
            st.info("Regime: Không có dữ liệu")

    with col_banner_right:
        # Quick action suggestion derived from regime + simple rules
        action = "HOLD"
        confidence = 0.35
        if regime_state.get('regime') == 'BULL':
            action = "BUY"
            confidence = max(confidence, regime_state.get('confidence', 0.4))
        elif regime_state.get('regime') in ('BEAR','CRASH_RISK'):
            action = "REDUCE"
            confidence = max(confidence, regime_state.get('confidence', 0.6))
        elif regime_state.get('regime') == 'SIDEWAY':
            action = "ACCUMULATION"
            confidence = max(confidence, regime_state.get('confidence', 0.3))
        st.markdown(f"**Gợi ý chiến lược:**  \n**{action}**  · Độ tin cậy **{confidence:.0%}**")

    st.markdown("---")

    # --- Row: Index cards + breadth + volatility ---
    c1, c2, c3, c4 = st.columns([1.2, 1, 1, 1])
    # VNINDEX card (sparkline + last price)
    try:
        vn_close = None
        if isinstance(vnindex_df, pd.DataFrame) and not vnindex_df.empty and 'close' in vnindex_df.columns:
            vn_close = vnindex_df['close'].astype(float).dropna()
            last = float(vn_close.iloc[-1])
            prev = float(vn_close.iloc[-2]) if len(vn_close) >= 2 else last
            pct = (last - prev) / prev * 100 if prev != 0 else 0.0
            c1.metric("VNINDEX", f"{last:,.2f}", f"{pct:+.2f}%")
            # sparkline
            spark = vn_close.tail(60).tolist()
            c1.line_chart(pd.Series(spark))
        else:
            c1.metric("VNINDEX", "N/A", "")
    except Exception:
        c1.metric("VNINDEX", "Error", "")

    # Market breadth (adv/dec) if available in ov
    try:
        breadth = ov.get('breadth') if isinstance(ov, dict) else None
        breadth = breadth if 'breadth' in dir() else None
        if isinstance(breadth, dict):
            adv = breadth.get('advancers', 0)
            dec = breadth.get('decliners', 0)
            c2.metric("Advancers", f"{adv}", "")
            c3.metric("Decliners", f"{dec}", "")
            # breadth ratio
            ratio = adv / (adv + dec) if (adv + dec) > 0 else 0
            c4.metric("Breadth Ratio", f"{ratio:.2%}", "")
        else:
            # fallback: compute simple breadth from top_gainers/top_losers if present
            tg = ov.get('top_gainers') if isinstance(ov, dict) else None
            tl = ov.get('top_losers') if isinstance(ov, dict) else None
            if isinstance(tg, pd.DataFrame) or isinstance(tl, pd.DataFrame):
                g = len(tg) if isinstance(tg, pd.DataFrame) else 0
                l = len(tl) if isinstance(tl, pd.DataFrame) else 0
                c2.metric("Advancers", f"{g}", "")
                c3.metric("Decliners", f"{l}", "")
                ratio = g / (g + l) if (g + l) > 0 else 0
                c4.metric("Breadth Ratio", f"{ratio:.2%}", "")
            else:
                c2.metric("Advancers", "N/A", "")
                c3.metric("Decliners", "N/A", "")
                c4.metric("Breadth Ratio", "N/A", "")
    except Exception:
        c2.metric("Advancers", "Error", "")
        c3.metric("Decliners", "Error", "")
        c4.metric("Breadth Ratio", "Error", "")

    st.markdown("---")

    # --- Row: Top movers + sector rotation + foreign flow heatmap ---
    left, mid, right = st.columns([1.6, 1, 1.2])
    
    # Top movers (gainers / losers)
    with left:
        st.markdown("#### 🔝 Top Movers")
        try:
            tg = ov.get('top_gainers') if isinstance(ov, dict) else None
            tl = ov.get('top_losers') if isinstance(ov, dict) else None
            if isinstance(tg, pd.DataFrame) and not tg.empty:
                tg_disp = tg.head(10).copy()
                st.dataframe(safe_df(tg_disp), use_container_width=True, hide_index=True)
            else:
                st.info("Không có dữ liệu Top Gainers")
            if isinstance(tl, pd.DataFrame) and not tl.empty:
                st.dataframe(safe_df(tl.head(10)), use_container_width=True, hide_index=True)
        except Exception as e:
            st.warning(f"Không hiển thị top movers: {e}")

    # Sector rotation signals
    with mid:
        st.markdown("#### 🏷️ Sector Rotation")
        try:
            if sector_signals:
                # Xử lý cả list (từ get_sector_rotation_signal) và dict
                if isinstance(sector_signals, list):
                    for item in sector_signals:
                        trigger = item.get('trigger', '')
                        sectors = item.get('sectors', [])
                        st.markdown(f"- 🟢 **{trigger}**: {', '.join(sectors[:3])}", unsafe_allow_html=True)
                elif isinstance(sector_signals, dict):
                    for s, sig in sector_signals.items():
                        lbl   = sig.get('signal', 'NEUTRAL') if isinstance(sig, dict) else str(sig)
                        score = sig.get('score', 0.0) if isinstance(sig, dict) else 0.0
                        color = '#22c55e' if lbl in ('OVERWEIGHT','ROTATE_IN') else ('#ef4444' if lbl in ('UNDERWEIGHT','ROTATE_OUT') else '#f59e0b')
                        st.markdown(f"- **{s}**: <span style='color:{color}'>{lbl}</span> · {score:.2f}", unsafe_allow_html=True)
            else:
                st.info("Không có tín hiệu sector rotation")
        except Exception as e:
            st.warning(f"Lỗi sector signals: {e}")

    # Foreign flow heatmap (top sectors or top symbols)
    with right:
        st.markdown("#### 🌊 Dòng tiền nước ngoài (tóm tắt)")
        try:
            ff_sample = None
            # try flow summary per symbol if available in ov
            # fallback: call DataFetcher lightweight for a few popular symbols
            if isinstance(ov, dict) and 'top_foreign_buy' in ov and isinstance(ov['top_foreign_buy'], pd.DataFrame):
                ff_sample = ov['top_foreign_buy'].head(10)
            else:
                # try to fetch for POPULAR list (light)
                ftmp = DataFetcher(source=source, start_date=start_str)
                rows = []
                for s in POPULAR[:8]:
                    try:
                        fs = ftmp.get_flow_summary(s)
                        rows.append({'symbol': s, 'foreign_net_30d': fs.get('foreign_net_30d', 0), 'prop_net_30d': fs.get('prop_net_30d', 0)})
                    except Exception:
                        rows.append({'symbol': s, 'foreign_net_30d': 0, 'prop_net_30d': 0})
                ff_sample = pd.DataFrame(rows).set_index('symbol')
            if isinstance(ff_sample, pd.DataFrame) and not ff_sample.empty:
                st.dataframe(safe_df(ff_sample), use_container_width=True, hide_index=False)
            else:
                st.info("Không có dữ liệu dòng tiền")
        except Exception as e:
            st.warning(f"Lỗi dòng tiền: {e}")

    st.markdown("---")

    # --- Analyst-style short commentary (auto-generated template) ---
    st.markdown("### 📝 Nhận định nhanh và chiến lược")
    try:
        # Build a concise commentary from regime + breadth + flows
        lines = []
        # regime summary
        if regime_state:
            lines.append(f"**Regime:** {regime_state.get('label','N/A')} (confidence {regime_state.get('confidence',0):.0%})")
        # breadth
        if isinstance(breadth, dict):
            lines.append(f"**Breadth:** Adv {breadth.get('advancers',0)} / Dec {breadth.get('decliners',0)}")
        # foreign flow
        if isinstance(ff_sample, pd.DataFrame) and 'foreign_net_30d' in ff_sample.columns:
            top_buy = ff_sample['foreign_net_30d'].nlargest(3).index.tolist()
            lines.append(f"**Foreign net (30d):** Top buys {', '.join(map(str, top_buy))}")
        # sector hint
        if sector_signals:
            if isinstance(sector_signals, list):
                overweight = [i.get('trigger','') for i in sector_signals]
                under = []
            else:
                overweight = [s for s, v in sector_signals.items() if isinstance(v,dict) and v.get('signal') in ('OVERWEIGHT','ROTATE_IN')]
                under = [s for s, v in sector_signals.items() if isinstance(v,dict) and v.get('signal') in ('UNDERWEIGHT','ROTATE_OUT')]
            if overweight:
                lines.append(f"**Sector pick:** Overweight {', '.join(overweight[:3])}")
            if under:
                lines.append(f"**Avoid:** {', '.join(under[:3])}")
        # final strategy sentence
        strategy_sentence = ""
        if regime_state.get('regime') == 'BULL':
            strategy_sentence = "Ưu tiên mua cổ phiếu có momentum và fundamentals; giữ tỷ lệ tiền mặt thấp."
        elif regime_state.get('regime') in ('BEAR','CRASH_RISK'):
            strategy_sentence = "Tăng tiền mặt, giảm tỷ trọng cổ phiếu rủi ro; ưu tiên phòng thủ và cổ phiếu chất lượng."
        elif regime_state.get('regime') == 'SIDEWAY':
            strategy_sentence = "Tập trung trading ngắn hạn, chọn cổ có thanh khoản và risk-reward rõ ràng."
        else:
            strategy_sentence = "Theo dõi thêm dữ liệu; không mở vị thế lớn."

        st.markdown("\n".join(lines))
        st.info(strategy_sentence)
    except Exception as e:
        st.warning(f"Không thể tạo nhận định tự động: {e}")

    # --- Debug / raw data (collapsible) ---
    with st.expander("🔧 Dữ liệu thô & Debug"):
        st.write("Market OV keys:", list(ov.keys()) if isinstance(ov, dict) else type(ov).__name__)
        st.write("Regime:", regime_state)
        st.write("Sector signals:", sector_signals)
        if isinstance(ov.get('top_gainers'), pd.DataFrame):
            st.write("Top gainers sample:", safe_df(ov['top_gainers'].head(5)))
    # ---------------------------
# Sector heatmap + Small multiples + Generate Full Report
# ---------------------------
try:
    # --- Sector rotation heatmap (seaborn) ---
    st.markdown("#### 🔁 Sector Rotation Heatmap")
    if sector_signals:
        # build DataFrame: rows = sectors, cols = metrics (score, signal->numeric)
        rows = []
        if isinstance(sector_signals, list):
            for item in sector_signals:
                rows.append({'sector': item.get('trigger',''), 'score': 1.0, 'signal_val': 1})
        elif isinstance(sector_signals, dict):
            for s, v in sector_signals.items():
                score = v.get('score', 0.0) if isinstance(v, dict) else 0.0
                sig   = v.get('signal', '')  if isinstance(v, dict) else ''
                sig_val = 1 if sig in ('OVERWEIGHT','ROTATE_IN') else (-1 if sig in ('UNDERWEIGHT','ROTATE_OUT') else 0)
                rows.append({'sector': s, 'score': score, 'signal_val': sig_val})
        df_sec = pd.DataFrame(rows).set_index('sector')
        if not df_sec.empty:
            # normalize score column for color scaling
            mat = df_sec[['score', 'signal_val']]
            fig_h, ax_h = plt.subplots(1, 1, figsize=(6, max(2, len(mat)*0.35)), dpi=90)
            sns.heatmap(mat, annot=True, fmt=".2f", cmap="vlag", center=0, cbar_kws={'shrink':0.6}, ax=ax_h)
            ax_h.set_title("Sector rotation (score, signal)")
            st.pyplot(fig_h, use_container_width=True)
            plt.close(fig_h)
        else:
            st.info("Không có dữ liệu sector rotation để vẽ heatmap.")
    else:
        st.info("Không có tín hiệu sector rotation để hiển thị heatmap.")

    # --- Small multiples: price series for top movers (if available) ---
    st.markdown("#### 📈 Small Multiples — Top Movers Price Series")
    top_symbols = []
    if isinstance(ov, dict) and isinstance(ov.get('top_gainers'), pd.DataFrame) and not ov['top_gainers'].empty:
        # try to extract symbol column heuristically
        tg = ov['top_gainers']
        sym_col = next((c for c in tg.columns if 'symbol' in c.lower() or 'code' in c.lower() or 'ticker' in c.lower()), None)
        if sym_col:
            top_symbols = tg[sym_col].astype(str).head(6).tolist()
    # fallback: use POPULAR
    if not top_symbols:
        top_symbols = POPULAR[:6]

    # fetch OHLC for these symbols (light)
    small_dfs = {}
    ftmp = None
    try:
        ftmp = DataFetcher(source=source, start_date=start_str)
    except Exception:
        ftmp = None

    for s in top_symbols:
        try:
            if ftmp:
                df_s = ftmp.get_ohlcv(s)
                if isinstance(df_s, pd.DataFrame) and not df_s.empty and 'close' in df_s.columns:
                    small_dfs[s] = df_s['close'].astype(float).tail(120)  # last ~120 bars
        except Exception:
            continue

    if small_dfs:
        n = len(small_dfs)
        cols = min(3, n)
        rows = (n + cols - 1) // cols
        fig_sm, axes_sm = plt.subplots(rows, cols, figsize=(4*cols, 2.2*rows), sharex=False, squeeze=False)
        axes_sm = axes_sm.flatten()
        for i, (sym, series) in enumerate(small_dfs.items()):
            ax = axes_sm[i]
            ax.plot(series.index, series.values, color='#60a5fa', lw=1.2)
            ax.set_title(sym, color='white', fontsize=9)
            ax.tick_params(colors='#9ca3af', labelsize=7)
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
            ax.set_facecolor('#0e1117')
            for spine in ax.spines.values():
                spine.set_color('#374151')
        # hide unused axes
        for j in range(i+1, len(axes_sm)):
            axes_sm[j].axis('off')
        plt.tight_layout(h_pad=0.6)
        st.pyplot(fig_sm, use_container_width=True)
        plt.close(fig_sm)
    else:
        st.info("Không có dữ liệu giá cho small multiples.")

    # --- Generate Full Report (AI agents) ---
    st.markdown("#### 🤖 Generate Full Report")
    gen_col1, gen_col2 = st.columns([1, 3])
    with gen_col1:
        gen_btn = st.button("Generate Full Report", key="gen_full_report", use_container_width=True)
    with gen_col2:
        st.caption("Tạo báo cáo chi tiết cho universe đã chọn (AI ensemble).")

    if gen_btn:
        with st.spinner("Đang tạo báo cáo bằng AI agents..."):
            try:
                # Build agent configs from sidebar selection (sel_agent_cfgs)
                agent_cfgs_local = sel_agent_cfgs if 'sel_agent_cfgs' in globals() else []
                if not agent_cfgs_local:
                    # fallback to default small set
                    agent_cfgs_local = [{'name':n,'provider':'openai','model':'gpt-4o-mini'} for n in ['ChatGPT','Claude']]

                agents = {}
                try:
                    agents = build_agents_from_config(agent_cfgs_local)
                except Exception:
                    # try build_agents_from_config may not be available; fallback empty
                    agents = {}

                # Build prompt: concise summary of selected symbols + regime + top movers
                prompt_lines = []
                prompt_lines.append(f"Generate a concise investment report for the following universe: {', '.join(selected)}.")
                if regime_state:
                    prompt_lines.append(f"Market regime: {regime_state.get('label','N/A')} (confidence {regime_state.get('confidence',0):.0%}).")
                # include top movers
                if isinstance(ov, dict) and isinstance(ov.get('top_gainers'), pd.DataFrame):
                    tg = ov['top_gainers'].head(5)
                    tg_list = tg.iloc[:,0].astype(str).tolist() if not tg.empty else []
                    if tg_list:
                        prompt_lines.append("Top gainers: " + ", ".join(tg_list))
                prompt_lines.append("Provide: 1) Executive summary (3-5 sentences). 2) Top 3 trade ideas with rationale and risk. 3) Portfolio allocation suggestion (cash %, sectors). Answer in JSON with keys: executive_summary, ideas, allocation.")
                prompt_text = "\n".join(prompt_lines)

                # Query each agent and aggregate
                decisions = {}
                for name, agent in agents.items():
                    try:
                        resp = agent.ask(SYSTEM_PROMPT, prompt_text)
                        dec = extract_json(resp) if resp else None
                        decisions[name] = dec if dec else {'error': 'no json', 'raw': resp}
                    except Exception as e:
                        decisions[name] = {'error': str(e)}

                # If no agents or all failed, fallback to simple heuristic report
                if not decisions:
                    fallback = {
                        "executive_summary": "No AI agents available; fallback heuristic: market uncertain, prefer selective accumulation.",
                        "ideas": [{"symbol": selected[0] if selected else "N/A", "action":"WATCH", "reason":"No agents available"}],
                        "allocation": {"cash_pct": 0.2, "sectors": []}
                    }
                    st.json(fallback)
                else:
                    # show raw agent outputs and a simple meta-ensemble if possible
                    st.markdown("**Agent outputs (raw / parsed)**")
                    for n, out in decisions.items():
                        st.markdown(f"**{n}**")
                        st.write(out)
                    # try meta-ensemble to produce a combined recommendation if meta_ensemble available
                    try:
                        # meta_ensemble expects decisions mapping -> (action, score)
                        # build simple mapping if agents returned structured 'action' and 'confidence'
                        ensemble_input = {}
                        for n, o in decisions.items():
                            if isinstance(o, dict) and 'action' in o:
                                ensemble_input[n] = {'action': o.get('action'), 'confidence': o.get('confidence', 0.0)}
                        if ensemble_input:
                            action, score = meta_ensemble(ensemble_input)
                            st.success(f"Ensemble suggestion: **{action}** (score {score:.2f})")
                    except Exception:
                        pass
            except Exception as e:
                st.error(f"Không thể tạo báo cáo: {e}")
except Exception as e:
    st.warning(f"Không thể hiển thị phần nâng cao thị trường: {e}")
 
# ═══════════════════════════════════════════════════════════════
# TAB: BIỂU ĐỒ
# ═══════════════════════════════════════════════════════════════
with tab_chart:
    col_l, col_r = st.columns([1, 3])
    with col_l:
        chart_sym  = st.selectbox("Mã:", selected)
        chart_type = st.radio("Loại:", ['Đầy đủ','Chỉ giá'], index=0)
    with col_r:
        with st.spinner(f"Tải {chart_sym}..."):
            try:
                fch = DataFetcher(source=source, start_date=start_str)
                ohlcv = fch.get_ohlcv(chart_sym)
                ff    = fch.get_foreign_flow(chart_sym)
                pp    = fch.get_proprietary_flow(chart_sym)
                feat  = build_features(ohlcv, ff, pp)

                if ohlcv is None or ohlcv.empty or 'close' not in ohlcv.columns:
                    st.error(f"Không có dữ liệu OHLCV cho {chart_sym}.")
                else:
                    last = float(ohlcv['close'].iloc[-1])
                    prev = float(ohlcv['close'].iloc[-2]) if len(ohlcv) >= 2 else last
                    chg  = (last - prev) / prev * 100 if prev != 0 else 0.0
                    vol  = float(ohlcv['volume'].iloc[-1]) if 'volume' in ohlcv.columns else 0.0

                    m1,m2,m3,m4 = st.columns(4)
                    m1.metric("Giá đóng cửa", f"{last:,.0f}", f"{chg:+.2f}%")
                    m2.metric("Cao nhất", f"{float(ohlcv['high'].iloc[-1]):,.0f}" if 'high' in ohlcv.columns else "N/A")
                    m3.metric("Thấp nhất", f"{float(ohlcv['low'].iloc[-1]):,.0f}" if 'low' in ohlcv.columns else "N/A")
                    m4.metric("Volume", f"{vol/1e6:.1f}M")

                    wy = detect_wyckoff_phase(feat)
                    af = compute_alpha_score(compute_alpha_factors(feat))
                    wy_col, af_col = st.columns(2)
                    wy_col.info(f"Wyckoff: **Phase {wy['phase']}** – {wy['description']} ({wy['confidence']:.0%})")
                    af_col.info(f"Alpha composite: **{af['composite']:+.2f}** | SM: {af['smart_money']:+.2f} | Vol: {af['volume']:+.2f}")

                    n_panels = (1 if chart_type == 'Chỉ giá' else 0) + (1 if show_macd else 0) + (1 if show_vol else 0) + 1
                    heights  = [3] + [1]*(n_panels-1)
                    fig, axes = plt.subplots(n_panels, 1, figsize=(14, 3*n_panels+1),
                                             gridspec_kw={'height_ratios': heights}, facecolor='#0e1117')
                    if n_panels == 1: axes = [axes]
                    ax_idx = 0
                    ax_p   = axes[ax_idx]; ax_idx += 1

                    for ax in axes:
                        ax.set_facecolor('#0e1117')
                        ax.tick_params(colors='#9ca3af', labelsize=8)
                        ax.spines[:].set_color('#374151')

                    close = feat['close'].astype(float)
                    idx   = feat.index

                    ax_p.plot(idx, close, color='#60a5fa', lw=1.3, label='Giá')
                    if show_ma and chart_type != 'Chỉ giá':
                        if 'ma20' in feat: ax_p.plot(idx, feat['ma20'], '#fbbf24', lw=0.9, label='MA20', alpha=0.85)
                        if 'ma50' in feat: ax_p.plot(idx, feat['ma50'], '#f472b6', lw=0.9, label='MA50', alpha=0.85)
                    if show_bb and chart_type != 'Chỉ giá' and 'bb_upper' in feat:
                        ax_p.fill_between(idx, feat['bb_lower'], feat['bb_upper'], alpha=0.07, color='#6ee7b7')
                        ax_p.plot(idx, feat['bb_upper'], '#6ee7b7', lw=0.7, ls='--', alpha=0.6)
                        ax_p.plot(idx, feat['bb_lower'], '#6ee7b7', lw=0.7, ls='--', alpha=0.6)

                    ax_p.set_title(f"{chart_sym}", color='white', fontsize=13, pad=6)
                    ax_p.legend(loc='upper left', fontsize=7, facecolor='#1f2937', labelcolor='white', framealpha=0.7)
                    ax_p.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x:,.0f}'))
                    ax_p.annotate(f'{last:,.0f}', xy=(idx[-1],last), xytext=(8,0),
                                  textcoords='offset points', color='#60a5fa', fontsize=8, va='center',
                                  bbox=dict(boxstyle='round,pad=0.2', facecolor='#1e3a5f', edgecolor='none'))

                    if show_macd and 'macd' in feat and ax_idx < n_panels:
                        ax_m = axes[ax_idx]; ax_idx += 1
                        hist = feat['macd_hist']
                        ax_m.bar(idx, hist, color=['#22c55e' if v >=0 else '#ef4444' for v in hist], width=1, alpha=0.7)
                        ax_m.plot(idx, feat['macd'], '#60a5fa', lw=0.8)
                        ax_m.plot(idx, feat['macd_signal'], '#fbbf24', lw=0.8)
                        ax_m.axhline(0, color='#4b5563', lw=0.5)
                        ax_m.set_ylabel('MACD', color='#9ca3af', fontsize=7)
                        ax_m.set_facecolor('#0e1117')
                        ax_m.tick_params(colors='#9ca3af', labelsize=7)
                        ax_m.spines[:].set_color('#374151')

                    if show_vol and 'volume' in feat and ax_idx < n_panels:
                        ax_v = axes[ax_idx]; ax_idx += 1
                        vol_s = feat['volume'].astype(float)
                        vc = ['#22c55e' if i==0 or close.iloc[i] >= close.iloc[i-1] else '#ef4444' for i in range(len(close))]
                        ax_v.bar(idx, vol_s, color=vc, width=1, alpha=0.75)
                        if 'vol_ma20' in feat: ax_v.plot(idx, feat['vol_ma20'], '#fbbf24', lw=0.8)
                        ax_v.yaxis.set_major_formatter(plt.FuncFormatter(lambda x,_: f'{x/1e6:.0f}M'))
                        ax_v.set_ylabel('Volume', color='#9ca3af', fontsize=7)
                        ax_v.set_facecolor('#0e1117')
                        ax_v.tick_params(colors='#9ca3af', labelsize=7)
                        ax_v.spines[:].set_color('#374151')

                    for ax in axes[:-1]: ax.tick_params(labelbottom=False)
                    axes[-1].xaxis.set_major_formatter(mdates.DateFormatter('%m/%Y'))
                    axes[-1].xaxis.set_major_locator(mdates.MonthLocator(interval=2))
                    plt.setp(axes[-1].xaxis.get_majorticklabels(), rotation=30, ha='right')
                    plt.tight_layout(h_pad=0.3)
                    st.pyplot(fig, use_container_width=True)
                    plt.close(fig)

            except Exception as e:
                st.error(f"Lỗi tải {chart_sym}: {e}")

    st.divider()
    st.markdown("#### 📋 So sánh hiệu suất")
    with st.spinner("Tải dữ liệu so sánh..."):
        rows = []
        fperf = DataFetcher(source=source, start_date=start_str)
        for sym in selected:
            try:
                df = fperf.get_ohlcv(sym)
                if df is None or df.empty or 'close' not in df.columns:
                    rows.append({'Mã':sym,'Giá':'N/A','1N':'-','1T':'-','1M':'-','3M':'-','Vol(M)':'-'})
                    continue
                c  = df['close'].astype(float)
                last_p = float(c.iloc[-1])
                rows.append({
                    'Mã': sym, 'Giá': f"{last_p:,.0f}",
                    '1N':  f"{c.pct_change(1).iloc[-1]:+.1%}" if len(c) >= 2 else '-',
                    '1T':  f"{c.pct_change(5).iloc[-1]:+.1%}"  if len(c) >=5  else '-',
                    '1M':  f"{c.pct_change(22).iloc[-1]:+.1%}" if len(c) >=22 else '-',
                    '3M':  f"{c.pct_change(66).iloc[-1]:+.1%}" if len(c) >=66 else '-',
                    'Vol(M)': f"{float(df['volume'].iloc[-1])/1e6:.1f}" if 'volume' in df.columns else '-',
                })
            except Exception:
                rows.append({'Mã':sym,'Giá':'N/A','1N':'-','1T':'-','1M':'-','3M':'-','Vol(M)':'-'})

        def cpct(v):
            s = str(v)
            if '+' in s: return 'color:#22c55e'
            if '-' in s and s!='-': return 'color:#ef4444'
            return ''
        df_perf = pd.DataFrame(rows)
        st.dataframe(safe_df(df_perf).style.map(cpct, subset=['1N','1T','1M','3M']),
                     use_container_width=True, hide_index=True)

# ═══════════════════════════════════════════════════════════════
# TAB: MẪU HÌNH
# ═══════════════════════════════════════════════════════
with tab_pattern:
    col_pl, col_pr = st.columns([1, 3])
    with col_pl:
        pat_sym    = st.selectbox("Mã phân tích:", selected, key='pat_sym')
        pat_period = st.radio("Kỳ dữ liệu:", ['3T','6T','1N','2N'], index=2, horizontal=True, key='pat_period')
        pat_days   = {'3T':90,'6T':180,'1N':365,'2N':730}[pat_period]
        pat_start  = (datetime.today()-timedelta(days=pat_days)).strftime('%Y-%m-%d')

        st.markdown("##### Bộ lọc mẫu hình")
        show_classic  = st.checkbox("Classic (H&S, Double Top/Bottom, Cup)", value=True)
        show_cont     = st.checkbox("Continuation (Triangle, Wedge, Flag, Channel)", value=True)
        show_candle   = st.checkbox("Candlestick", value=True)
        show_sm       = st.checkbox("Smart Money (Spring, Upthrust, Box)", value=True)
        min_conf      = st.slider("Confidence tối thiểu:", 0.5, 0.95, 0.60, 0.05, key='pat_conf')
        run_pat_btn   = st.button("🔍 Quét mẫu hình", use_container_width=True, key='pat_btn')

    with col_pr:
        if run_pat_btn:
            with st.spinner(f"Đang quét mẫu hình {pat_sym}..."):
                try:
                    fp  = DataFetcher(source=source, start_date=pat_start)
                    ohlcv_p = fp.get_ohlcv(pat_sym)
                    ff_p    = fp.get_foreign_flow(pat_sym)
                    pp_p    = fp.get_proprietary_flow(pat_sym)
                    feat_p  = build_features(ohlcv_p, ff_p, pp_p)
                    patterns= scan_all_patterns(ohlcv_p)
                    st.session_state[f'patterns_{pat_sym}'] = patterns
                    st.session_state[f'ohlcv_{pat_sym}']   = ohlcv_p
                except Exception as e:
                    st.error(f"Lỗi: {e}")

        pat_key = f'patterns_{pat_sym}'
        if pat_key in st.session_state:
            patterns = st.session_state[pat_key]
            ohlcv_p  = st.session_state[f'ohlcv_{pat_sym}']
            summary  = patterns.get('summary', {})

            bias_color = {'BULLISH':'#22c55e','BEARISH':'#ef4444','NEUTRAL':'#f59e0b'}.get(
                summary.get('bias','NEUTRAL'), '#6b7280')
            st.markdown(f"""<div style="background:{bias_color}22;border-left:4px solid {bias_color};
                padding:10px 14px;border-radius:8px;margin-bottom:12px">
                <b style="color:{bias_color};font-size:1.1em">{summary.get('bias','NEUTRAL')} — Strength: {summary.get('bias_strength',0):.0%}</b>
                &nbsp;· &nbsp; {summary.get('total_patterns',0)} mẫu hình phát hiện
            </div>""", unsafe_allow_html=True)

            sm1,sm2,sm3,sm4 = st.columns(4)
            sm1.metric("Tổng mẫu hình", summary.get('total_patterns',0))
            sm2.metric("🟢 Bullish", summary.get('bullish_count',0), f"Score: {summary.get('buy_score',0):.2f}")
            sm3.metric("🔴 Bearish", summary.get('bearish_count',0), f"Score: {summary.get('sell_score',0):.2f}")
            sm4.metric("Top Pattern", summary.get('top_patterns',[{}])[0].get('pattern', '-') if summary.get('top_patterns') else '-')

            st.markdown("#### 📊 Biểu đồ mẫu hình")
            try:
                fig_p = plot_patterns(ohlcv_p, patterns, symbol=pat_sym, figsize=(14,7))
                st.pyplot(fig_p, use_container_width=True)
                import matplotlib.pyplot as plt; plt.close(fig_p)
            except Exception as e:
                st.warning(f"Không vẽ được biểu đồ: {e}")

            all_pats_list = []
            if show_classic:  all_pats_list += patterns.get('classic', [])
            if show_cont:     all_pats_list += patterns.get('continuation', [])
            if show_candle:   all_pats_list += patterns.get('candlestick', [])
            if show_sm:       all_pats_list += patterns.get('smart_money', [])
            all_pats_list = [p for p in all_pats_list if p.get('confidence',0) >= min_conf]

            if all_pats_list:
                st.markdown("#### 📋 Chi tiết mẫu hình")
                pat_rows = []
                for p in sorted(all_pats_list, key=lambda x: x.get('confidence',0), reverse=True):
                    icon = {'bullish':'🟢','bearish':'🔴','neutral':'🟡'}.get(p.get('type','neutral'),'⚪')
                    tgt_u = p.get('target_up','')
                    tgt_d = p.get('target_down','')
                    pat_rows.append({
                        'Mẫu hình':   f"{icon} {p.get('pattern', p.get('name',''))}",
                        'Loại':       p.get('type','').upper(),
                        'Tín hiệu':   p.get('signal',''),
                        'Confidence': f"{p.get('confidence',0):.0%}",
                        'Mô tả':      p.get('description', p.get('desc','')),
                        'Target ↑':   f"{tgt_u:,.0f}" if isinstance(tgt_u, (int,float)) else str(tgt_u or ''),
                        'Target ↓':   f"{tgt_d:,.0f}" if isinstance(tgt_d, (int,float)) else str(tgt_d or ''),
                    })
                def cpat(v):
                    v = str(v)
                    if 'BULLISH' in v or 'BUY' in v or 'STRONG' in v: return 'color:#22c55e'
                    if 'BEARISH' in v or 'SELL' in v: return 'color:#ef4444'
                    if 'NEUTRAL' in v: return 'color:#f59e0b'
                    return ''
                df_pat = pd.DataFrame(pat_rows)
                st.dataframe(df_pat.style.map(cpat, subset=['Loại','Tín hiệu']),
                             use_container_width=True, hide_index=True)

                st.markdown("#### 🏆 Top mẫu hình quan trọng nhất")
                for i, p in enumerate(sorted(all_pats_list, key=lambda x: x.get('confidence',0), reverse=True)[:3]):
                    col_i = {'bullish':'#22c55e','bearish':'#ef4444','neutral':'#f59e0b'}.get(p.get('type','neutral'), '#6b7280')
                    with st.expander(f"{'🥇'if i==0 else'🥈'if i==1 else'🥉'} {p.get('pattern',p.get('name',''))} — {p.get('confidence',0):.0%}"):
                        st.markdown(f"**Loại:** {p.get('type','').upper()} | **Tín hiệu:** {p.get('signal','')}")
                        st.info(p.get('description', p.get('desc','')))
                        tgt_up   = p.get('target_up')
                        tgt_down = p.get('target_down')
                        if tgt_up:   st.success(f"Target tăng: **{tgt_up:,.0f}**")
                        if tgt_down: st.error(f"Target giảm: **{tgt_down:,.0f}**")
            else:
                st.info(f"Không tìm thấy mẫu hình với confidence >= {min_conf:.0%}")
        else:
            st.info("Nhấn **Quét mẫu hình** để bắt đầu phân tích.")

# ═══════════════════════════════════════════════════════════════
# TAB: PHÂN TÍCH AI
# ═══════════════════════════════════════════════════════════════
with tab_ai:
    run_btn = st.button("🚀 Chạy phân tích AI Level 6", type="primary", use_container_width=True)
    if run_btn:
        if not sel_agents:
            st.warning("Chưa chọn AI agent nào.")
        else:
            cfg = load_config()
            cfg['universe']['symbols']   = selected
            cfg['ai_agents']['enabled']  = sel_agents
            cfg['vnstock']['source']     = source
            cfg['vnstock']['start_date'] = start_str

            with st.spinner(f"Đang phân tích {len(selected)} mã qua {len(sel_agents)} AI Agents..."):
                try:
                    tmp = tempfile.NamedTemporaryFile(mode='w', suffix='.yaml', delete=False, encoding='utf-8')
                    import yaml; yaml.dump(cfg, tmp, allow_unicode=True); tmp.close()
                    full_result = run_full_pipeline(config_path=tmp.name)
                    os.unlink(tmp.name)
                    st.session_state['full_result'] = full_result
                    st.session_state['run_time'] = datetime.now().strftime('%H:%M %d/%m/%Y')
                    st.success(f"✅ Hoàn tất {len(full_result['symbols'])} mã lúc {st.session_state['run_time']}")
                except Exception as e:
                    st.error(f"Lỗi pipeline: {e}")
                    import traceback; st.code(traceback.format_exc())
                    st.stop()

    if 'full_result' in st.session_state:
        res   = st.session_state['full_result']
        syms  = res['symbols']
        regime_r = res.get('regime', {})

        filtered = {
            sym: data for sym, data in syms.items()
            if data.get('score', 0) >= min_score
            and (not signal_filter or data.get('action','HOLD') in signal_filter)
            and (not wyckoff_filter or data.get('wyckoff',{}).get('phase','') in wyckoff_filter)
        }

        action_groups = {}
        for sym, data in filtered.items():
            a = data.get('action','HOLD')
            action_groups.setdefault(a, []).append(sym)

        cols_m = st.columns(5)
        for i, action in enumerate(['STRONG_BUY','BREAKOUT','BUY','HOLD','SELL']):
            syms_a = action_groups.get(action, [])
            cols_m[i].metric(
                sig_badge(action),
                len(syms_a),
                ', '.join(syms_a[:3]) + ('...' if len(syms_a) >3 else '') or '-'
            )

        st.divider()
        st.markdown(f"#### 📊 Tín hiệu ({len(filtered)} mã sau lọc)")
        rows = []
        for sym, data in sorted(filtered.items(), key=lambda x: x[1].get('score',0), reverse=True):
            wy = data.get('wyckoff', {})
            af = data.get('alpha_scores', {})
            pos= data.get('position', {})
            row = {
                'Mã':          sym,
                'Tín hiệu':    sig_badge(data.get('action','HOLD')),
                'Score':       f"{data.get('score',0):+.3f}",
                'Wyckoff':     f"Phase {wy.get('phase','?')} ({wy.get('confidence',0):.0%})",
                'Momentum':    f"{af.get('momentum',0):+.2f}",
                'Smart Money': f"{af.get('smart_money',0):+.2f}",
                'Volume':      f"{af.get('volume',0):+.2f}",
                'Tier':        pos.get('tier_desc','-'),
                'Size %':      f"{pos.get('size_pct',0)*100:.1f}%",
                'Stop Loss':   f"{pos.get('stop_loss','N/A'):,.0f}" if pos.get('stop_loss') else 'N/A',
            }
            for ag in sel_agents:
                dec = data.get('details',{}).get(ag,{})
                a   = dec.get('action','-')
                c   = dec.get('confidence',0)
                row[ag] = f"{SIGNAL_ICON.get(a.upper(),'')} {a} {float(c):.0%}" if a!='-' else '-'
            rows.append(row)

        def csig(v):
            v = str(v)
            if any(x in v for x in ['STRONG','BREAKOUT']): return 'color:#22c55e;font-weight:bold'
            if 'BUY' in v: return 'color:#34d399'
            if 'ACCUM' in v: return 'color:#60a5fa'
            if 'SELL' in v: return 'color:#ef4444;font-weight:bold'
            if 'REDUCE' in v : return 'color:#fb923c'
            return ''

        df_sig = pd.DataFrame(rows)
        hl_cols = ['Tín hiệu'] + sel_agents
        hl_cols = [c for c in hl_cols if c in df_sig.columns]
        st.dataframe(df_sig.style.map(csig, subset=hl_cols),
                     use_container_width=True, hide_index=True, height=400)

        st.divider()
        st.markdown("#### 🔍 Chi tiết phân tích")
        sel_detail = st.selectbox("Chọn mã:", list(filtered.keys()) or list(syms.keys()))
        if sel_detail and sel_detail in syms:
            data = syms[sel_detail]
            wy   = data.get('wyckoff', {})
            af   = data.get('alpha_scores', {})
            sig  = data.get('engine_signal', {})
            pos  = data.get('position', {})
            fund = data.get('fundamental', {})
            flow = data.get('flow_summary', {})

            col_a, col_b = st.columns([1, 2])
            with col_a:
                st.metric(f"{sel_detail}", sig_badge(data.get('action','HOLD')), f"Score: {data.get('score',0):+.3f}")
                st.markdown("**Wyckoff Analysis**")
                phase_c = {'A':'#ef4444','B':'#f59e0b','C':'#fbbf24','D':'#22c55e','E':'#10b981'}.get(wy.get('phase','?'),'#6b7280')
                st.markdown(f"<span style='color:{phase_c};font-size:1.3em;font-weight:bold'>Phase {wy.get('phase','?')}</span> — {wy.get('description','')} ({wy.get('confidence',0):.0%})", unsafe_allow_html=True)
                if wy.get('is_buy_zone'): st.success("✅ VÙNG MUA – Wyckoff B/C/D")
                if wy.get('is_sell_zone'): st.error("⚠️ PHÂN PHỐI – Xem xét bán")

                st.markdown("**Alpha Factors**")
                af_df = pd.DataFrame([{'Factor': k.replace('_',' ').title(), 'Score': f"{v:+.3f}", 'Bar': '█' * int(abs(v)*10) if v == v else ''} for k,v in af.items()])
                st.dataframe(af_df, use_container_width=True, hide_index=True) 

                if pos:
                    st.markdown("**Position Sizing**")
                    st.info(f"**{pos.get('tier_desc','-')}**\n\nSize: {pos.get('size_pct',0)*100:.1f}% | {pos.get('size_vnd',0)/1e6:.0f}M VND\n\nStop Loss: {pos.get('stop_loss','N/A')}\n\nRisk: {pos.get('risk_vnd',0)/1e6:.1f}M VND")

                if fund:
                    st.markdown("**Chỉ số cơ bản**")
                    st.dataframe(pd.DataFrame([fund]).T.rename(columns={0:'Giá trị'}), use_container_width=True)

                if flow:
                    st.markdown("**Dòng tiền 30 ngày**")
                    ff_net = flow.get('foreign_net_30d', 0)
                    pp_net = flow.get('prop_net_30d', 0)
                    if ff_net: st.metric("Ngoại net", f"{ff_net/1e9:.2f}B", delta="Mua ròng" if ff_net >0 else "Bán ròng")
                    if pp_net: st.metric("Tự doanh net", f"{pp_net/1e9:.2f}B", delta="Mua ròng" if pp_net >0 else "Bán ròng")

            with col_b:
                st.markdown("**Kỹ thuật**")
                st.code(data.get('tech_summary',''), language=None)
                st.markdown("**Signal Engine**")
                st.text(format_signal_report(sig) if sig else 'N/A')
                st.markdown("**AI Agents**")
                ad = []
                for name, dec in data.get('details',{}).items():
                    ad.append({'Agent': name, 'Quyết định': sig_badge(dec.get('action','-')),
                               'Confidence': f"{float(dec.get('confidence',0)):.0%}",
                               'Wyckoff Phase': dec.get('wyckoff_phase','-'), 'Target': dec.get('target_price','-'),
                               'Stop Loss': dec.get('stop_loss','-'), 'Key Risk': dec.get('key_risk','-'), 'Lý do': dec.get('reasoning','-')})
                st.dataframe(pd.DataFrame(ad), use_container_width=True, hide_index=True)

            with st.expander("🔗 Causal Discovery"): st.text(data.get('causal_report',''))

            # ── ERS + VP + SMC + DCF (từ pipeline Level 6) ─────────────
            ers = data.get('ers', {})
            vp  = data.get('vp', {})
            smc = data.get('smc', {})
            dcf = data.get('dcf', {})
            comps = data.get('comps', {})

            if ers and ers.get('ers_score', 0) > 0:
                with st.expander(f"⚠️ Event Risk Score: {ers.get('ers_score',0)}/30 {ers.get('color','')}"):
                    st.markdown(f"**Action:** {ers.get('action','')}")
                    st.markdown(f"**Max Position:** {ers.get('max_position',0):.0%}  |  **Stop Loss:** {(ers.get('stoploss') or 0):.0%}")
                    sc = ers.get('scenario', {})
                    if sc:
                        e1,e2,e3 = st.columns(3)
                        e1.metric("🟢 Bull", f"{sc.get('bull_case',0):.0%}")
                        e2.metric("🟡 Base", f"{sc.get('base_case',0):.0%}")
                        e3.metric("🔴 Bear", f"{sc.get('bear_case',0):.0%}")
                    st.text(data.get('ers_report',''))

            if vp and vp.get('poc'):
                with st.expander(f"📊 Volume Profile — POC: {vp.get('poc',0):,.0f}"):
                    v1,v2,v3 = st.columns(3)
                    v1.metric("POC", f"{vp.get('poc',0):,.0f}", help="Point of Control")
                    v2.metric("VAH", f"{vp.get('vah',0):,.0f}", help="Value Area High")
                    v3.metric("VAL", f"{vp.get('val',0):,.0f}", help="Value Area Low")
                    st.metric("Bias", vp.get('bias',''))
                    st.text(data.get('vp_report',''))

            if smc and smc.get('passed', 0) > 0:
                with st.expander(f"🧠 Smart Money: {smc.get('passed',0)}/4 — {smc.get('confidence','')}"):
                    for k, v in smc.get('checks', {}).items():
                        st.markdown(f"{'✅' if v else '❌'} {k.replace('_',' ').title()}")
                    st.text(data.get('smc_report',''))

            if dcf and dcf.get('intrinsic_per_share'):
                with st.expander(f"💰 DCF Valuation — Intrinsic: {dcf.get('intrinsic_per_share',0):,.0f} VND"):
                    d1,d2,d3 = st.columns(3)
                    d1.metric("Intrinsic Value", f"{dcf.get('intrinsic_per_share',0):,.0f}")
                    d2.metric("WACC", f"{dcf.get('wacc',0):.1%}")
                    d3.metric("Growth", f"{dcf.get('growth_stage1',0):.1%}")
                    if comps and comps.get('verdict'):
                        st.markdown(f"**Comps Verdict:** {comps.get('verdict','')}  |  PE Premium: {comps.get('pe_premium',0):+.1f}%")
                    st.text(data.get('dcf_report',''))

            if data.get('ic_memo'):
                with st.expander("📋 IC Memo đầy đủ"):
                    st.text(data.get('ic_memo',''))

            with st.expander("📄 Raw JSON"): st.json(data.get('details',{}))

# ═══════════════════════════════════════════════════════════════
# TAB: PORTFOLIO
# ═══════════════════════════════════════════════════════════════
with tab_portfolio:
    if 'full_result' not in st.session_state:
        st.info("Chạy Phân tích AI trước.")
    else:
        res      = st.session_state['full_result']
        syms     = res['symbols']
        regime_r = res.get('regime',{})
        port     = res.get('portfolio',{})
        regime   = regime_r.get('regime','SIDEWAY')
        alloc    = REGIME_ALLOCATION.get(regime, {'equity':0.5,'cash':0.5})
        eq_pct   = min(alloc['equity'] * risk_mult, 0.95)
        cash_pct = 1 - eq_pct

        c1,c2,c3,c4 = st.columns(4)
        c1.metric("Regime", regime_r.get('label',''))
        c2.metric("Tỷ trọng CP", f"{eq_pct:.0%}", f"Risk mult: {risk_mult:.1f}x")
        c3.metric("Cash", f"{cash_pct:.0%}")
        c4.metric("Portfolio", f"{port_value/1e9:.2f}B VND")

        st.divider()
        signal_map = {sym: data.get('engine_signal',{}) for sym, data in syms.items()}
        alloc_df, _ = compute_portfolio_allocation(signal_map, regime_r, portfolio_value=port_value)
        if not alloc_df.empty:
            st.markdown("#### 📊 Phân bổ tối ưu")
            def calloc(v):
                v = str(v)
                if 'STRONG' in v or 'BREAK' in v: return 'color:#22c55e;font-weight:bold'
                if 'BUY' in v: return 'color:#34d399'
                if 'ACCUM' in v: return 'color:#60a5fa'
                return ''
            st.dataframe(safe_df(alloc_df).style.map(calloc, subset=['Tín hiệu']), use_container_width=True, hide_index=True)

        st.divider()
        st.markdown("#### 📉 Risk Metrics Portfolio")
        risk = port.get('risk', {})
        if risk:
            r1,r2,r3,r4 = st.columns(4)
            r1.metric("Annual Return", f"{risk.get('annual_return',0):+.1f}%")
            r2.metric("Annual Vol", f"{risk.get('annual_vol',0):.1f}%")
            r3.metric("Sharpe Ratio", f"{risk.get('sharpe_ratio',0):.2f}")
            r4.metric("Max Drawdown", f"{risk.get('max_drawdown',0):.1f}%")
            st.metric("Avg Correlation", f"{risk.get('avg_correlation',0):.3f}", help="Tương quan trung bình giữa các mã. Thấp = đa dạng hóa tốt")

        st.divider()
        st.markdown("#### 🔥 Entry / Exit Strategy")
        st.info("Entry (MUA): Phase B/C + Volume dry-up + Smart money accumulation → ACCUMULATION → chờ Break Phase D\nAdd Position: Breakout confirmed + Volume spike > 1.5x → STRONG BUY\nExit (BÁN): Wyckoff Distribution + Volume climax + Macro deterioration → SELL\nStop Loss: 2× ATR dưới giá entry")

# ═══════════════════════════════════════════════════════════════
# TAB: BÁO CÁO
# ═══════════════════════════════════════════════════════════════
with tab_report:
    if 'full_result' not in st.session_state:
        st.info("Chạy Phân tích AI trước để xuất báo cáo.")
    else:
        res      = st.session_state['full_result']
        syms     = res['symbols']
        regime_r = res.get('regime',{})
        run_time = st.session_state.get('run_time','')
        st.markdown(f"#### 📄 Báo cáo Super AI Quant VN — {run_time}")

        col_opt1, col_opt2, col_opt3 = st.columns(3)
        with col_opt1:
            report_syms = st.multiselect("Chọn mã đưa vào báo cáo:", list(syms.keys()), default=list(syms.keys()))
        with col_opt2:
            include_causal   = st.checkbox("Bao gồm Causal Discovery", value=True)
            include_fund     = st.checkbox("Bao gồm Cơ bản", value=True)
        with col_opt3:
            include_tech     = st.checkbox("Bao gồm Kỹ thuật", value=True)
            include_ai_raw   = st.checkbox("Bao gồm AI raw JSON", value=False)

        st.divider()
        def build_text_report():
            lines = ["="*70, "  SUPER AI QUANT VN — BÁO CÁO PHÂN TÍCH", f"  Ngày: {run_time}  |  Nguồn: {source}  |  Từ: {start_str}", "="*70, " ", format_regime_report(regime_r), " ", f"Causal Discovery:\n{res.get('causal_report','')} ", " ", "="*70, "  CHI TIẾT TỪNG MÃ", "="*70]
            for sym in report_syms:
                if sym not in syms: continue
                data = syms[sym]
                wy, sig, af, pos, fund = data.get('wyckoff',{}), data.get('engine_signal',{}), data.get('alpha_scores',{}), data.get('position',{}), data.get('fundamental',{})
                lines += [" ", f"── {sym} {'─'*50}", f"Tín hiệu:  {data.get('action','HOLD')}  |  Score: {data.get('score',0):+.3f}", f"Wyckoff:   Phase {wy.get('phase','?')} — {wy.get('description','')} ({wy.get('confidence',0):.0%})", f"Alpha:     Momentum={af.get('momentum',0):+.2f}  SM={af.get('smart_money',0):+.2f}  Vol={af.get('volume',0):+.2f}"]
                if pos: lines.append(f"Position:  {pos.get('tier_desc','-')} | Size {pos.get('size_pct',0)*100:.1f}% | Stop {pos.get('stop_loss','N/A')}")
                if include_tech: lines += [" ", "Kỹ thuật: ", data.get('tech_summary','')]
                if sig: lines += [" ", format_signal_report(sig)]
                if include_fund and fund: lines += [" ", f"Cơ bản: {json.dumps(fund, ensure_ascii=False)}"]
                lines += [" ", "AI Agents: "]
                for name, dec in data.get('details',{}).items(): lines.append(f"  [{name}] {dec.get('action','-')} ({dec.get('confidence',0):.0%}) — {dec.get('reasoning','-')}")
                if include_ai_raw: lines += [" ", "Raw JSON: ", json.dumps(data.get('details',{}), ensure_ascii=False, indent=2)]
            lines += [" ", "="*70, "  CÔNG THỨC: MACRO + SMART MONEY + LIQUIDITY + WYCKOFF + BREAKOUT", "  THỊ TRƯỜNG = DÒNG TIỀN + THANH KHOẢN + TÂM LÝ + VĨ MÔ", "="*70]
            return "\n".join(lines)

        report_text = build_text_report()
        with st.expander("👁 Preview báo cáo", expanded=True):
            st.text(report_text[:3000] + ("\n...(còn nữa)" if len(report_text) >3000 else ""))

        st.divider()
        col_dl1, col_dl2, col_dl3 = st.columns(3)
        with col_dl1:
            st.download_button("📄 Xuất TXT", data=report_text.encode('utf-8'), file_name=f"SuperAI_Quant_{datetime.now().strftime('%Y%m%d_%H%M')}.txt", mime="text/plain", use_container_width=True)
        with col_dl2:
            try:
                excel_rows = []
                for sym in report_syms:
                    if sym not in syms: continue
                    data = syms[sym]
                    wy, af, pos, fund = data.get('wyckoff',{}), data.get('alpha_scores',{}), data.get('position',{}), data.get('fundamental',{})
                    row = {'Mã': sym, 'Tín hiệu': data.get('action',''), 'Score': data.get('score',0), 'Wyckoff Phase': wy.get('phase',''), 'Wyckoff Confidence': wy.get('confidence',0), 'Wyckoff Description': wy.get('description',''), 'Alpha Composite': af.get('composite',0), 'Alpha Momentum': af.get('momentum',0), 'Alpha Smart Money': af.get('smart_money',0), 'Alpha Volume': af.get('volume',0), 'Alpha Trend': af.get('trend',0), 'Position Tier': pos.get('tier',''), 'Position Size %': pos.get('size_pct',0)*100, 'Stop Loss': pos.get('stop_loss',''), 'Risk VND': pos.get('risk_vnd',0), 'PE': fund.get('pe',''), 'PB': fund.get('pb',''), 'ROE': fund.get('roe',''), 'EPS': fund.get('eps',''), 'Regime': res.get('regime',{}).get('regime','')}
                    for ag in sel_agents:
                        dec = data.get('details',{}).get(ag,{})
                        row[f'{ag}_action'] = dec.get('action','')
                        row[f'{ag}_confidence'] = dec.get('confidence',0)
                        row[f'{ag}_reasoning'] = dec.get('reasoning','')
                    excel_rows.append(row)
                df_excel = pd.DataFrame(excel_rows)
                buf = io.BytesIO()
                with pd.ExcelWriter(buf, engine='openpyxl') as writer:
                    df_excel.to_excel(writer, sheet_name='Signals', index=False)
                    signal_map = {sym: syms[sym].get('engine_signal',{}) for sym in report_syms if sym in syms}
                    alloc_df, _ = compute_portfolio_allocation(signal_map, regime_r, portfolio_value=port_value)
                    if not alloc_df.empty: alloc_df.to_excel(writer, sheet_name='Portfolio', index=False)
                    pd.DataFrame([{'Regime': regime_r.get('regime',''), 'Label': regime_r.get('label',''), 'Strategy': regime_r.get('strategy',''), 'Confidence': regime_r.get('confidence',0), 'Run Time': run_time}]).to_excel(writer, sheet_name='Regime', index=False)
                st.download_button("📊 Xuất Excel", data=buf.getvalue(), file_name=f"SuperAI_Quant_{datetime.now().strftime('%Y%m%d_%H%M')}.xlsx", mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
            except Exception as e: st.error(f"Lỗi xuất Excel: {e}")
        with col_dl3:
            export_data = {'run_time': run_time, 'regime': {k:v for k,v in regime_r.items() if k != 'signals'}, 'symbols': {sym: {'action': syms[sym].get('action',''), 'score': syms[sym].get('score',0), 'wyckoff_phase': syms[sym].get('wyckoff',{}).get('phase',''), 'alpha': syms[sym].get('alpha_scores',{}), 'decisions': syms[sym].get('details',{})} for sym in report_syms if sym in syms}}
            st.download_button("🗂 Xuất JSON", data=json.dumps(export_data, ensure_ascii=False, indent=2).encode('utf-8'), file_name=f"SuperAI_Quant_{datetime.now().strftime('%Y%m%d_%H%M')}.json", mime="application/json", use_container_width=True)

# ═══════════════════════════════════════════════════════════════
# TAB: AUTO RESEARCH AGENT
# ═══════════════════════════════════════════════════════════════
with tab_agent:
    st.markdown("### 🧠 Claude Autonomous Research Agent")
    st.caption("Claude tự chủ động gọi vnstock_data tools để suy luận — không pre-fetch dữ liệu")

    # Check availability
    try:
        from claude_autonomous_agent import create_autonomous_agent, VN_SKILLS
        _agent_available = True
    except ImportError:
        _agent_available = False
        st.warning("⚠️ Chưa có module `claude_autonomous_agent.py`. Copy file vào thư mục project.")

    if _agent_available:
        col_ag1, col_ag2 = st.columns([1, 2])
        with col_ag1:
            ag_sym   = st.selectbox("Mã cổ phiếu:", selected, key='ag_sym')
            ag_skill = st.selectbox("Skill:", list(VN_SKILLS.keys()), key='ag_skill')
            ag_cmd   = st.radio("Command:", [
                '/ic-memo  — Memo đầy đủ',
                '/comps    — Peer Analysis',
                '/dcf      — DCF Valuation',
                '/earnings — BCTC Quality',
                '/screen   — Stock Screener',
                '/morning-note — Morning Brief',
                'Custom Task',
            ], key='ag_cmd')

            if 'Custom' in ag_cmd:
                custom_task = st.text_area("Nhiệm vụ:", height=80,
                    placeholder="VD: Phân tích VCB có nên mua không...")
            else:
                custom_task = ''

            max_iter = st.slider("Max tool calls:", 3, 15, 8)
            ag_btn   = st.button("🚀 Chạy Agent", type="primary", use_container_width=True, key='ag_run')

        with col_ag2:
            st.info("""**Cách hoạt động:**
Claude tự gọi các tools:
• get_stock_price • get_fundamental
• get_income_statement • get_cash_flow
• get_foreign_flow • get_company_news
• get_market_overview • screen_stocks
• get_macro_data • get_derivatives""")

            if ag_btn:
                _api_key = os.getenv('ANTHROPIC_API_KEY', '')
                if not _api_key:
                    st.error("Cần ANTHROPIC_API_KEY trong .env")
                else:
                    with st.spinner(f"Agent đang nghiên cứu {ag_sym}..."):
                        try:
                            agent = create_autonomous_agent(source=source, start_date=start_str)
                            agent.max_iterations = max_iter
                            cmd = ag_cmd.split(' ')[0]
                            if cmd == '/ic-memo':   result = agent.cmd_ic_memo(ag_sym)
                            elif cmd == '/comps':   result = agent.cmd_comps(ag_sym)
                            elif cmd == '/dcf':     result = agent.cmd_dcf(ag_sym)
                            elif cmd == '/earnings':result = agent.cmd_earnings(ag_sym)
                            elif cmd == '/screen':  result = agent.cmd_screen("ROE > 15%, PE < 20")
                            elif cmd == '/morning-note': result = agent.cmd_morning_note(selected[:5])
                            else:                   result = agent.run(custom_task, skill=ag_skill)
                            st.session_state['agent_result'] = result
                            st.success(f"✅ {result['iterations']} iterations · {len(result['tool_calls'])} tool calls")
                        except Exception as e:
                            st.error(f"Lỗi: {e}")
                            import traceback; st.code(traceback.format_exc())

        if 'agent_result' in st.session_state:
            res_ag = st.session_state['agent_result']
            with st.expander(f"🔧 Tool Calls ({len(res_ag['tool_calls'])} calls)", expanded=False):
                for i, tc in enumerate(res_ag['tool_calls'], 1):
                    st.code(f"{i}. {tc['tool']}({json.dumps(tc['input'])[:80]}...)")
            st.markdown("#### 📋 Kết quả")
            st.markdown(res_ag['result'])
            st.download_button("📄 Tải báo cáo .md",
                data=res_ag['result'].encode('utf-8'),
                file_name=f"AutoResearch_{ag_sym}_{datetime.now().strftime('%Y%m%d_%H%M')}.md",
                mime="text/markdown")

# ═══════════════════════════════════════════════════════════════
# TAB: MULTIBAGGER AGENT
# ═══════════════════════════════════════════════════════════════
with tab_multibagger:
    render_multibagger_tab(MB_CONFIG)
"""
ui.py - Streamlit Multibagger Tab
✅ Loại bỏ asyncio.run() → tránh crash event loop
✅ Scope biến chuẩn, xử lý lỗi UI gracefully
✅ Tích hợp sẵn engine mới
"""
import streamlit as st
import pandas as pd
import time
from src.multibagger.vnstock_async import VnstockAsyncBridge
from src.multibagger.multibagger_engine import compute_yartseva_features, calculate_vn_multibagger_score, MultibaggerConfig

def render_multibagger_tab(config: dict):
    st.header("🚀 VN_MULTIBAGGER_AGENT")
    st.caption("Mô hình Yartseva (2025) • 7 yếu tố định lượng • Real-time screening")
    tab1, tab2 = st.tabs(["🔍 Screening", "📡 Real-time Monitor"])
    with tab1: _screening_tab(config)
    with tab2: _realtime_tab(config)

def _screening_tab(config: dict):
    with st.expander("⚙️ Bộ lọc", expanded=True):
        c1, c2 = st.columns(2)
        with c1:
            exch = st.selectbox("Sàn", ["HOSE", "HNX", "UPCOM", "Tất cả"])
            min_fcf = st.slider("Min FCF/P (%)", 0.0, 10.0, config["scoring"]["min_fcf_p"]*100, 0.5)/100
        with c2:
            min_roa = st.slider("Min ROA (%)", 0.0, 30.0, config["scoring"]["min_roa"]*100, 1.0)/100
            max_pr = st.slider("Max Price Range 12M", 0.0, 1.0, config["scoring"]["max_price_range_12m"], 0.05)

    if st.button("🔎 Chạy Screening", type="primary", use_container_width=True):
        with st.spinner("🔄 Đang quét thị trường..."):
            client = VnstockAsyncBridge()
            cfg = MultibaggerConfig(min_fcf_p=min_fcf, min_roa=min_roa, max_price_range_12m=max_pr)
            
            equities = client.get_list_equities(exch if exch != "Tất cả" else None)
            if equities.empty:
                st.warning("⚠️ Không lấy được danh sách CP. Kiểm tra API/Internet.")
                return

            results = []
            prog = st.progress(0)
            for i, (_, row) in enumerate(equities.iterrows()):
                sym = row.get("symbol") or row.get("ticker")
                if not sym: continue
                
                q = client.get_quote(sym)
                if not q or not q.get("price"): continue
                
                f = client.get_fundamentals(sym)
                macro = {"rate_trend": config.get("macro", {}).get("rate_trend", "stable")}
                
                feat = compute_yartseva_features(sym, q, f, macro)
                res = calculate_multibagger_score(feat, cfg)
                
                if res['score'] >= 3:
                    results.append({
                        "Mã": res['symbol'], "Điểm": res['score'], "Tín hiệu": res['signal'],
                        "FCF/P": f"{feat.get('fcf_p',0):.1%}", "ROA": f"{feat.get('roa',0):.1%}",
                        "Range12M": f"{feat.get('price_range_12m',0):.2f}", "Vùng mua": res['entry_zone']
                    })
                prog.progress(min(1.0, (i+1)/len(equities)))
            prog.empty()

            if results:
                df = pd.DataFrame(results)
                def color_sig(v):
                    return "background-color: #d4edda" if v=="BUY" else "background-color: #fff3cd" if v=="HOLD" else "background-color: #f8d7da"
                st.dataframe(df.style.map(color_sig, subset=["Tín hiệu"]), use_container_width=True, hide_index=True)
                if st.button("📥 Export CSV"):
                    st.download_button("Tải CSV", df.to_csv(index=False).encode("utf-8-sig"), "multibagger_screen.csv", "text/csv")
            else:
                st.warning("⚠️ Không tìm thấy mã đạt ngưỡng. Thử nới lỏng bộ lọc.")

def _realtime_tab(config: dict):
    wl = st.text_input("Mã theo dõi (phẩy)", value=", ".join(config["ui"]["default_watchlist"]))
    symbols = [s.strip().upper() for s in wl.split(",") if s.strip()]
    
    col_btn, col_stop = st.columns([4,1])
    with col_btn:
        start_run = st.button("▶️ Bắt đầu", type="primary")
    with col_stop:
        stop_run = st.button("⏹️ Dừng")

    if 'mb_rt_active' not in st.session_state: st.session_state.mb_rt_active = False
    if stop_run: st.session_state.mb_rt_active = False
    if start_run: st.session_state.mb_rt_active = True
    
    if st.session_state.mb_rt_active and symbols:
        placeholder = st.empty()
        client = VnstockAsyncBridge()
        cfg = MultibaggerConfig()
        
        while st.session_state.mb_rt_active:
            with placeholder.container():
                st.markdown(f"🔄 Cập nhật: {pd.Timestamp.now().strftime('%H:%M:%S')}")
                cols = st.columns(min(4, len(symbols)))
                for col, sym in zip(cols, symbols[:4]):
                    q = client.get_quote(sym)
                    if q and q.get("price"):
                        f = client.get_fundamentals(sym)
                        feat = compute_yartseva_features(sym, q, f)
                        res = calculate_multibagger_score(feat, cfg)
                        with col:
                            st.metric(sym, f"{res['score']}/7", res['signal'])
                            st.caption(f"FCF/P: {res['breakdown']['fcf_p']['val']}")
            time.sleep(config["polling"]["quote_interval"])
import numpy as np
import pandas as pd

# Tigramite PCMCI
from tigramite import data_processing as pp
from tigramite.pcmci import PCMCI
from tigramite.independence_tests.parcorr import ParCorr

# LiNGAM (causal discovery phi tuyến)
try:
    import lingam
    LINGAM_AVAILABLE = True
except ImportError:
    LINGAM_AVAILABLE = False
    print("[CausalDiscovery] lingam chưa cài – bỏ qua LiNGAM.")


# ── PCMCI ──────────────────────────────────────────────────────────────

def run_pcmci(
    df: pd.DataFrame,
    var_names: list[str] | None = None,
    tau_max: int = 4,
    alpha: float = 0.05,
) -> list[dict]:
    """
    Chạy PCMCI trên DataFrame (rows=time, cols=variables).
    Trả về list các quan hệ nhân quả có ý nghĩa thống kê.
    """
    if var_names is None:
        var_names = list(df.columns)

    data = df[var_names].dropna().values
    if data.shape[0] < 30:
        print("[PCMCI] Không đủ dữ liệu (cần ≥ 30 điểm).")
        return []

    dataframe = pp.DataFrame(data, var_names=var_names)
    pcmci = PCMCI(dataframe=dataframe, cond_ind_test=ParCorr(), verbosity=0)
    results = pcmci.run_pcmci(tau_max=tau_max, pc_alpha=alpha)

    p_matrix   = results['p_matrix']
    val_matrix = results['val_matrix']
    n = len(var_names)

    links = []
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for tau in range(1, tau_max + 1):
                p = float(p_matrix[i, j, tau])
                if p < alpha:
                    links.append({
                        'cause':    var_names[i],
                        'effect':   var_names[j],
                        'lag':      tau,
                        'strength': round(float(val_matrix[i, j, tau]), 3),
                        'p_value':  round(p, 4),
                        'method':   'PCMCI',
                    })

    return sorted(links, key=lambda x: x['p_value'])


# ── LiNGAM (instantaneous causal) ────────────────────────────────────

def run_lingam(
    df: pd.DataFrame,
    var_names: list[str] | None = None,
    threshold: float = 0.1,
) -> list[dict]:
    """
    Chạy DirectLiNGAM để phát hiện nhân quả tức thời (lag=0).
    Trả về list các quan hệ vượt ngưỡng.
    """
    if not LINGAM_AVAILABLE:
        return []
    if var_names is None:
        var_names = list(df.columns)

    data = df[var_names].dropna().values
    if data.shape[0] < 30:
        return []

    try:
        model = lingam.DirectLiNGAM()
        model.fit(data)
        adj = model.adjacency_matrix_   # shape (n, n)
    except Exception as e:
        print(f"[LiNGAM] Lỗi: {e}")
        return []

    n = len(var_names)
    links = []
    for i in range(n):
        for j in range(n):
            if i != j and abs(adj[i, j]) >= threshold:
                links.append({
                    'cause':    var_names[j],
                    'effect':   var_names[i],
                    'lag':      0,
                    'strength': round(float(adj[i, j]), 3),
                    'p_value':  None,
                    'method':   'LiNGAM',
                })
    return links


# ── Report ─────────────────────────────────────────────────────────────

def generate_causal_report(links: list[dict]) -> str:
    """Tạo báo cáo text từ danh sách quan hệ nhân quả."""
    if not links:
        return "Không phát hiện quan hệ nhân quả có ý nghĩa thống kê."

    lines = ["=== BÁO CÁO CAUSAL DISCOVERY ==="]
    for lk in links:
        lag_str = f"lag={lk['lag']} ngày" if lk['lag'] > 0 else "tức thời"
        p_str   = f", p={lk['p_value']}" if lk['p_value'] is not None else ""
        lines.append(
            f"[{lk['method']}] {lk['cause']} → {lk['effect']} "
            f"({lag_str}, strength={lk['strength']}{p_str})"
        )

    # Tóm tắt các biến ảnh hưởng nhiều nhất
    from collections import Counter
    causes = Counter(lk['cause'] for lk in links)
    top3   = causes.most_common(3)
    lines.append("\n--- Top biến nguyên nhân ---")
    for var, cnt in top3:
        lines.append(f"  {var}: {cnt} liên kết")

    return "\n".join(lines)

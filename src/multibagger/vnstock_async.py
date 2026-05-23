"""
vnstock_async.py - Robust Sync Data Bridge
✅ Xử lý đúng signature vnstock_data v3.1.7+/v4.x
✅ Retry + Exponential Backoff cho lỗi mạng/API 5xx
✅ Chuyển đổi DataFrame an toàn, không crash pipeline
"""
import time
import pandas as pd
from typing import Dict, Any, Optional
from vnstock_data import Market, Fundamental, Reference
import loguru

class VnstockAsyncBridge:
    def __init__(self, session_id: str = "default", max_retries: int = 3):
        self.session_id = session_id
        self.max_retries = max_retries
        self.logger = loguru.logger.bind(session=session_id)

    def _safe_call(self, func, *args, **kwargs) -> Optional[Any]:
        for attempt in range(self.max_retries):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                err_msg = str(e).lower()
                if any(k in err_msg for k in ['520', '503', '502', 'timeout', 'connection', 'refused']):
                    wait = (2 ** attempt) + 1
                    self.logger.warning(f"⚠️ API tạm lỗi. Chờ {wait}s... ({attempt+1}/{self.max_retries})")
                    time.sleep(wait)
                else:
                    self.logger.error(f"[API Error] {e}")
                    return None
        self.logger.error("❌ API thất bại sau nhiều lần thử.")
        return None

    def _to_df(self, data) -> pd.DataFrame:
        if data is None: return pd.DataFrame()
        if isinstance(data, pd.DataFrame): return data
        try: return pd.DataFrame(data)
        except: return pd.DataFrame()

    def get_list_equities(self, exchange: Optional[str] = None) -> pd.DataFrame:
        """Reference.equity là PROPERTY → truy cập trực tiếp."""
        try:
            ref = Reference()
            df = self._safe_call(ref.equity.list)
            df = self._to_df(df)
            if exchange and not df.empty and 'exchange' in df.columns:
                df = df[df['exchange'].astype(str).str.upper() == exchange.upper()].reset_index(drop=True)
            return df
        except Exception as e:
            self.logger.error(f"Lỗi list_equities: {e}")
            return pd.DataFrame()

    def get_quote(self, symbol: str, board: str = "ALL") -> Dict:
        """Market.equity là FUNCTION trong v4, thử linh hoạt v3/v4."""
        # Try v4 signature first
        df = self._safe_call(Market.equity, board, symbol)
        if df is None:
            # Fallback v3 signature
            df = self._safe_call(Market.equity().quote, symbol=symbol)
        return self._to_df(df).iloc[-1].to_dict() if not self._to_df(df).empty else {}

    def get_fundamentals(self, symbol: str) -> Dict:
        """Fundamental.equity là PROPERTY."""
        fund = Fundamental()
        ratio_df = self._safe_call(fund.equity.ratio, symbol=symbol)
        cf_df = self._safe_call(fund.equity.cash_flow, symbol=symbol)
        return {
            "ratio": self._to_df(ratio_df).iloc[-1].to_dict() if not self._to_df(ratio_df).empty else {},
            "cash_flow": self._to_df(cf_df).iloc[-1].to_dict() if not self._to_df(cf_df).empty else {}
        }
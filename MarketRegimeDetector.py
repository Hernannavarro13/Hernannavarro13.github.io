import argparse
from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import yfinance as yf
except ImportError as e:
    raise ImportError(
        "This script requires yfinance. Install it with: pip install yfinance"
    ) from e


@dataclass
class RegimeSummary:
    date: str
    close: float
    trend_regime: str
    vol_regime: str
    momentum_regime: str
    market_regime: str
    realized_vol_20d: Optional[float]
    momentum_20d: Optional[float]
    atr_pct: Optional[float]


class MarketRegimeDetector:
    def __init__(
        self,
        ma_fast: int = 50,
        ma_slow: int = 200,
        ma_slope_window: int = 20,
        vol_window: int = 20,
        momentum_window: int = 20,
        atr_window: int = 14,
    ):
        self.ma_fast = ma_fast
        self.ma_slow = ma_slow
        self.ma_slope_window = ma_slope_window
        self.vol_window = vol_window
        self.momentum_window = momentum_window
        self.atr_window = atr_window

    def add_features(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        df["return"] = df["Close"].pct_change()
        df["log_return"] = np.log(df["Close"] / df["Close"].shift(1))

        df["ma_fast"] = df["Close"].rolling(self.ma_fast).mean()
        df["ma_slow"] = df["Close"].rolling(self.ma_slow).mean()
        df["ma_fast_slope"] = (
            df["ma_fast"] - df["ma_fast"].shift(self.ma_slope_window)
        ) / self.ma_slope_window

        df["momentum_20d"] = df["Close"].pct_change(self.momentum_window)
        df["realized_vol_20d"] = (
            df["log_return"].rolling(self.vol_window).std() * np.sqrt(252)
        )

        high_low = df["High"] - df["Low"]
        high_prev_close = (df["High"] - df["Close"].shift(1)).abs()
        low_prev_close = (df["Low"] - df["Close"].shift(1)).abs()

        df["true_range"] = pd.concat(
            [high_low, high_prev_close, low_prev_close], axis=1
        ).max(axis=1)

        df["atr"] = df["true_range"].rolling(self.atr_window).mean()
        df["atr_pct"] = df["atr"] / df["Close"]

        return df

    def classify_trend(self, row) -> str:
        if pd.isna(row["ma_fast"]) or pd.isna(row["ma_slow"]):
            return "Unknown"

        if row["Close"] > row["ma_fast"] and row["ma_fast"] > row["ma_slow"]:
            return "Bullish"
        if row["Close"] < row["ma_fast"] and row["ma_fast"] < row["ma_slow"]:
            return "Bearish"
        return "Neutral"

    def classify_volatility(self, row) -> str:
        vol = row["realized_vol_20d"]
        if pd.isna(vol):
            return "Unknown"
        if vol < 0.15:
            return "Low"
        if vol < 0.25:
            return "Medium"
        return "High"

    def classify_momentum(self, row) -> str:
        mom = row["momentum_20d"]
        if pd.isna(mom):
            return "Unknown"
        if mom > 0.02:
            return "Positive"
        if mom < -0.02:
            return "Negative"
        return "Flat"

    def map_regime(self, trend: str, vol: str, momentum: str) -> str:
        if "Unknown" in {trend, vol, momentum}:
            return "Unknown"

        if trend == "Bullish" and vol == "Low" and momentum == "Positive":
            return "Calm Bull Trend"
        if trend == "Bullish" and vol in {"Medium", "High"}:
            return "Volatile Bull"
        if trend == "Bearish" and vol == "High" and momentum == "Negative":
            return "Panic Bear"
        if trend == "Bearish" and vol in {"Low", "Medium"}:
            return "Calm Bear Trend"
        if trend == "Neutral" and vol == "Low":
            return "Low-Vol Chop"
        if trend == "Neutral" and vol in {"Medium", "High"}:
            return "High-Vol Chop"
        return "Transition"

    def detect_regimes(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self.add_features(df)
        df["trend_regime"] = df.apply(self.classify_trend, axis=1)
        df["vol_regime"] = df.apply(self.classify_volatility, axis=1)
        df["momentum_regime"] = df.apply(self.classify_momentum, axis=1)
        df["market_regime"] = df.apply(
            lambda row: self.map_regime(
                row["trend_regime"], row["vol_regime"], row["momentum_regime"]
            ),
            axis=1,
        )
        return df

    def latest_regime_summary(self, df: pd.DataFrame) -> RegimeSummary:
        row = df.iloc[-1]
        return RegimeSummary(
            date=str(row.get("Date", df.index[-1])),
            close=float(row["Close"]),
            trend_regime=row["trend_regime"],
            vol_regime=row["vol_regime"],
            momentum_regime=row["momentum_regime"],
            market_regime=row["market_regime"],
            realized_vol_20d=None if pd.isna(row["realized_vol_20d"]) else float(row["realized_vol_20d"]),
            momentum_20d=None if pd.isna(row["momentum_20d"]) else float(row["momentum_20d"]),
            atr_pct=None if pd.isna(row["atr_pct"]) else float(row["atr_pct"]),
        )


def fetch_price_data(symbol: str = "SPY", period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    df = yf.download(
        symbol,
        period=period,
        interval=interval,
        auto_adjust=False,
        progress=False,
    )

    if df.empty:
        raise ValueError(f"No data returned for {symbol} with period={period}, interval={interval}")

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    df = df.reset_index()

    required_cols = {"Date", "Open", "High", "Low", "Close", "Volume"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {sorted(missing)}")

    df["Date"] = pd.to_datetime(df["Date"])
    return df


def run_regime_analysis(symbol: str = "SPY", period: str = "2y", interval: str = "1d") -> Tuple[pd.DataFrame, RegimeSummary]:
    prices = fetch_price_data(symbol=symbol, period=period, interval=interval)
    detector = MarketRegimeDetector()
    regime_df = detector.detect_regimes(prices)
    summary = detector.latest_regime_summary(regime_df)
    return regime_df, summary


def regime_to_numeric(df: pd.DataFrame) -> pd.DataFrame:
    mapping: Dict[str, int] = {
        "Unknown": 0,
        "Low-Vol Chop": 1,
        "High-Vol Chop": 2,
        "Transition": 3,
        "Calm Bull Trend": 4,
        "Volatile Bull": 5,
        "Calm Bear Trend": 6,
        "Panic Bear": 7,
    }
    out = df.copy()
    out["regime_code"] = out["market_regime"].map(mapping).fillna(0).astype(int)
    return out


def plot_regime_chart(df: pd.DataFrame, symbol: str = "SPY", save_path: str = "spy_regime_chart.png") -> None:
    plot_df = regime_to_numeric(df).copy()

    fig, ax = plt.subplots(figsize=(16, 8))
    ax.plot(plot_df["Date"], plot_df["Close"], label=f"{symbol} Close")
    ax.plot(plot_df["Date"], plot_df["ma_fast"], label="50 DMA")
    ax.plot(plot_df["Date"], plot_df["ma_slow"], label="200 DMA")

    change_points = plot_df["market_regime"].ne(plot_df["market_regime"].shift(1))
    label_rows = plot_df.loc[change_points].copy()

    for _, row in label_rows.iterrows():
        if pd.notna(row["Close"]):
            ax.annotate(
                row["market_regime"],
                (row["Date"], row["Close"]),
                textcoords="offset points",
                xytext=(0, 8),
                ha="center",
                fontsize=8,
                rotation=35,
            )

    ax.set_title(f"{symbol} Price With Regime Labels")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_regime_timeline(df: pd.DataFrame, symbol: str = "SPY", save_path: str = "spy_regime_timeline.png") -> None:
    plot_df = regime_to_numeric(df).copy()

    fig, ax = plt.subplots(figsize=(16, 4))
    ax.step(plot_df["Date"], plot_df["regime_code"], where="post")

    y_labels = {
        0: "Unknown",
        1: "Low-Vol Chop",
        2: "High-Vol Chop",
        3: "Transition",
        4: "Calm Bull Trend",
        5: "Volatile Bull",
        6: "Calm Bear Trend",
        7: "Panic Bear",
    }
    ax.set_yticks(list(y_labels.keys()))
    ax.set_yticklabels(list(y_labels.values()))
    ax.set_title(f"{symbol} Regime Timeline")
    ax.set_xlabel("Date")
    ax.set_ylabel("Regime")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Fetch market data, detect regimes, and chart them.")
    parser.add_argument("--symbol", default="SPY", help="Ticker symbol, default SPY")
    parser.add_argument("--period", default="2y", help="yfinance period, default 2y")
    parser.add_argument("--interval", default="1d", help="yfinance interval, default 1d")
    parser.add_argument("--save-csv", default="", help="Optional path to save full regime data as CSV")
    parser.add_argument("--chart-path", default="spy_regime_chart.png", help="Path for labeled price chart")
    parser.add_argument("--timeline-path", default="spy_regime_timeline.png", help="Path for regime timeline chart")
    args = parser.parse_args()

    regime_df, summary = run_regime_analysis(
        symbol=args.symbol,
        period=args.period,
        interval=args.interval,
    )

    print("\nLatest regime summary\n")
    print(f"Date: {summary.date}")
    print(f"Close: {summary.close:.2f}")
    print(f"Trend: {summary.trend_regime}")
    print(f"Volatility: {summary.vol_regime}")
    print(f"Momentum: {summary.momentum_regime}")
    print(f"Market Regime: {summary.market_regime}")

    if summary.realized_vol_20d is not None:
        print(f"20d Realized Vol: {summary.realized_vol_20d:.2%}")
    if summary.momentum_20d is not None:
        print(f"20d Momentum: {summary.momentum_20d:.2%}")
    if summary.atr_pct is not None:
        print(f"ATR %: {summary.atr_pct:.2%}")

    print("\nLast 10 rows\n")
    cols = [
        "Date", "Close", "trend_regime", "vol_regime",
        "momentum_regime", "market_regime", "realized_vol_20d",
        "momentum_20d", "atr_pct"
    ]
    print(regime_df[cols].tail(10).to_string(index=False))

    plot_regime_chart(regime_df, symbol=args.symbol, save_path=args.chart_path)
    plot_regime_timeline(regime_df, symbol=args.symbol, save_path=args.timeline_path)
    print(f"\nSaved labeled price chart to: {args.chart_path}")
    print(f"Saved regime timeline chart to: {args.timeline_path}")

    if args.save_csv:
        regime_df.to_csv(args.save_csv, index=False)
        print(f"Saved full results to: {args.save_csv}")


if __name__ == "__main__":
    main()

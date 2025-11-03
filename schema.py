from typing import TypedDict, List, Optional, Any


# -----------------------------
# Sub-Schemas
# -----------------------------

class CurrentValue(TypedDict):
    price: float
    volume: str
    rsi: float
    macd: float
    macd_signal: float
    sma_50: float
    atr_14: float
    pe_ratio: float
    eps: float
    upcoming_earnings: str


class ModelMetrics(TypedDict):
    MAE: float
    RMSE: float
    MAPE: float
    SMAPE: float


class PredictedAvgPrices(TypedDict):
    Next_Day_Price: float
    Next_Week_Avg: float
    Next_Month_Avg: float
    Next_Quarter_Avg: float
    Next_Half_Year_Avg: float
    Next_Year_Avg: float


class ResearcherState(TypedDict):
    bull: List[str]
    bear: List[str]
    facilitator: List[str]
    round: int


class RiskDebateState(TypedDict):
    positive: List[str]
    negative: List[str]
    facilitator: List[str]
    round: int


# -----------------------------
# MAIN STATE
# -----------------------------

class MarketState(TypedDict):
    stock: str
    user_action: str
    num_stocks: int

    current_value: CurrentValue
    metrics: ModelMetrics
    future_averages: PredictedAvgPrices

    news_summaries: List[str]
    reddit_summaries: List[str]

    researcher: ResearcherState
    risk_debate: RiskDebateState
    strategist: str

    # Streamlit Containers
    news_container: Optional[Any]
    reddit_container: Optional[Any]
    researcher_round1_container: Optional[Any]
    researcher_round2_container: Optional[Any]
    researcher_round3_container: Optional[Any]
    risk_analysis_round1_container: Optional[Any]
    risk_analysis_round2_container: Optional[Any]
    risk_analysis_round3_container: Optional[Any]
    strategy_container: Optional[Any]

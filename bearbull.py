# agents.py

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import json



llm = init_chat_model(
    "llama-3.3-70b-versatile",   # <-- Groq reasoning model
    model_provider="groq",       # <-- switch provider
    temperature=0.5,
    api_key="api key"  # <-- your Groq key
)



# =========================================
# CLEANUP FUNCTION (inside agents.py)
# =========================================
def cleanup_output(output: str) -> str:
    """
    Cleans the model output (removes code fences or extra quotes).
    """
    if not output:
        return ""
    output = output.strip()
    if output.startswith("```"):
        output = output.replace("```", "").strip()
    return output


# =========================================
# BULL AGENT
# =========================================
def bull_agent(state: dict) -> dict:
    """
    Bull agent — optimistic market analyst.
    Responds based on context and last Bear statement.
    """
    # Ensure proper structure
    # state.setdefault("researcher", {})
    # state["researcher"].setdefault("bull", [])
    # state["researcher"].setdefault("bear", [])
    # state["researcher"].setdefault("facilitator", [])
    # state["researcher"]["round"] = state["researcher"].get("round", 0) + 1

    # 🔹 Safely extract multiple parts from flat state
    stock = state["stock"]
    #sector = state["sector"]
    current_metrics = state["current_value"]
    predicted_avg_prices = state["future_averages"]
    model_metrics = state["metrics"]
    news_summaries = state["news_summaries"]
    social_media_summary = state["reddit_summaries"]

    researcher = state.get("researcher", {}).copy()

    # 🔹 Optional: Combine into one context for the LLM
    context = json.dumps(
        {
            "Stock": stock,
            "Current Metrics": current_metrics,
            "Predicted Prices": predicted_avg_prices,
            "Model Performance": model_metrics,
            "News Headlines": news_summaries,
            "Social Media Sentiment": social_media_summary
        },
        indent=2,
        default=str  
    )
    last_bear = researcher.get("bear", [])[-1] if researcher.get("bear") else ""

    prompt = f"""
You are the Bull Agent — a data-driven optimistic market analyst.

Context:
{context}

Bear Agent previously said: "{last_bear}"

Your task:
- Respond with a concise, factual, and optimistic bullish view about the stock (2–3 sentences).
- Use specific data or trends from the context to support your optimism.
- If the Bear raised a valid concern, acknowledge it briefly but counter it logically.
- Avoid exaggeration or emotional tone.
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    bull_reply = cleanup_output(res.content)
    researcher["bull"].append(bull_reply)
    # print(stock)
    # print(context)
    return {**state, "researcher": researcher}


# =========================================
# BEAR AGENT
# =========================================
def bear_agent(state: dict) -> dict:
    """
    Bear agent — cautious market analyst.
    Responds based on context and last Bull statement.
    """
    # state.setdefault("researcher", {})
    # state["researcher"].setdefault("bear", [])
    # state["researcher"].setdefault("bull", [])
    # state["researcher"].setdefault("facilitator", [])
    # state["researcher"]["round"] = state["researcher"].get("round", 0) + 1

    # 🔹 Safely extract multiple parts from flat state
    stock = state["stock"]
    #sector = state["sector"]
    current_metrics = state["current_value"]
    predicted_avg_prices = state["future_averages"]
    model_metrics = state["metrics"]
    news_summaries = state["news_summaries"]
    social_media_summary = state["reddit_summaries"]

    researcher = state.get("researcher", {}).copy()

    # 🔹 Optional: Combine into one context for the LLM
    context = json.dumps(
        {
            "Stock": stock,
            "Current Metrics": current_metrics,
            "Predicted Prices": predicted_avg_prices,
            "Model Performance": model_metrics,
            "News Headlines": news_summaries,
            "Social Media Sentiment": social_media_summary
        },
        indent=2,
        default=str  
    )
    # last_bull = state["researcher"]["bull"][-1] if state["researcher"]["bull"] else ""
    last_bull = researcher.get("bull", [])[-1] if researcher.get("bull") else ""

    prompt = f"""
You are the Bear Agent — a data-driven cautious market analyst.

Context:
{context}

Bull Agent previously said: "{last_bull}"

Your task:
- Respond with a concise, factual, and cautious bearish view about the stock (2–3 sentences).
- Use specific data or market risks from the context to support your caution.
- If the Bull raised optimism, acknowledge valid points but counter them with risk factors.
- Avoid emotional tone or bias — stay analytical and grounded.
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    bear_reply = cleanup_output(res.content)
    researcher["bear"].append(bear_reply)
    return {**state, "researcher": researcher}


# =========================================
# FACILITATOR AGENT
# =========================================
def facilitator_agent(state: dict) -> dict:
    """
    Facilitator agent — observes the debate and summarizes the current round.
    """
    # state.setdefault("researcher", {})
    # state["researcher"].setdefault("bear", [])
    # state["researcher"].setdefault("bull", [])
    # state["researcher"].setdefault("facilitator", [])
    # state["researcher"]["round"] = state["researcher"].get("round", 0) + 1
    researcher = state.get("researcher", {}).copy()
    last_bull = researcher.get("bull", [])[-1] if researcher.get("bull") else ""
    last_bear = researcher.get("bear", [])[-1] if researcher.get("bear") else ""

    # last_bull = state["researcher"]["bull"][-1] if state["researcher"]["bull"] else ""
    # last_bear = state["researcher"]["bear"][-1] if state["researcher"]["bear"] else ""

    prompt = f"""
You are the Facilitator Agent — a neutral observer in a Bull vs Bear debate.

Latest Bull statement:
"{last_bull}"

Latest Bear statement:
"{last_bear}"

Your task:
- Provide a short (2–3 sentences) summary comparing their perspectives.
- Highlight if either side made stronger evidence-based arguments.
- qoute the stronger argument
- Stay strictly neutral and concise.
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    facilitator_reply = cleanup_output(res.content)
    researcher["facilitator"].append(facilitator_reply)
    researcher["round"] = researcher.get("round") + 1
    return {**state, "researcher": researcher}


# def run_researcher_debate_round(state: dict) -> dict:
#     """
#     Runs:
#     ✅ Bull Agent
#     ✅ Bear Agent
#     ✅ Facilitator Agent

#     Returns fully updated state.
#     """
#     state = bull_agent(state)
#     state = bear_agent(state)
#     state = facilitator_agent(state)
#     return state



# initial_state = {
#   "stock": "TSLA",
#   "user_action": "Buy",
#   "num_stocks": 1,
#   "current_value": {
#     "price": 456.55999755859375,
#     "volume": "82980800",
#     "rsi": 57.53794599163974,
#     "macd": 10.844141579925065,
#     "macd_signal": 11.675328347828284,
#     "sma_50": 410.6402001953125,
#     "atr_14": 18.15000261579241,
#     "pe_ratio": 321.10138,
#     "eps": 1.46,
#     "upcoming_earnings": "2026-01-28, 2025-10-22, 2025-07-23, 2025-04-22, 2025-01-29"
#   },
#   "metrics": {
#     "MAE": 114.53200576716232,
#     "RMSE": 125.46185070212148,
#     "MAPE": 0.3332221652998497,
#     "SMAPE": 41.04262246028171
#   },
#   "future_averages": {
#     "Next Day Price": 289.59,
#     "Next Week Avg": 278.25,
#     "Next Month Avg": 229.39,
#     "Next Quarter Avg": 202.24,
#     "Next Half-Year Avg": 193.99,
#     "Next Year Avg": 162.47
#   },
#   "news_summaries": [
#     "in Sweden, Tesla registered only 133 new vehicles in October, a nearly 89% drop from a year earlier . demand continues to falter despite broader EV market growth .",
#     "TLDR NHTSA received 16 new reports of people trapped in vehicles . the company must provide extensive records by December 10 or face fines up to $27,874 per day . china is preparing new safety standards for electronic door handles .",
#     "the deal is said to be worth over 3 trillion won (approximately $2.1 billion) it will see the south Korean battery giant supply cells to Tesla over a three-year period . this supply is reportedly for Tesla\u2019s energy storage system (ESS) business .",
#     "TSLA's annual shareholder meeting is shaping up less like a corporate vote . the proposal would give Musk the right to earn up to 304 million Tesla shares over the next decade if he\u2019s able to sixfold the company's market cap . a $1 trillion stock award could make Musk the world's first trillionaire .",
#     "this marks a +13% gain year-to-date, recouping all 2025 losses and nearing its all-time high ($489 in late 2024) the recent uptrend has TSLA outperforming many rivals, although it still lags the broader S&P 500\u2019s 18% gain this year . EPS came in at $0."
#   ],
#   "reddit_summaries": [
#     "**Market Trend Overview: Tesla, Inc. (TSLA) Stock**\n\nThe public discussion around Tesla, Inc. (TSLA) stock is relatively muted in the given Reddit summary. However, it's worth noting that the AI and cryptocurrency-related discussions may have a broader impact on the market sentiment. There is no direct mention of TSLA stock performance or sentiment, but the overall market trend seems to be influenced by the dominance of NVDA and AMZN."
#   ],
#   "researcher": {
#     "bull": [],
#     "bear": [],
#     "facilitator": [],
#     "round": 0
#   },
#   "risk_debate": {
#     "positive": [],
#     "negative": [],
#     "facilitator": [],
#     "round": 0
#   },
#   "strategist": "",
#   "news_container": "DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(23,), _index=5), _parent=DeltaGenerator(), _block_type='flex_container', _form_data=FormData(form_id=''))",
#   "reddit_container": "DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(24,), _index=1), _parent=DeltaGenerator(), _block_type='flex_container', _form_data=FormData(form_id=''))",
#   "researcher_round1_container": "DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(25,)), _parent=DeltaGenerator(), _block_type='flex_container', _form_data=FormData(form_id=''))",
#   "researcher_round2_container": "DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(26,)), _parent=DeltaGenerator(), _block_type='flex_container', _form_data=FormData(form_id=''))",
#   "researcher_round3_container": "DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(27,)), _parent=DeltaGenerator(), _block_type='flex_container', _form_data=FormData(form_id=''))",
#   "risk_analysis_round1_container": "DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(28,)), _parent=DeltaGenerator(), _block_type='flex_container', _form_data=FormData(form_id=''))",
#   "risk_analysis_round2_container": "DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(29,)), _parent=DeltaGenerator(), _block_type='flex_container', _form_data=FormData(form_id=''))",
#   "risk_analysis_round3_container": "DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(30,)), _parent=DeltaGenerator(), _block_type='flex_container', _form_data=FormData(form_id=''))",
#   "strategy_container": "DeltaGenerator(_provided_cursor=RunningCursor(_parent_path=(31,)), _parent=DeltaGenerator(), _block_type='flex_container', _form_data=FormData(form_id=''))",
#   "company_name": "TSLA"
# }


# updated_state = run_researcher_debate_round(initial_state)
# print(updated_state["researcher"])

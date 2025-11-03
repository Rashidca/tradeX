from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage
from langgraph.graph import StateGraph, END


import json
import os
data = {
    "stock": "TCS",
    "sector": "Information Technology",
    "current_metrics": {
        "current_price": 3852.4,
        "price_change_percent": 1.45,
        "rsi": 67.8,
        "volume": 1840000,
        "volatility": "moderate",
        "market_cap": "14.3T INR"
    },
    "predicted_avg_prices": {
        "next_day": 3875.6,
        "next_week": 3910.2,
        "next_month": 3988.4,
        "next_quarter": 4105.3,
        "next_6_months": 4232.7,
        "next_year": 4390.1
    },
    "model_metrics": {
        "MAPE": 2.4,
        "RMSE": 42.8,
        "R2": 0.94,
        "model_name": "LSTM Forecast Model v3.1"
    },
    "news_headlines": [
        "TCS posts stronger-than-expected Q2 earnings with steady margin growth",
        "Rupee stability and new deal wins support IT export momentum",
        "Analysts see long-term upside driven by AI transformation demand"
    ],
    "social_media_summary": [
        "Investors on Reddit describe TCS as a 'safe long-term compounder'",
        "Some users expect short-term correction due to high RSI and sector rotation"
    ]
}


# File name
filename = "market_news.json"

# Write to JSON file
with open(filename, "w") as f:
    json.dump(data, f, indent=4)

print(f"✅ JSON file '{filename}' created successfully!")
with open("market_news.json", "r") as f:
    market_data = json.load(f)

context = f"""
stock: {market_data['stock']}
sector: {market_data['sector']}

News Summary: {"; ".join(market_data['news_headlines'])}

Current Metrics:
  • Current Price: ₹{market_data['current_metrics']['current_price']}
  • Change: {market_data['current_metrics']['price_change_percent']}%
  • RSI: {market_data['current_metrics']['rsi']}
  • Volume: {market_data['current_metrics']['volume']}
  • Volatility: {market_data['current_metrics']['volatility']}
  • Market Cap: {market_data['current_metrics']['market_cap']}

Predicted Average Prices:
  • Next Day: ₹{market_data['predicted_avg_prices']['next_day']}
  • Next Week: ₹{market_data['predicted_avg_prices']['next_week']}
  • Next Month: ₹{market_data['predicted_avg_prices']['next_month']}
  • Next Quarter: ₹{market_data['predicted_avg_prices']['next_quarter']}
  • Next 6 Months: ₹{market_data['predicted_avg_prices']['next_6_months']}
  • Next Year: ₹{market_data['predicted_avg_prices']['next_year']}

Model Performance:
  • Model Used: {market_data['model_metrics']['model_name']}
  • MAPE: {market_data['model_metrics']['MAPE']}%
  • RMSE: {market_data['model_metrics']['RMSE']}
  • R² Score: {market_data['model_metrics']['R2']}

Social Media Sentiment: {"; ".join(market_data['social_media_summary'])}
"""
os.environ["GROQ_API_KEY"] = "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"
llm = ChatGroq(model="llama-3.1-8b-instant")

class DebateState(dict):
    bull: list
    bear: list
    facilitator: list
    round: int

def bull_agent(state: DebateState):
    last_bear = state["bear"][-1] if state["bear"] else ""
    prompt = f"""You are the Bull Agent — a data-driven optimistic market analyst.

        Context:
        {context}

        Bear Agent previously said: "{last_bear}"

        Your task:
        - Respond with a concise, factual, and optimistic view (2–3 sentences).
        - Use specific data or trends from the context to support your optimism.
        - If the Bear raised a valid concern, acknowledge it briefly but counter it logically.
        - Avoid exaggeration or emotional tone."""
    res = llm.invoke([HumanMessage(content=prompt)])
    state["bull"].append(res.content)
    return state

def bear_agent(state: DebateState):
    last_bull = state["bull"][-1] if state["bull"] else ""
    prompt = f"""You are the Bear Agent — a data-driven cautious market analyst.

      Context:
      {context}

      Bull Agent previously said: "{last_bull}"

      Your task:
      - Respond with a realistic and cautious perspective (2–4 sentences).
      - Use data, sentiment, or model reliability concerns from the context.
      - Highlight potential risks or weaknesses logically.
      - Keep tone neutral and factual — no exaggeration."""
    res = llm.invoke([HumanMessage(content=prompt)])
    state["bear"].append(res.content)
    return state

def facilitator(state: DebateState):
    last_bull = state["bull"][-1]
    last_bear = state["bear"][-1]
    prompt = f"""oYou are a neutral Facilitator moderating a market debate.

Context:
{context}

Bull said: "{last_bull}"
Bear said: "{last_bear}"

Your task:
1. Identify and summarize each side's main point in one short, precise line each — no extra details.
2. Compare their logic, data alignment, and factual soundness.
3. Clearly declare which argument was **stronger in this round**, with a short justification.

Be concise and professional.
Example format:
- Bull: [one-line summary]
- Bear: [one-line summary]
- Stronger Argument: [Bull/Bear] — [short reason]"""
    res = llm.invoke([HumanMessage(content=prompt)])
    state["facilitator"].append(res.content)
    state["round"] += 1
    return state
def clean_state_value(value):
    """Recursively clean newlines and unicode escapes from nested dict/list values."""
    if isinstance(value, dict):
        return {k: clean_state_value(v) for k, v in value.items()}
    elif isinstance(value, list):
        return [clean_state_value(v) for v in value]
    elif isinstance(value, str):
        return value.replace("\n", " ").replace("\\u20b9", "₹").strip()
    else:
        return value

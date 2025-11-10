# =========================================
# RISK ANALYSIS DEBATE TEAM (Positive vs Negative)
# =========================================

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import json


# ✅ Initialize same Groq LLM
llm = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.7,
    api_key="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"
)


# ✅ Reuse cleanup function
def cleanup_output(output: str) -> str:
    if not output:
        return ""
    output = output.strip()
    if output.startswith("```"):
        output = output.replace("```", "").strip()
    return output


# =========================================
# POSITIVE RISK AGENT
# =========================================
def positive_risk_agent(state: dict) -> dict:
    """
    Positive Risk Agent — argues why the user's intended action is acceptable
    and the potential risks are manageable given the data.
    """
    # Extract info
    stock = state["stock"]
    #sector = state["sector"]
    nums_stocks = state["num_stocks"]
    user_action = state["user_action"]
    current_metrics = state["current_value"]
    predicted_avg_prices = state["future_averages"]
    model_metrics = state["metrics"]
    news_summaries = state["news_summaries"]
    social_media_summary = state["reddit_summaries"]
    researchers_facilitator = state["researcher"]["facilitator"]
    risk_debate = state.get("risk_debate", {}).copy()

    # 🔹 Optional: Combine into one context for the LLM
    context = json.dumps(
        {
            "Stock": stock,
            "User Action": user_action,
            "Number of Stocks": nums_stocks,
            "Current Metrics": current_metrics,
            "Predicted Prices": predicted_avg_prices,
            "Model Performance": model_metrics,
            "News Headlines": news_summaries,
            "Social Media Sentiment": social_media_summary,
            "Researcher Facilitator": researchers_facilitator
        },
        indent=2,
        default=str
    )

    last_negative = risk_debate.get("negative", [])[-1] if risk_debate.get("negative") else ""

    prompt = f"""
You are the Positive Risk Agent — an optimistic but data-driven risk analyst.

Context:
{context}

The Negative Risk Agent previously said: "{last_negative}"

Your task:
- Argue why the user's intended action ({user_action}) is acceptable or manageable.
- Use metrics (price prediction, sentiment, market stability) to justify why the risk is worth taking.
- If the Negative agent raises valid concerns, acknowledge them but provide logical counterpoints.
- Keep the tone factual and concise (2–3 sentences).
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    pos_reply = cleanup_output(res.content)
    risk_debate["positive"] = risk_debate.get("positive", []) + [pos_reply]
    return {**state, "risk_debate": risk_debate}


# =========================================
# NEGATIVE RISK AGENT
# =========================================
def negative_risk_agent(state: dict) -> dict:
    """
    Negative Risk Agent — argues why the user's intended action might be too risky
    or why caution is advisable.
    """
    stock = state["stock"]
    #sector = state["sector"]
    nums_stocks = state["num_stocks"]
    user_action = state["user_action"]
    current_metrics = state["current_value"]
    predicted_avg_prices = state["future_averages"]
    model_metrics = state["metrics"]
    news_summaries = state["news_summaries"]
    social_media_summary = state["reddit_summaries"]
    researchers_facilitator = state["researcher"]["facilitator"]

    risk_debate = state.get("risk_debate", {}).copy()

    # 🔹 Optional: Combine into one context for the LLM
    context = json.dumps(
        {
            "Stock": stock,
            "User Action": user_action,
            "Number of Stocks": nums_stocks,
            "Current Metrics": current_metrics,
            "Predicted Prices": predicted_avg_prices,
            "Model Performance": model_metrics,
            "News Headlines": news_summaries,
            "Social Media Sentiment": social_media_summary,
            "Researcher Facilitator": researchers_facilitator
        },
        indent=2,
        default=str
    )

    last_positive = risk_debate.get("positive", [])[-1] if risk_debate.get("positive") else ""

    prompt = f"""
You are the Negative Risk Agent — a cautious, risk-aware analyst.

Context:
{context}

The Positive Risk Agent previously said: "{last_positive}"

Your task:
- Argue why the user's intended action ({user_action}) could be risky right now.
- Use evidence such as volatility, uncertainty, or market instability to support your reasoning.
- If the Positive agent made good points, acknowledge them but emphasize the possible downsides.
- Stay factual and realistic (2–3 sentences).
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    neg_reply = cleanup_output(res.content)
    risk_debate["negative"] = risk_debate.get("negative", []) + [neg_reply]
    return {**state, "risk_debate": risk_debate}


# =========================================
# RISK FACILITATOR AGENT
# =========================================
def risk_facilitator_agent(state: dict) -> dict:
    """
    Risk Facilitator Agent — observes the Positive vs Negative Risk debate and
    summarizes the outcome with a balanced conclusion.
    """
    risk_debate = state.get("risk_debate", {}).copy()
    last_positive = risk_debate.get("positive", [])[-1] if risk_debate.get("positive") else ""
    last_negative = risk_debate.get("negative", [])[-1] if risk_debate.get("negative") else ""

    prompt = f"""
You are the Risk Facilitator Agent — a neutral observer in a Positive vs Negative Risk debate.

Latest Positive Risk Agent statement:
"{last_positive}"

Latest Negative Risk Agent statement:
"{last_negative}"

Your task:
- Provide a short (2–3 sentences) summary comparing their perspectives.
- Decide which side provided stronger, evidence-based reasoning about risk.
- Quote the stronger reasoning and give a risk stance: "Low Risk", "Moderate Risk", or "High Risk".
- also show how much riskiness out of 10.
- Remain strictly neutral, factual, and concise.
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    facilitator_reply = cleanup_output(res.content)
    risk_debate["facilitator"] = risk_debate.get("facilitator", []) + [facilitator_reply]
    risk_debate["round"]=risk_debate.get("round",0)+1
    return {**state, "risk_debate": risk_debate}



# if __name__ == "__main__":
#     print("\n✅ Running Risk Debate Test…\n")

#     # Dummy state
#     state = {
#         "stock": "TSLA",
#         "num_stocks": 5,
#         "user_action": "Buy",
#         "current_value": {"price": 250.4, "rsi": 55.1},
#         "future_averages": {"7_days": 260.2, "30_days": 280.4},
#         "metrics": {"mse": 0.12, "mae": 0.08},
#         "news_summaries": ["Tesla beats earnings expectations."],
#         "reddit_summaries": ["Sentiment mildly positive."],
#         "risk_debate": {}
#     }

#     # Run agents
#     state = positive_risk_agent(state)
#     state = negative_risk_agent(state)
#     state = risk_facilitator_agent(state)

#     # Print results
#     print("✅ Positive Agent:\n", state["risk_debate"]["positive"][-1], "\n")
#     print("✅ Negative Agent:\n", state["risk_debate"]["negative"][-1], "\n")
#     print("✅ Facilitator:\n", state["risk_debate"]["facilitator"][-1], "\n")

#     print("✅ Round:", state["risk_debate"]["round"])
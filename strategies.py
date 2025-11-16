# =========================================
# STRATEGIST AGENT (LLM-based, full debate + beginner-friendly summary)
# =========================================

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage
import json

llm = init_chat_model(
    "llama-3.3-70b-versatile",
    model_provider="groq",
    temperature=0.7,
    api_key="api key"
)

def cleanup_output(output: str) -> str:
    if not output:
        return ""
    output = output.strip()
    if output.startswith("```"):
        output = output.replace("```", "").strip()
    return output


def strategist_agent(state: dict) -> dict:
    """
    Strategist Agent — reads all facilitator summaries and gives a clear,
    beginner-friendly final recommendation about what to do and how many stocks to act on.
    """

    user_action = state.get("user_action")  # "Buy" / "Sell" / "Hold"
    num_stocks = state.get("num_stocks", 1)
    risk_debate = state.get("risk_debate", {})
    facilitator_summaries = risk_debate.get("facilitator", [])

    context = json.dumps(
        {
            "User Action": user_action,
            "User Suggested Stocks": num_stocks,
            "Facilitator Round Summaries": facilitator_summaries
        },
        indent=2
    )

    prompt = f"""
You are the **Strategist Agent**, the final decision-maker after a team of risk analysts debated.
Your role is to explain the outcome in plain, friendly English — as if you’re talking to someone new to investing.

Context:
{context}

Your task:
1. Read all facilitator summaries and understand the full story of the debate — how risk changed and what tone the discussion took.
2. Decide whether the user's intended action ("{user_action}") is wise right now.
3. Suggest whether they should **buy/sell/hold fully, partially, or even increase the amount** — you can recommend more or fewer stocks than the user suggested ({num_stocks}).
4. Quote or paraphrase one key point from the facilitator summaries that most influenced your decision.
5. End with a **simple, humanized explanation** that a beginner could understand — calm, kind, and clear. Avoid jargon.

### Output Format:
**What I Understood:** <1–2 sentences explaining what the debate revealed overall.>
**Key Insight:** "<quote or paraphrased insight from facilitators>"
**My Advice:** <final action with exact number of stocks>
**Why (in simple words):** <clear, beginner-friendly reasoning – like talking to a friend who’s just learning about stocks.>
"""

    res = llm.invoke([HumanMessage(content=prompt)])
    strategist_reply = cleanup_output(res.content)
    state["strategist_result"] = strategist_reply
    return state

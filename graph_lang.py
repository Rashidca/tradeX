# graph_setup.py

from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from typing import Any

# Import your tools
from news_agent_tool import news_agent_tool
from news_ui import news_ui_agent
from reddit2 import RedditStockNewsTool
from reddit_ai import reddit_ai_agent


# ✅ Shared memory state for the whole graph
class NewsState(BaseModel):
    company_name: str = Field(..., description="Company being analyzed")
    news_analyst: list[str] | None = None
    reddit_summary: list[str] | None = None
    ai_analysis: str | None = None

    news_container: Any | None = None
    reddit_container: Any | None = None
    ai_container: Any | None = None

    class Config:
        arbitrary_types_allowed = True

    def get(self, key, default=None):
        return getattr(self, key, default)


# ✅ Define callable node wrappers to avoid ToolNode's message requirement
def run_news_agent(state: NewsState):
    print("[Graph] -> Enter run_news_agent with company_name=", state.company_name)
    result = news_agent_tool.invoke({
        "company_name": state.company_name,
        "news_analyst": state.news_analyst,
        "news_container": state.news_container,
    })
    try:
        na = result.get("news_analyst") if isinstance(result, dict) else None
        print("[Graph] <- Exit run_news_agent; summaries=", 0 if not na else len(na))
    except Exception as e:
        print("[Graph] !! run_news_agent post-invoke logging error:", e)
    return result

def run_news_ui_agent(state: NewsState):
    count = 0 if not state.news_analyst else len(state.news_analyst)
    print(f"[Graph] -> Enter run_news_ui_agent; rendering {count} summaries for {state.company_name}")
    result = news_ui_agent.invoke({
        "company_name": state.company_name,
        "news_analyst": state.news_analyst,
        "news_container": state.news_container,
    })
    print("[Graph] <- Exit run_news_ui_agent")
    return result

def run_reddit_agent(state: NewsState):
    # Reddit tool expects 'Company (TICKER)'; fallback to company only if ticker unknown
    company_and_ticker = state.company_name
    try:
        from re import search
        # If state.company_name already like 'Tesla (TSLA)', keep as is
        if not search(r"\(.*?\)", state.company_name) and state.company_name:
            company_and_ticker = f"{state.company_name} ({state.company_name})"
    except Exception:
        pass

    tool = RedditStockNewsTool()
    print("[Graph] -> Enter run_reddit_agent; query=", company_and_ticker)
    result = tool.invoke({"company_and_ticker": company_and_ticker})
    print("[Graph] .. run_reddit_agent raw result type:", type(result).__name__)
    # Normalize to list[str] for downstream compatibility
    normalized_summary = []
    if isinstance(result, list):
        normalized_summary = [str(item) for item in result]
    elif isinstance(result, str):
        normalized_summary = [result]
    elif result is not None:
        normalized_summary = [str(result)]
    print("[Graph] <- Exit run_reddit_agent; normalized items=", len(normalized_summary))

    # Ensure we return a dict to merge into state
    return {
        "company_name": state.company_name,
        "news_analyst": state.news_analyst,
        "reddit_summary": normalized_summary,
        "news_container": state.news_container,
        "reddit_container": state.reddit_container,
        "ai_container": state.ai_container,
    }

def run_reddit_ai_agent(state: NewsState):
    count = 0 if not state.reddit_summary else len(state.reddit_summary)
    print(f"[Graph] -> Enter run_reddit_ai_agent; posts={count} for {state.company_name}")
    result = reddit_ai_agent.invoke({
        "company_name": state.company_name,
        "reddit_summary": state.reddit_summary or [],
    })
    print("[Graph] <- Exit run_reddit_ai_agent")
    return result


# ✅ Build Graph
graph = StateGraph(NewsState)
graph.add_node("NewsAgent", run_news_agent)
graph.add_node("NewsUIAgent", run_news_ui_agent)
graph.add_node("RedditAgent", run_reddit_agent)
graph.add_node("RedditAIAgent", run_reddit_ai_agent)

graph.add_edge(START, "NewsAgent")
graph.add_edge("NewsAgent", "NewsUIAgent")
graph.add_edge("NewsUIAgent", "RedditAgent")
graph.add_edge("RedditAgent", "RedditAIAgent")
graph.add_edge("RedditAIAgent", END)

# ✅ Compile graph
news_app = graph.compile()

from typing import List, Optional, Any
import streamlit as st
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# ===============================================
# 1. Input Schema
# ===============================================
class NewsUIArgs(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    news_analyst: Optional[List[str]] = Field(default=None, description="List of summarized news articles")
    news_container: Optional[Any] = Field(default=None, description="Streamlit container reference")

    model_config = {"arbitrary_types_allowed": True}


# ===============================================
# 2. Streamlit Display Logic
# ===============================================
def display_news_in_streamlit(company_name: str, news_analyst: Optional[List[str]], container):
    """Display summarized news inside Streamlit container."""
    try:
        print(f"[UI] Rendering news for {company_name}; summaries={(0 if not news_analyst else len(news_analyst))}")
    except Exception as e:
        print("[UI] Logging error:", e)
    
    if news_analyst:
        # Ensure all UI elements are placed inside the provided container
        with container:
            for i, summary in enumerate(news_analyst, start=1):
                if summary and summary.strip():
                    print(f"[UI] -> Draw expander for item {i}; length={len(summary)}")
                    with st.expander(f"🗞️ News {i}"):
                        st.write(summary)
    else:
        container.warning("No recent news summaries found.")
        print("[UI] Warning: No recent news summaries found.")

    return "Displayed Successfully"


# ===============================================
# 3. LangChain Tool Definition
# ===============================================
class NewsUIAgent(BaseTool):
    """Display summarized news in Streamlit UI."""

    name = "news_ui_agent"
    description = "Display summarized company news articles inside Streamlit interface."
    args_schema = NewsUIArgs

    def _run(self, **kwargs):
        """Always handle input as dictionary for LangGraph compatibility."""
        print("🔍 [NewsUIAgent] Received input:", kwargs)

        company_name = kwargs.get("company_name")
        news_analyst = kwargs.get("news_analyst")
        news_container = kwargs.get("news_container")
        print(company_name, news_analyst, news_container)
        if not company_name:
            raise ValueError("❌ No company_name found in input")

        # IMPORTANT: Do NOT render Streamlit UI here (graph may run off main thread).
        # Return data only; the caller should render on the Streamlit main thread.

        # ✅ Always return dict (LangGraph expects it)
        return {
            "company_name": company_name,
            "news_analyst": news_analyst,
            "reddit_summary": None,
            "ai_analysis": None,
            "news_container": news_container,
        }

    async def _arun(self, **kwargs):
        return self._run(**kwargs)


# ✅ Instantiate Tool
news_ui_agent = NewsUIAgent()

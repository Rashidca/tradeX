from typing import List, Optional, Any
import streamlit as st
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field


# ===============================================
# 1. Input Schema
# ===============================================
class RedditUIArgs(BaseModel):
    company_name: str = Field(..., description="Name of the company")
    reddit_summary: Optional[List[str]] = Field(default=None, description="List of summarized reddit posts")
    reddit_container: Optional[Any] = Field(default=None, description="Streamlit container reference")

    model_config = {"arbitrary_types_allowed": True}


# ===============================================
# 2. Streamlit Display Logic
# ===============================================
def display_reddit_in_streamlit(company_name: str, reddit_summary: Optional[List[str]], container):
    """Display summarized reddit posts inside Streamlit container."""
    try:
        print(f"[UI] Rendering reddit for {company_name}; posts={(0 if not reddit_summary else len(reddit_summary))}")
    except Exception as e:
        print("[UI] Logging error:", e)

    if reddit_summary:
        with container:
            for i, post in enumerate(reddit_summary, start=1):
                if post and str(post).strip():
                    with st.expander(f"💬 Reddit Post {i}"):
                        st.write(str(post))
    else:
        container.warning("No Reddit summaries available to display.")
        print("[UI] Warning: No Reddit summaries available to display.")

    return "Displayed Successfully"


# ===============================================
# 3. LangChain Tool Definition (non-rendering)
# ===============================================
class RedditUIAgent(BaseTool):
    """Return reddit display payload for Streamlit UI (no rendering in tool)."""

    name = "reddit_ui_agent"
    description = "Prepare reddit summaries for Streamlit interface."
    args_schema = RedditUIArgs

    def _run(self, **kwargs):
        print("🔍 [RedditUIAgent] Received input:", kwargs)

        company_name = kwargs.get("company_name")
        reddit_summary = kwargs.get("reddit_summary")
        reddit_container = kwargs.get("reddit_container")

        if not company_name:
            raise ValueError("❌ No company_name found in input")

        # IMPORTANT: Do NOT render Streamlit UI here (graph may run off main thread).
        # Return data only; the caller should render on the Streamlit main thread.
        return {
            "company_name": company_name,
            "reddit_summary": reddit_summary,
            "reddit_container": reddit_container,
        }

    async def _arun(self, **kwargs):
        return self._run(**kwargs)


# ✅ Instantiate Tool
reddit_ui_agent = RedditUIAgent()



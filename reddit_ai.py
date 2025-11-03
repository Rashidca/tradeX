# reddit_ai_agent_node.py
import streamlit as st
from langchain_core.tools import BaseTool
from pydantic import BaseModel, Field

# -------------------------------
# Input Schema
# -------------------------------
class RedditAIArgs(BaseModel):
    reddit_summary: list[str] = Field(..., description="Summarized Reddit discussion texts")
    company_name: str = Field(..., description="Company name")

# -------------------------------
# Display Function
# -------------------------------
def display_reddit_summary(company_name: str, reddit_summary: list[str]):
    st.subheader(f"🧵 Reddit Discussions about {company_name}")

    if not reddit_summary:
        st.warning("No Reddit summaries available to display.")
        return "No Reddit posts found."

    for i, post in enumerate(reddit_summary, start=1):
        with st.expander(f"💬 Reddit Post {i}"):
            st.write(post)

    return "Displayed Reddit summaries successfully."

# -------------------------------
# LangGraph Tool
# -------------------------------
class RedditAIAgent(BaseTool):
    name = "reddit_ai_agent"
    description = "Display summarized Reddit discussions in Streamlit"
    args_schema = RedditAIArgs

    def _run(self, company_name: str, reddit_summary: list[str]):
        # Do not render UI inside the tool; return dict for LangGraph state merge
        return {
            "company_name": company_name,
            "reddit_summary": reddit_summary,
        }

    async def _arun(self, company_name: str, reddit_summary: list[str]):
        return {
            "company_name": company_name,
            "reddit_summary": reddit_summary,
        }

reddit_ai_agent = RedditAIAgent()

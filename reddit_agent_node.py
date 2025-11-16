# reddit_agent_node.py

import os
import re
import certifi
import praw
from typing import Type
from transformers import pipeline

# LangGraph imports
from langgraph.graph import StateGraph, END
from langgraph.prebuilt import ToolNode
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq

# ==========================================
# SSL Fix for macOS
# ==========================================
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# ==========================================
# Initialize Models
# ==========================================
print("Initializing Groq LLM...")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
print("Groq LLM initialized ✅")

print("Initializing local T5 summarizer...")
local_summarizer = pipeline("summarization", model="t5-base")
print("Local summarizer initialized ✅")

# ==========================================
# Define Reddit Tool
# ==========================================
class RedditStockNewsTool(BaseTool):
    """Fetch and summarize stock news from Reddit for the last week."""
    name: str = "reddit_stock_news_search"
    description: str = "Get recent stock news from Reddit for a company (last week)."

    class ArgsSchema(BaseModel):
        company_and_ticker: str = Field(description="Company name and ticker, e.g., 'Tesla (TSLA)'")

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(self, company_and_ticker: str) -> str:
        try:
            match = re.search(r"\((.*?)\)", company_and_ticker)
            if not match:
                return "Error: Input must contain a ticker in parentheses, like 'Tesla (TSLA)'."
            ticker = match.group(1).strip()
            company_name = company_and_ticker.split('(')[0].strip()
        except Exception:
            return f"Error parsing input: '{company_and_ticker}'. Use 'CompanyName (TICKER)'."

        print(f"\nFetching posts for {company_name} ({ticker}) from Reddit last week...")

        try:
            reddit = praw.Reddit(
                client_id="[your client id]",
                client_secret="[your client_secret]",
                user_agent="[your user_agent id]"
            )

            stock_subreddits = "stocks+StockMarket+investing"
            search_query = f'("{company_name} stock") OR ({ticker})'
            posts_text = []

            for submission in reddit.subreddit(stock_subreddits).search(
                search_query,
                limit=10,
                sort='relevance',
                time_filter='week'
            ):
                if submission.selftext and len(submission.selftext) > 100:
                    text_without_links = re.sub(r'http\S+', '', submission.title + ". " + submission.selftext)
                    posts_text.append(text_without_links)

            if not posts_text:
                return "No relevant posts were found from the last week."

            full_raw_text = " ".join(posts_text)
            summary_result = local_summarizer(full_raw_text, min_length=50, do_sample=False)
            return summary_result[0]['summary_text']

        except Exception as e:
            return f"Error running Reddit tool: {e}"

# ==========================================
# Build LangGraph Node
# ==========================================
reddit_tool = RedditStockNewsTool()
reddit_node = ToolNode(reddit_tool)

# Create graph that receives company_and_ticker and outputs summary
reddit_graph = StateGraph(dict)

# Add the Reddit node
reddit_graph.add_node("RedditAgent", reddit_node)

# Define edges
reddit_graph.add_edge("__start__", "RedditAgent")
reddit_graph.add_edge("RedditAgent", "__end__")

# Compile Reddit Agent
reddit_agent = reddit_graph.compile()

print("✅ Reddit Agent Node Ready")


# ✅ Optional: standalone quick test
if __name__ == "__main__":
    test_input = {"company_and_ticker": "Tesla (TSLA)"}
    print("Testing Reddit Agent Node...")
    output = reddit_agent.invoke(test_input)
    print("Output:\n", output)

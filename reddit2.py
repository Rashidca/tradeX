import os
import re
from typing import Type

import certifi 
import praw
from transformers import pipeline

# LangChain Imports (updated for Pydantic v2)
from pydantic import BaseModel, Field
from langchain_core.tools import BaseTool
from langchain_groq import ChatGroq
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph

# -----------------------------
# SSL Fix for macOS + Conda
# -----------------------------
os.environ["SSL_CERT_FILE"] = certifi.where()
os.environ["REQUESTS_CA_BUNDLE"] = certifi.where()

# -----------------------------
# Step 0: Set your Groq API Key
# -----------------------------
GROQ_API_KEY = "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"  # Replace with your Groq API key
os.environ["GROQ_API_KEY"] = GROQ_API_KEY

# -----------------------------
# Step 1: Initialize Models
# -----------------------------
print("Initializing Groq LLM...")
llm = ChatGroq(model="llama-3.1-8b-instant", temperature=0)
print("Groq LLM initialized ✅")

print("Initializing local T5 summarizer...")
local_summarizer = pipeline("summarization", model="t5-base")
print("Local summarizer initialized ✅")

# -----------------------------
# Step 2: Define Reddit Tool
# -----------------------------
class RedditStockNewsTool(BaseTool):
    """Fetch and summarize stock news from Reddit for the last week."""
    name: str = "reddit_stock_news_search"
    description: str = "Get recent stock news from Reddit for a company (last week)."

    class ArgsSchema(BaseModel):
        stock: str = Field(description="Company name and ticker, e.g. 'Tesla (TSLA)'")

    args_schema: Type[BaseModel] = ArgsSchema

    def _run(self, company_and_ticker: str) -> str:
        # Extract company and ticker
        try:
            match = re.search(r"\((.*?)\)", company_and_ticker)
            if not match:
                return "Error: Input must contain a ticker in parentheses, like 'Tesla (TSLA)'."
            ticker = match.group(1).strip()
            company_name = company_and_ticker.split('(')[0].strip()
        except Exception:
            return f"Error parsing input: '{company_and_ticker}'. Use 'CompanyName (TICKER)'."

        print(f"\nFetching posts for {company_name} ({ticker}) from Reddit last week...")

        # Initialize PRAW
        try:
            reddit = praw.Reddit(
                client_id="O9Yer2rCexvhCkslsxlT-w",
                client_secret="06CHjc1FVQ3AtiYcXf1n42RyD37YDQ",
                user_agent="TradeX:v1.0 (by u/Agitated-Chair-4187)"
            )

            print("Reddit authentication successful:", reddit.user.me())  # test authentication

            # Search subreddits
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

# -----------------------------
# Step 3: Run the Tool
# -----------------------------
if __name__ == "__main__":
    tool = RedditStockNewsTool()
    company_input = "Tesla(TSLA)"
    summary = tool._run(company_input)

    print("\n--- Reddit Stock News Summary ---")
    print(summary)
# -----------------------------
# Step 3: Wrap as LangGraph Node
# -----------------------------
reddit_tool = RedditStockNewsTool()
reddit_node = ToolNode(
    tools=[reddit_tool],           # <- pass a list (tool instance is fine)
    name="RedditAgent",            # <- give the node a name
)
# Create minimal state machine (if used independently)
reddit_graph = StateGraph(dict)
reddit_graph.add_node("RedditAgent", reddit_node)
reddit_graph.set_entry_point("RedditAgent")
reddit_graph.set_finish_point("RedditAgent")

# Export compiled agent
reddit_agent = reddit_graph.compile()

print("✅ Reddit Agent Node Ready for LangGraph Pipeline")

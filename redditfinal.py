# --- reddit_stock_summary_colab.py ---
import re
import praw
import yfinance as yf
from transformers import pipeline
from langchain_groq import ChatGroq



def get_reddit_stock_summary(data: dict) -> dict:
    """
    Fetch and summarize Reddit stock discussions for a given market dictionary:
    Example Input: {"Marketstate": "TSLA"}
    Output: {"Marketstate": "TSLA", "reddit_news_summary": ["summary text"]}
    """

    # --- Step 1: Extract ticker from dictionary ---
    ticker = data.get("stock")
    if not ticker:
        return {"error": "❌ 'stock' key missing in dictionary."}

    # --- Step 2: Load Groq API Key ---
    # try:
    #     GROQ_API_KEY = userdata.get("GROQ_API_KEY")
    # except userdata.SecretNotFoundError:
    #     return {"error": "❌ GROQ_API_KEY not found in Colab Secrets."}

    # --- Step 3: Get company name using yfinance ---
    try:
        company_info = yf.Ticker(ticker).info
        company_name = company_info.get("longName", ticker)
    except Exception:
        company_name = ticker

    print(f"\n💼 Processing {company_name} ({ticker})")

    # --- Step 4: Initialize Reddit client ---
    reddit = praw.Reddit(
                client_id="[your client id]",
                client_secret="[your client_secret]",
                user_agent="[your user agent id]"
            )

    # --- Step 5: Fetch Reddit posts (mix of 'new' and 'relevant') ---
    subreddits = "stocks+StockMarket+investing"
    search_query = f'("{company_name}" OR "{ticker}")'
    posts_text = []

    try:
        print("🔍 Fetching recent & relevant Reddit posts...")
        combined_results = []
        for sort_type in ["new", "relevance"]:
            results = list(
                reddit.subreddit(subreddits).search(
                    search_query,
                    limit=5,
                    sort=sort_type,
                    time_filter="week"
                )
            )
            combined_results.extend(results)

        for submission in combined_results:
            if submission.selftext and len(submission.selftext) > 100:
                text = re.sub(r"http\S+", "", submission.title + ". " + submission.selftext)
                posts_text.append(text)

        # Backup: If no recent posts found
        if not posts_text:
            print("⚠️ No recent posts found. Fetching older month data...")
            results = list(
                reddit.subreddit(subreddits).search(
                    search_query,
                    limit=7,
                    sort="relevance",
                    time_filter="month"
                )
            )
            for submission in results:
                if submission.selftext and len(submission.selftext) > 100:
                    text = re.sub(r"http\S+", "", submission.title + ". " + submission.selftext)
                    posts_text.append(text)

    except Exception as e:
        data["reddit_news_summary"] = [f"❌ Reddit fetch error: {e}"]
        return data

    if not posts_text:
        data["reddit_news_summary"] = [f"⚠️ No Reddit posts found for {company_name} ({ticker})."]
        return data

    # --- Step 6: Local summarization with T5 ---
    try:
        print("🧠 Pre-summarizing Reddit posts using T5...")
        local_summarizer = pipeline("summarization", model="t5-base")
        combined_text = " ".join(posts_text)
        short_summary = local_summarizer(
            combined_text, min_length=50, max_new_tokens=200, do_sample=False
        )[0]["summary_text"]
    except Exception as e:
        data["reddit_news_summary"] = [f"❌ Local summarization error: {e}"]
        return data

    # --- Step 7: Refine with Groq Llama 3 ---
    try:
        print("🦙 Refining with Llama 3 (Groq)...")
        llm = ChatGroq(
            groq_api_key="[your groq api key]",
            model_name="llama-3.1-8b-instant",
            temperature=0.3,
            
        )

        prompt = f"""
        You are an AI financial analyst.
        Based on the Reddit summary below, write a concise, factual, and sentiment-aware
        overview of the public discussion around {company_name} ({ticker}) stock.
        Keep it helpful for understanding market trends.

        Reddit Summary:
        {short_summary}

        Output only the refined summary.
        """

        response = llm.invoke(prompt)
        data["reddit_summaries"] = [response.content.strip()]
        return data

    except Exception as e:
        data["reddit_summaries"] = [f"❌ Groq API error: {e}"]
        return data


#--- MAIN EXECUTION (Example) ---
# market_data = {"stock": "NVDA"}
# updated_data = get_reddit_stock_summary(market_data)

# print("\n✅ Final Output Dictionary:\n")
# print(updated_data)

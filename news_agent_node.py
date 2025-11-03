from GoogleNews import GoogleNews
from newspaper import Article
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
import re
from transformers import pipeline
from langgraph.prebuilt import ToolNode
from langgraph.graph import StateGraph

# -------------------------------
# Summarization Model
# -------------------------------
summarizer = pipeline(
    "summarization",
    model="t5-small",
    tokenizer="t5-small"
)

# -------------------------------
# Helper: Fetch article text
# -------------------------------
def fetch_article_text(url):
    """Extract readable text from a news URL."""
    try:
        article = Article(url)
        article.download()
        article.parse()
        return article.text.strip()
    except Exception:
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.text, 'html.parser')
            paragraphs = soup.find_all('p')[:3]
            return " ".join([p.get_text(strip=True) for p in paragraphs])
        except:
            return "Could not fetch article text."

# -------------------------------
# Core News Summarizer Tool
# -------------------------------
def news_agent(context: dict):
    """
    Fetch and summarize recent news for a company.
    """
    company_name = context.get("stock", "")
  
    days = 7
    max_results = 5
    text_limit = 2000

    if not company_name:
        raise ValueError("Input dictionary must include 'stcok'.")

    today = datetime.now()
    start_date = today - timedelta(days=days)

    googlenews = GoogleNews(lang='en')
    googlenews.set_time_range(start_date.strftime("%m/%d/%Y"), today.strftime("%m/%d/%Y"))
    googlenews.search(company_name)
    results = googlenews.results(sort=True)[:max_results]

    summaries = []
    for item in results:
        url = item.get("link", "").split("&")[0]
        text = fetch_article_text(url)
        clean_text = re.sub(r'\s+', ' ', text).strip()[:text_limit]

        # Token cleanup
        tokens = summarizer.tokenizer.encode(clean_text, truncation=True, max_length=500, add_special_tokens=False)
        clean_text = summarizer.tokenizer.decode(tokens, skip_special_tokens=True)

        # Summarize
        if len(clean_text.split()) < 30:
            summary = clean_text
        else:
            try:
                summary = summarizer(
                    clean_text,
                    max_new_tokens=80,
                    min_length=25,
                    do_sample=False
                )[0]['summary_text']
            except Exception:
                summary = "Summary generation failed."
        print("Generated summary:", summary)
        summaries.append(summary)
    print(summaries)
    context["news_summaries"] = summaries
    # print("*"*50)
    print(context["news_summaries"])
    # print("context after news_agent:", context)
    return context

# if __name__ == "__main__":
#     context = {"company_name": "Apple"}
#     print(f"🔍 Fetching news for: {context['company_name']} ...\n")

#     result = news_agent(context)

#     print("✅ Received summaries:\n")
#     for i, summary in enumerate(result["news_summaries"], start=1):
#         print(f"{i}. {summary}\n")


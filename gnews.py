from GoogleNews import GoogleNews
from newspaper import Article
from datetime import datetime, timedelta
from bs4 import BeautifulSoup
import requests
from langchain.tools import Tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import initialize_agent, AgentType
import json
import re


def fetch_and_display_news(company_name, days=14, max_results=10, text_limit=2000):
    """Fetch Google News articles for a company and display them with full text."""

    # --- fallback text fetcher ---
    def fetch_article_fallback(url):
        """Fetch article text using requests + BeautifulSoup if newspaper3k fails."""
        try:
            r = requests.get(url, headers={"User-Agent": "Mozilla/5.0"})
            soup = BeautifulSoup(r.text, 'html.parser')
            headings = soup.find_all(['h1', 'h2'])
            if not headings:
              paragraphs = soup.find_all('p')[:3]  # get first 3 paragraphs as fallback
              text = " | ".join([p.get_text(strip=True) for p in paragraphs])
              return f"No headings found. Showing intro instead:\n{text}"

            text = " | ".join([h.get_text(strip=True) for h in headings])
            return text
        except:
            return "Article text not available."

    # --- fetch articles from Google News ---
    today = datetime.now()
    start_date = today - timedelta(days=days)

    googlenews = GoogleNews(lang='en')
    googlenews.set_time_range(start_date.strftime("%m/%d/%Y"), today.strftime("%m/%d/%Y"))
    googlenews.search(company_name)

    results = googlenews.results(sort=True)[:max_results]
    formatted_results = []

    for item in results:
        url = item['link'].split("&")[0]  # clean URL
        text = ""
        try:
            article = Article(url)
            article.download()
            article.parse()
            text = article.text.strip()
        except:
            pass
        if not text:
            text = fetch_article_fallback(url)

        clean_text = re.sub(r'\s+', ' ', text).strip()

        formatted_results.append(
            # #"title": item.get("title"),
            # "media": item.get("media"),
            # #"date": item.get("date"),
            # "link": url,
            text.replace("\n"," ").strip()[:text_limit]
        )

    # --- display articles ---
    if not formatted_results:
        return "No recent news found"

    '''output=""
    for i, article in enumerate(formatted_results, 1):
      output+=(f"\n{i}. 🗞 {article['title']}")
      output+=(f"   📰 Source: {article['media']}")
      output+=(f"   📅 Date: {article['date']}")
      output+=(f"   🔗 Link: {article['link']}")
      output+=(f"   📝 Full Text:\n{article['full_text'][:text_limit]}")
      output+=("\n" + "-"*100)
    return output'''

    return formatted_results


  #langchain tool
    news_tool= Tool(
        name="Fetch company news",
        func= fetch_and_display_news,
        description="fetches relevent news of the given company in the given time range from google news, returns summaries of each article"

    )

    llm = ChatGoogleGenerativeAI(
        model="gemini-2.5-pro",  # Choose the appropriate model
        temperature=0.7
    )

    agent = initialize_agent(
        tools=[news_tool],
        llm=llm,
        agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True
    )
    result = agent.run("Get me the latest news about Apple from the past 7 days.")
    print(result.replace("\n"," ").strip()[:text_limit])
    print("\n"+"-"*100)




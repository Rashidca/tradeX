import streamlit as st
from news_agent_node import news_agent

def display_news_ui(result: dict):
    """
    PURE UI FUNCTION — no LangGraph node.
    Called AFTER news_app.invoke().
    """

    
    summaries = result.get("news_summaries", [])

   

    if not summaries:
        st.info("No news summaries found.")
        return

    for i, summary in enumerate(summaries, start=1):
        with st.expander(f"News {i}"):
            st.write(summary)


# -------------------------
# Standalone execution
# -------------------------

  # ✅ Run the news_agent

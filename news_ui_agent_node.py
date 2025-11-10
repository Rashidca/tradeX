import streamlit as st
from news_agent_node import news_agent

def display_news_ui(MarketState: dict):
    """
    PURE UI FUNCTION — no LangGraph node.
    Called AFTER news_app.invoke().
    """

   
    summaries = MarketState.get("news_summaries")

    print(summaries)
    news_container = MarketState.get("news_container")
    if news_container:
        with news_container:
            for i, summary in enumerate(summaries, start=1):
                with st.expander(f"News {i}"):
                    st.write(summary)
    else:
        st.info("No news container found.")
        #return
    # if not summaries:
    #     st.info("No news summaries found.")
    #     return

    # for i, summary in enumerate(summaries, start=1):
    #     with st.expander(f"News {i}"):
    #         st.write(summary)
    # print(result["news_summaries"])
    
    return MarketState
# -------------------------
# Standalone execution
# -------------------------
# if __name__ == "__main__":
#     result = {"news_summaries": ["Apple is a good company", "Apple is a bad company"]}
#     display_news_ui(result)
  # ✅ Run the news_agent

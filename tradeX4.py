import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ticker_dataset import get_stock_data, get_priority_values
from tuned_function import forecast_aapl_cv, plot_forecast
from reddit_agent_node import reddit_agent

# LangGraph imports for news agent
from news_agent_node import news_agent, ToolNode, StateGraph

# ==============================
# File path for storing CSV
# ==============================
filepath = "/Users/afrahanas/Downloads/project3/AAPL_all_data.csv"

# ==============================
# Streamlit Setup
# ==============================
if "stock_data_dict" not in st.session_state:
    st.session_state["stock_data_dict"] = {}

st.set_page_config(page_title="TradeX", page_icon="💹", layout="centered")

st.title("💹 TradeX")
st.write("Intelligent platform for stock forecasting & news summarization")

st.subheader("📊 Input Details")

# --------------------------
# User Inputs
# --------------------------
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "")
action = st.selectbox("Select Action:", ["Buy", "Sell", "Hold"])
num_stocks = st.number_input("Enter Number of Stocks:", min_value=1, step=1)

# --------------------------
# Proceed Button
# --------------------------
if st.button("🚀 Proceed"):

    if ticker.strip() == "":
        st.warning("⚠️ Please enter a valid stock ticker.")
    else:

        # --------------------------
        # Fetch stock data & compute matrix
        # --------------------------
        with st.spinner(f"Fetching latest data for {ticker.upper()}..."):
            df = get_stock_data(ticker.upper())
            matrix = get_priority_values(df)

        st.success(f"✅ TradeX Started for {ticker.upper()}")
        st.write(f"**Action:** {action}")
        st.write(f"**Number of Stocks:** {num_stocks}")
        st.write("---")

        # Flatten dataframe columns for Prophet
        df = df.reset_index()
        new_cols = []
        for col in df.columns:
            if col[0].lower() == 'date':
                new_cols.append('Date')
            elif 'close' in col[0].lower():
                new_cols.append('Price')
            else:
                new_cols.append(f"{col[0]}_{col[1]}" if col[1] else col[0])
        df.columns = new_cols
        df.to_csv(filepath, index=False)

        # Display current matrix
        st.subheader("📊 Current Matrix")
        st.dataframe(pd.DataFrame([matrix]))

        # --------------------------
        # Forecasting
        # --------------------------
        metrics_placeholder = st.empty()
        graph_placeholder = st.empty()

        with st.spinner("⏳ Running Forecasting Model..."):
            model, forecast_full, results_df, final_metrics = forecast_aapl_cv(filepath, periods=365)

            # Display metrics
            with metrics_placeholder.container():
                st.subheader("📈 Forecast Metrics")
                st.write(final_metrics)

            # Plot forecast
            st.subheader("📈 Tuned Forecast Plot")
            with st.container():
                fig = plot_forecast(forecast_full)
                st.pyplot(fig)

        # --------------------------
        # News Summarization via LangGraph
        # --------------------------
        st.write("---")
        st.subheader(f"📰 Latest News Summaries for {ticker.upper()}")

        try:
            with st.spinner(f"Fetching news for {ticker.upper()}..."):
        # Directly call the news_agent function
                context = {"company_name": ticker.upper()}
                result = news_agent(context)  # <- call the function directly
                summaries = result.get("news_analyst", [])

        # Display summaries
                if summaries:
                    for i, summary in enumerate(summaries, start=1):
                        with st.expander(f"🗞️ News {i}"):
                            st.write(summary)
                else:
                    st.warning("No recent news summaries found.")

        except Exception as e:
             st.error(f"❌ Error fetching news: {e}")
        # --------------------------
            # Reddit Summarization via LangGraph
        # --------------------------
        st.write("---")
        st.subheader(f"🧵 Reddit Discussions for {ticker.upper()}")

        try:
             with st.spinner(f"Fetching Reddit discussions for {ticker.upper()}..."):
                context = {"company_and_ticker": f"{ticker.upper()} ({ticker.upper()})"}
                result = reddit_agent(context)
                reddit_summary = result.get("reddit_analyst", [])

                if reddit_summary:
                    st.success("✅ Reddit summary generated successfully!")
                    st.write(reddit_summary)
                else:
                    st.warning("No Reddit summary available for this stock.")

        except Exception as e:
            st.error(f"❌ Error fetching Reddit data: {e}")


import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ticker_dataset import get_stock_data, get_priority_values
from tuned_function import forecast_aapl_cv, plot_forecast, calculate_future_averages

# ✅ LangGraph imports
from news_agent_node import news_agent


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
        # Fetch stock data
        # --------------------------
        with st.spinner(f"Fetching latest data for {ticker.upper()}..."):
            df = get_stock_data(ticker.upper())

        if df is None or df.empty:
            st.error("❌ No data returned for this ticker. Please check the symbol or try again later.")
        else:
            # --------------------------
            # Compute Matrix
            # --------------------------
            matrix = get_priority_values(df)
            st.success(f"✅ TradeX Started for {ticker.upper()}")
            st.write(f"**Action:** {action}")
            st.write(f"**Number of Stocks:** {num_stocks}")
            st.write("---")

            # Normalize & save CSV for Prophet
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

            st.subheader("📊 Current Matrix")
            st.dataframe(pd.DataFrame([matrix]))

      
          # --------------------------
# News Summarization (NO LangGraph)
# --------------------------
            st.write("---")
            st.subheader(f"📰 Latest News Summaries for {ticker.upper()}")

            with st.spinner("Fetching & summarizing latest news..."):
                try:
                    # ✅ Prepare input for your summarizer
                    context = {"company_name": ticker.upper()}

                    # ✅ Call your summarizer
                    result = news_agent(context)

                    # ✅ Extract summaries
                    summaries = result.get("news_analyst", [])

                    if not summaries:
                        st.error("❌ No news summaries found.")
                    else:
                        st.success("✅ Latest News Summaries:")

                        # ✅ Expander for each news
                        for i, summary in enumerate(summaries, start=1):
                            with st.expander(f"📰 News {i}", expanded=False):
                                st.write(summary)

                except Exception as e:
                    st.error(f"❌ Error fetching or summarizing news: {e}")

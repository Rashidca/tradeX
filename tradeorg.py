import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt

from ticker_dataset import get_stock_data, get_priority_values
from tuned_function import forecast_aapl_cv, plot_forecast, calculate_future_averages
from news_agent_node import news_agent
# LangGraph imports for news agent
from graph_lang import news_app

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
            # metrics_placeholder = st.empty()
            # graph_placeholder = st.empty()
            # predicted_placeholder = st.empty()
            # result_placeholder = st.empty()

            # with st.spinner("⏳ Running Forecasting Model..."):
            #     try:
            #         model, results_df, forecast_full, final_metrics = forecast_aapl_cv(filepath, periods=365)
            #         st.success("✅ Forecasting Completed")

            #         results_avg = calculate_future_averages(forecast_full)

            #         with result_placeholder.container():
            #             st.subheader("📊 Forecast Results (Test Set)")
            #             st.dataframe(results_df)

            #         with result_placeholder.container():
            #             st.subheader("📊 Future Averages")
            #             st.write(results_avg)

            #         with predicted_placeholder.container():
            #             st.subheader("📈 Forecast Predictions")
            #             st.write(forecast_full)

            #         with metrics_placeholder.container():
            #             st.subheader("📈 Forecast Metrics")
            #             st.write(final_metrics)

            #         st.subheader("📈 Tuned Forecast Plot")
            #         with st.container():
            #             fig = plot_forecast(forecast_full)
            #             st.pyplot(fig)

            #     except Exception as e:
            #         st.error(f"❌ Error during forecasting: {e}")

            # --------------------------
            # News Summarization via LangGraph
            # --------------------------
            st.write("---")
            st.subheader(f"📰 Latest News Summaries for {ticker.upper()}")
            news_container = st.container()
            try:
                with st.spinner(f"Fetching and displaying news for {ticker.upper()} via LangGraph..."):
        # The dictionary containing all state information
                    initial_state = {
            "company_name": ticker.upper(), 
            "news_container": news_container
                }
        
        # 🌟 CRITICAL FIX: Directly invoke with the dictionary (standard LangGraph state initialization)
        # If the graph still errors, the problem is deeper in the node definitions.
                news_app.invoke(initial_state) 
        
        # Alternative (if the above fails, try passing as a message list, though this is less common for state graphs):
        # news_app.invoke([initial_state]) 

            except Exception as e:
            # Use the debugging print from the NewsUIAgent to see what the state looks like
                print(f"DEBUG: Exception during LangGraph invoke: {e}")
                st.error(f"❌ Error fetching or displaying news: {e}")
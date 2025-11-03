import streamlit as st
import json
import pandas as pd
import matplotlib.pyplot as plt

from ticker_dataset import get_stock_data, get_priority_values
from tuned_function import forecast_aapl_cv, plot_forecast, calculate_future_averages
from hybrid_model import forecast_aapl_hybrid_fixed,smape,flatten_dataframe_columns,summarize_future_prices

# LangGraph imports for news agent
from langraphnew import news_app
from news_ui import display_news_in_streamlit
from reddit_ui import display_reddit_in_streamlit

# ==============================
# File path for storing CSV
# ==============================
filepath = "/Users/afrahanas/Downloads/project3/AAPL_all_data.csv"
content_dict = {
  "stock": "",
  "user_action": "",
  "num_stocks": "",
  "current_value": {
    "price": "",
    "volume": "",
    "rsi": "",
    "macd": "",
    "macd_signal": "",
    "sma_50": "",
    "atr_14": "",
    "pe_ratio": "",
    "eps": "",
    "upcoming_earnings": ""
  },
  "metrics": {
    "MAE": "",
    "RMSE": "",
    "MAPE": "",
    "SMAPE": ""
  },
  "future_averages": {
    "Next Day Price": "",
    "Next Week Avg": "",
    "Next Month Avg": "",
    "Next Quarter Avg": "",
    "Next Half-Year Avg": "",
    "Next Year Avg": ""
  },
  "news_summaries": [],
  "reddit_summaries": [],
  "researcher": {
    "bull": [],
    "bear": [],
    "facilitator": [],
    "round": 0
  },
  "risk_debate": {
    "positive": [],
    "negative": [],
    "facilitator": [],
    "round": 0
  },
  "strategist": "",
  "news_container": None,
  "reddit_container": None,
  "researcher_round1_container": None,
  "researcher_round2_container": None,
  "researcher_round3_container": None,
  "risk_analysis_round1_container": None,
  "risk_analysis_round2_container": None,
  "risk_analysis_round3_container": None,
  "strategy_container": None
}

 

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
    content_dict["stock"] = ticker.upper()
    content_dict["user_action"] = action
    content_dict["num_stocks"] = num_stocks

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
            content_dict["current_value"] = matrix

            # Flatten dataframe columns for Prophet
            df = flatten_dataframe_columns(df)
            df = df.iloc[2:] if len(df) > 2 else df
            # Convert all numeric columns to float64 (excluding Date)
            for col in df.columns:
                if 'date' not in col.lower():
                    df[col] = pd.to_numeric(df[col], errors='coerce')
 



            # Display current matrix
            st.subheader("📊 Current Matrix")
            st.dataframe(pd.DataFrame([matrix]))

            # --------------------------
            # Forecasting
            # --------------------------
            metrics_placeholder = st.empty()
            graph_placeholder = st.empty()
            # predicted_placeholder = st.empty()
            result_placeholder = st.empty()
            


            with st.spinner("⏳ Running Forecasting Model..."):
                try:
                    model_prophet, model_xgb, forecast_full, results_df, metrics, forecast_future = forecast_aapl_hybrid_fixed(df)
                    st.success("✅ Forecasting Completed")
                    

                    results_avg = summarize_future_prices(forecast_future)
                    final_metrics=metrics["Hybrid"]
                    print(final_metrics)
                    print(results_avg)

                    # with result_placeholder.container():
                    #     st.subheader("📊 Forecast Results (Test Set)")
                    #     st.dataframe(results_df)

                    with result_placeholder.container():
                        st.subheader("📊 Future Averages")
                        st.write(results_avg)

                    # with predicted_placeholder.container():
                    #     st.subheader("📈 Forecast Predictions")
                    #     st.write(forecast_full)

                    with metrics_placeholder.container():
                        st.subheader("📈 Forecast Metrics")
                        st.write(final_metrics)

                    st.subheader("📈 Tuned Forecast Plot")
                    with st.container():
                        fig = plot_forecast(forecast_full)
                        st.pyplot(fig)

                except Exception as e:
                    st.error(f"❌ Error during forecasting: {e}")
                else:
                    # Update content_dict only if forecasting succeeded
                    content_dict["metrics"] = final_metrics
                    content_dict["future_averages"] = results_avg
                        
                        
            # --------------------------
            # News + Reddit via LangGraph
            # --------------------------
            st.write("---")
            st.subheader(f"📰 Latest News Summaries for {ticker.upper()}")
            news_container = st.container()
            reddit_container = st.container()
            researcher_round1_container = st.container()
            researcher_round2_container = st.container()
            researcher_round3_container = st.container()
            risk_analysis_round1_container = st.container()
            risk_analysis_round2_container = st.container()
            risk_analysis_round3_container = st.container()
            strategy_container = st.container()
            content_dict["news_container"] = news_container
            content_dict["reddit_container"] = reddit_container
            content_dict["researcher_round1_container"] = researcher_round1_container
            content_dict["researcher_round2_container"] = researcher_round2_container
            content_dict["researcher_round3_container"] = researcher_round3_container
            content_dict["risk_analysis_round1_container"] = risk_analysis_round1_container
            content_dict["risk_analysis_round2_container"] = risk_analysis_round2_container
            content_dict["risk_analysis_round3_container"] = risk_analysis_round3_container
            content_dict["strategy_container"] = strategy_container
            try:
                with st.spinner(f"Fetching news and Reddit for {ticker.upper()} via LangGraph..."):
                    # context = {"company_name": ticker.upper(), "news_container": news_container, "reddit_container": reddit_container, "news_analyst": None}
                    # print(context)
                    result = news_app.invoke(content_dict)
                    # Render on main thread using returned data
                    if isinstance(result, dict):
                        summaries = result.get("news_analyst") or []
                        reddit_summaries = result.get("reddit_summary") or []
                    else:
                        try:
                            summaries = result.get("news_analyst") or []  # type: ignore[attr-defined]
                            reddit_summaries = result.get("reddit_summary") or []  # type: ignore[attr-defined]
                        except Exception:
                            summaries = []
                            reddit_summaries = []
                    # Update content_dict with news and reddit
                    content_dict["news_summaries"] = summaries
                    content_dict["reddit_summaries"] = reddit_summaries
                    display_news_in_streamlit(ticker.upper(), summaries, news_container)
                    display_reddit_in_streamlit(ticker.upper(), reddit_summaries, reddit_container)
                    
            except Exception as e:
                st.error(f"❌ Error fetching or displaying news: {e}")
            # Mirror for backward-compatibility and print schema to terminal
            content_dic = content_dict
            try:
                print("[CONTENT_DICT]", json.dumps(content_dict, indent=2, default=str))
            except Exception as e:
                print("[CONTENT_DICT] JSON serialization error:", e)

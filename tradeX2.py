import streamlit as st
from ticker_dataset import get_stock_data, get_priority_values
import pandas as pd 
from tuned_function import forecast_aapl_cv
import matplotlib.pyplot as plt

# ==============================
# TradeX - Stock Forecasting App
# ==============================
filepath="/Users/afrahanas/Downloads/project3/AAPL_all_data.csv"
if "stock_data_dict" not in st.session_state:
    st.session_state["stock_data_dict"] = {}

st.set_page_config(page_title="TradeX", page_icon="💹", layout="centered")

st.title("💹 TradeX")
st.write("An intelligent platform for stock analysis and forecasting")

st.subheader("📊 Input Details")

# --- User Inputs ---
ticker = st.text_input("Enter Stock Ticker (e.g., AAPL, TSLA, MSFT):", "")
action = st.selectbox("Select Action:", ["Buy", "Sell", "Hold"])
num_stocks = st.number_input("Enter Number of Stocks:", min_value=1, step=1)

# ==============================
# Proceed button logic
# ==============================
if st.button("🚀 Proceed"):
    if ticker.strip() == "":
        st.warning("⚠️ Please enter a valid stock ticker.")
    else:
        with st.spinner(f"Fetching latest data for {ticker.upper()}..."):
            df = get_stock_data(ticker.upper())
            matrix = get_priority_values(df)

        st.success(f"✅ TradeX Started for {ticker.upper()}")
        st.write(f"**Action:** {action}")
        st.write(f"**Number of Stocks:** {num_stocks}")
        st.write("---")

        # Flatten columns
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

        # --- Display latest metrics ---
        st.subheader("📊 Current Matrix")
        st.dataframe(pd.DataFrame([matrix]))

        # --- Placeholders for Prophet outputs ---
        metrics_placeholder = st.empty()
        graph_placeholder = st.empty()

        with st.spinner("⏳ Running Forecasting Model..."):
            model, forecast_full, results_df, final_metrics = forecast_aapl_cv(filepath, periods=365)

            # ✅ Display final metrics
            with metrics_placeholder.container():
                st.subheader("📈 Forecast Metrics")
                st.write(final_metrics)

            # ✅ Plot forecast vs actuals
            with graph_placeholder.container():
                st.subheader("📊 Forecast Graph")
                fig, ax = plt.subplots(figsize=(12,6))
                ax.plot(results_df['ds'], results_df['y'], label='Actual', color='blue')
                ax.plot(results_df['ds'], results_df['yhat'], label='Predicted', color='orange')
                ax.plot(forecast_full['ds'], forecast_full['yhat'], label='Forecast', color='green', linestyle='--')
                ax.set_xlabel('Date')
                ax.set_ylabel('Price')
                ax.set_title(f'{ticker.upper()} Forecast')
                ax.legend()
                ax.grid(True)
                st.pyplot(fig)

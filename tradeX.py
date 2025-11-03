import streamlit as st
from ticker_dataset import get_stock_data, get_priority_values  # import your functions
import pandas as pd 
# ==============================
# TradeX - Stock Forecasting App
# ==============================

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
            # ✅ Fetch dataset dynamically based on user ticker
            df = get_stock_data(ticker.upper())

            # ✅ Get latest row of important metrics
            matrix = get_priority_values(df)
        
            # --- Store updates into dictionary ---
            # st.session_state["stock_data_dict"][ticker.upper()] = {
            #     "Ticker": ticker.upper(),
            #     "Action": action,
            #     "Num_Stocks": num_stocks,
            #     "currentMatrix": matrix
            # }
        st.success(f"✅ TradeX Started for {ticker.upper()}")
        st.write(f"**Action:** {action}")
        st.write(f"**Number of Stocks:** {num_stocks}")
        st.write("---")
        
        df.to_csv('/Users/afrahanas/Downloads/project3/AAPL_all_data.csv')  # If my_data.csv exists, it will be replaced


        # --- Placeholder for output ---
        currentmatrix = st.empty()

        # ✅ Display latest metrics in a table
        with currentmatrix.container():
            st.subheader("📊 Current Matrix")
            st.dataframe(pd.DataFrame([matrix]))
        data={"stock": ticker, "action": action, "num_stocks": num_stocks, "currentMatrix": matrix}
        print(data)
        print(df.columns)
import streamlit as st


# dummy_state = {
#     "stock": "AAPL",
#     "num_stocks": 10,
#     "user_action": "Buy",

#     # ✅ Current metrics (anything you want)
#     "current_value": {
#         "price": 185.32,
#         "volume": 51234000,
#         "rsi": 62.4,
#         "macd": 1.28,
#         "macd_signal": 0.93
#     },

#     # ✅ Predicted averages
#     "future_averages": {
#         "7_day": 188.5,
#         "14_day": 191.3,
#         "30_day": 197.8
#     },

#     # ✅ Model performance
#     "metrics": {
#         "mse": 1.24,
#         "mae": 0.88,
#         "rmse": 1.11
#     },

#     # ✅ News summary
#     "news_summaries": [
#         "Apple reports strong iPhone sales beating expectations.",
#         "Apple invests heavily in AI chips, boosting investor confidence."
#     ],

#     # ✅ Reddit sentiment
#     "reddit_summaries": [
#         "Community sentiment is moderately positive.",
#         "Many traders expect AAPL to rise short-term."
#     ],

#     # ✅ Risk Debate Storage
#     "risk_debate": {
#         "positive": [
#             "Positive Round 1: AAPL shows strong revenue momentum and investor sentiment is rising."
#         ],
#         "negative": [
#             "Negative Round 1: Market volatility remains high, and rising rates may impact tech valuations."
#         ],
#         "facilitator": [
#             "Facilitator Round 1: Both agents give evidence, but Positive agent uses more relevant metrics."
#         ],
#         "round": 1
#     },

#     # ✅ UI Containers (Streamlit placeholders)
#     # Create these in Streamlit as:
#     # state["risk_analysis_round1_container"] = st.container()
#     "risk_analysis_round1_container": None,
#     "risk_analysis_round2_container": None,
#     "risk_analysis_round3_container": None
# }

def display_risk_analysis_rounds(state: dict):
    """
    Display Positive, Negative, Facilitator outputs for each risk analysis round
    using the container placeholders stored inside the same state dictionary.
    """

    risk = state.get("risk_debate", {})

    positive_list = risk.get("positive", [])
    negative_list = risk.get("negative", [])
    facilitator_list = risk.get("facilitator", [])

    total_rounds = len(positive_list)

    # ✅ Loop through rounds
    for i in range(total_rounds):

        # ✅ Your container keys follow this pattern:
        # risk_analysis_round1_container, risk_analysis_round2_container, ...
        container_key = f"risk_analysis_round{i+1}_container"
        container = state.get(container_key)

        if not container:
            continue  # Skip if container not found

        with container:
            st.subheader(f"⚖️ Risk Analysis – Round {i + 1}")

            st.markdown("### 🟢 Positive Risk Agent (Optimistic)")
            st.write(positive_list[i] if i < len(positive_list) else "No positive risk data")

            st.markdown("### 🔴 Negative Risk Agent (Cautious)")
            st.write(negative_list[i] if i < len(negative_list) else "No negative risk data")

            st.markdown("### 🟡 Facilitator Summary")
            st.write(facilitator_list[i] if i < len(facilitator_list) else "No summary available")

    return state
# if __name__ == "__main__":
#     display_risk_analysis_rounds(dummy_state)


def display_risk_analysis_rounds_withexpander(state: dict):
    """
    Display Positive, Negative, Facilitator outputs for each risk analysis round
    using the container placeholders stored inside the same state dictionary.
    """

    risk = state.get("risk_debate", {})

    positive_list = risk.get("positive", [])
    negative_list = risk.get("negative", [])
    facilitator_list = risk.get("facilitator", [])

    total_rounds = len(positive_list)

    # ✅ Loop through rounds
    for i in range(total_rounds):

        # ✅ Your container keys follow this pattern:
        # risk_analysis_round1_container, risk_analysis_round2_container, ...
        container_key = f"risk_analysis_round{i+1}_container"
        container = state.get(container_key)

        if not container:
            continue  # Skip if container not found

        with container:
            with st.expander(f"⚖️ Risk Analysis – Round {i + 1}"):
            

                st.markdown("### 🟢 Positive Risk Agent (Optimistic)")
                st.write(positive_list[i] if i < len(positive_list) else "No positive risk data")

                st.markdown("### 🔴 Negative Risk Agent (Cautious)")
                st.write(negative_list[i] if i < len(negative_list) else "No negative risk data")

                st.markdown("### 🟡 Facilitator Summary")
                st.write(facilitator_list[i] if i < len(facilitator_list) else "No summary available")

    return state
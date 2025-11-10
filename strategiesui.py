import streamlit as st

def display_strategist_result(state: dict):
    """
    Display the final strategist recommendation inside
    the Streamlit container stored in the state dictionary.
    """

    strategist_text = state.get("strategist_result", "No strategist result available.")
    container = state.get("strategy_container")

    if not container:
        st.error("Strategy container not found in state.")
        return state

    with container:
        st.subheader("📘 Final Strategy Recommendation")
        st.write(strategist_text)

    return state

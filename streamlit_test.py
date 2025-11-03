import streamlit as st

st.title("✅ Streamlit is Working!")
st.write("Hello! This is a basic Streamlit app running in your environment.")

# A simple input + output
name = st.text_input("Enter your name:")
if name:
    st.success(f"Welcome, {name}!")

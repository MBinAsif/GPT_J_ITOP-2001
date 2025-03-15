import streamlit as st
import requests

st.title("💬 Income Tax Ordinance AI")

query = st.text_input("Ask your tax-related question:")
if st.button("Get Answer"):
    response = requests.get(f"http://localhost:8000/query?query={query}")
    st.write(response.json()["response"])

# chatbot_ui.py
import streamlit as st
import requests

st.title("💬 Income Tax Ordinance AI")

query = st.text_input("Ask your tax-related question:")
if st.button("Get Answer"):
    response = requests.get(f"http://localhost:8000/query?query={query}")
    answer = response.json()["response"]
    
    # For example, if the answer starts with "Summary:" followed by details.
    if "Summary:" in answer:
        summary_part, detail_part = answer.split("Summary:", 1)
        # Further split to isolate the bullet points if needed:
        if "\n\n" in detail_part:
            summary, details = detail_part.split("\n\n", 1)
        else:
            summary, details = detail_part, ""
        
        st.markdown("### Summary")
        st.markdown(summary)
        
        with st.expander("Show Detailed Explanation"):
            st.write(details)
    else:
        # Fallback: Display entire answer
        st.write(answer)

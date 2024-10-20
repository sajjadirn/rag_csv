import streamlit as st
import requests
import json

# FastAPI Backend URL
API_URL = "http://localhost:8000"

st.markdown("""
    <h1>Survey Analysis using LLMs 
        <img src="https://cdn.prod.website-files.com/6605d052b6285185135a565d/661ec0e12e8cd7d447aaad95_Bounce%20Insights%20Logo.svg" alt="Llama2" >
    </h1>
    """, unsafe_allow_html=True)



user_input = st.text_input("Ask a question:")

if st.button("Send Query"):
    if user_input:
        query_payload = {
            "question": user_input,
            "chat_history": []
        }
        response = requests.post(f"{API_URL}/ask/", json=query_payload)
        
        if response.status_code == 200:
            result = response.json()
            st.write(f"Answer: {result['answer']}")
        else:
            st.error("Error in querying the backend.")
    else:
        st.error("Please enter a question.")

# rag_csv_example
A RAG system which reads a csv file and lets the user ask questions about the csv file, uses fastapi and streamlit to achieve this 

# How to run on localhost

Type the following commandas in order


python -m venv .venv 


.venv\Scripts\activate


pip install -r .\requirements.txt


curl https://huggingface.co/TheBloke/Llama-2-7B-Chat-GGML/resolve/main/llama-2-7b-chat.ggmlv3.q8_0.bin


uvicorn backend:app --reload


streamlit run frontend.py
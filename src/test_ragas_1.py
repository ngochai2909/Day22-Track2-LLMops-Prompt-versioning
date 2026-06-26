import os
import asyncio
from datasets import Dataset
from ragas import evaluate
from ragas.metrics import faithfulness
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

os.environ["GOOGLE_API_KEY"] = os.environ.get("GOOGLE_API_KEY", "") # Ensure set

llm = ChatGoogleGenerativeAI(model="gemini-1.5-flash")
emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004")

ds = Dataset.from_dict({
    "question": ["What is AI?"],
    "answer": ["AI is Artificial Intelligence."],
    "contexts": [["Artificial Intelligence (AI) is a field of science."]],
})

result = evaluate(ds, metrics=[faithfulness], llm=llm, embeddings=emb)
print(result)

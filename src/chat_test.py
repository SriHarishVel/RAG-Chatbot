from src.graph import app

question = "What are the key benefits of Agentic AI?"

result = app.invoke({
    "question": question,
    "context": "",
    "answer": "",
    "score": 0.0
})

print("\nQuestion:")
print(question)

print("\nAnswer:")
print(result["answer"])

print("\nSimilarity score:")
print(result["score"])

print("\nRetrieved context:")
print(result["context"][:1000])
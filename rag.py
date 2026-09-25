
from pathlib import Path
import json
import faiss
from sentence_transformers import SentenceTransformer

from config import get_client, MODEL_NAME

PROJECT_DIR = Path(__file__).resolve().parent
VECTOR_DB_DIR = PROJECT_DIR / "vector_store"

index = faiss.read_index(
    str(VECTOR_DB_DIR / "research_index.faiss")
)

with open(VECTOR_DB_DIR / "chunks.json", "r", encoding="utf-8") as f:
    all_chunks = json.load(f)

embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
client = get_client()


def retrieve_chunks(query, top_k=5):
    query_embedding = embedding_model.encode(
        [query],
        convert_to_numpy=True
    ).astype("float32")

    faiss.normalize_L2(query_embedding)

    scores, indices = index.search(query_embedding, top_k)

    results = []

    for score, idx in zip(scores[0], indices[0]):
        chunk = all_chunks[idx]

        results.append({
            "source": chunk["source"],
            "page": chunk["page"],
            "text": chunk["text"],
            "score": float(score)
        })

    return results


def ask_groq(prompt):
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )

    return response.choices[0].message.content


def generate_answer(question, top_k=5):
    results = retrieve_chunks(question, top_k)

    context = "\n\n".join(
        f"[Source: {r['source']} | Page: {r['page']}]\n{r['text']}"
        for r in results
    )

    prompt = f"""
You are an AI Research Assistant.

Answer the question using ONLY the research paper context.

Do not invent facts.
If the answer is not available in the context, say:
"The answer is not available in the provided research paper."

QUESTION:
{question}

RESEARCH PAPER CONTEXT:
{context}
"""

    return ask_groq(prompt), results


def explain_simply(topic):
    results = retrieve_chunks(topic, top_k=5)

    context = "\n\n".join(
        result["text"] for result in results
    )

    prompt = f"""
Explain the following topic simply using ONLY the
provided research paper context.

TOPIC:
{topic}

CONTEXT:
{context}

Avoid unnecessary jargon and do not invent information.
"""

    return ask_groq(prompt), results


def extract_key_findings():
    results = retrieve_chunks(
        "main findings results conclusions",
        top_k=8
    )

    context = "\n\n".join(
        result["text"] for result in results
    )

    prompt = f"""
Extract the key findings and important conclusions
from the research paper context.

Use ONLY the provided context.
Return the findings as a numbered list.
Do not invent information.

CONTEXT:
{context}
"""

    return ask_groq(prompt), results

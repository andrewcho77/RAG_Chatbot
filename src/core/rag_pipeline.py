import openai
import json
from typing import List
from embeddings import fetch_openai_embeddings
from pinecone_client import PineconeClient


def query_pinecone(query: str, top_k: int) -> List:
    """Fetch top_k matches from Pinecone based on query embedding."""
    pinecone_client = PineconeClient()
    pinecone_client.initialize_index()

    query_embedding = fetch_openai_embeddings([query])
    matches = pinecone_client.query(query_embedding, top_k)

    return matches


def format_context(matches: List) -> str:
    """Extracts relevant text from Pinecone matches."""
    return "\n".join(match["metadata"]["text"] for match in matches)


def generate_answer(question: str, context: str) -> str:
    """Generate an answer using OpenAI's chat completions model."""
    user_prompt = f"""
    UFC Fight Stats:
    {context}

    Question: {question}
    """

    response = openai.chat.completions.create(
        model="gpt-4o",
        messages=[
            {
                "role": "system",
                "content": """You are an expert in UFC fight statistics. 
             Use the following UFC fight stats to answer the question in a conversational tone. The
             provided data is per round and per fighter. In order to give total fight analysis, you would need to
             aggregate across rounds. If all of the rounds are not available, don't hallucinate and give responses that
             you can't back up with facts. You can do a best guess aggregation by looking at the EVENT, BOUT, FIGHTER, and ROUND.
             """,
            },
            {"role": "user", "content": user_prompt},
        ],
        max_tokens=300,
    )

    return response.choices

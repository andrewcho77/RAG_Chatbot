import argparse
from src import query_pinecone, format_context, generate_answer, PineconeClient


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run a UFC RAG query")
    parser.add_argument("query", type=str, help="The question to ask the RAG system.")

    args = parser.parse_args()

    question = args.query
    print(f"\n Querying: {question}")

    pinecone_client = PineconeClient()
    matches = query_pinecone(question, top_k=10)
    context = format_context(matches)
    answer = generate_answer(question, context)

    print("\n Answer:")
    print(answer)

    print("\n Context Retrieved:")
    print(context)

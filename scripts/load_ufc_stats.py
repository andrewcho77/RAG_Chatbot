import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv

load_dotenv()

INDEX_NAME = "ufc-stats"
OPENAI_EMBEDDINGS_DIMENSION = 1536  # dimension of OpenAI's text-embedding-ada-002

pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))

cloud = os.environ.get("PINECONE_CLOUD") or "aws"
region = os.environ.get("PINECONE_REGION") or "us-east-1"
spec = ServerlessSpec(cloud=cloud, region=region)


if INDEX_NAME not in pc.list_indexes().names():
    pc.create_index(
        name=INDEX_NAME,
        dimension=OPENAI_EMBEDDINGS_DIMENSION,
        metric="cosine",
        spec=spec,
    )

print(pc)

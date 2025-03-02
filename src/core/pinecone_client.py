from pinecone import Pinecone, ServerlessSpec
from tqdm import tqdm
from src.core.config import (
    PINECONE_API_KEY,
    PINECONE_CLOUD,
    PINECONE_REGION,
    INDEX_NAME,
    OPENAI_EMBEDDINGS_DIMENSION,
)
from typing import List
from src.core.data_loader import save_embeddings_checkpoint


class PineconeClient:
    def __init__(self):
        self.pc = Pinecone(api_key=PINECONE_API_KEY)
        self.index = None

    def initialize_index(self):
        """Initialize Pinecone index if it doesn't exist."""
        spec = ServerlessSpec(cloud=PINECONE_CLOUD, region=PINECONE_REGION)

        if INDEX_NAME not in self.pc.list_indexes().names():
            self.pc.create_index(
                name=INDEX_NAME,
                dimension=OPENAI_EMBEDDINGS_DIMENSION,
                metric="cosine",
                spec=spec,
            )

        self.index = self.pc.Index(INDEX_NAME)

    def query(self, query_embedding: str, top_k: int) -> List:
        """Query the Pinecone index with embeddings."""
        if not self.index:
            raise ValueError("Pinecone index is not initalized.")

        response = self.index.query(
            vector=query_embedding, top_k=top_k, include_metadata=True
        )
        return response["matches"]

    def upload_vectors(self, batch_size: int = 2):
        """
        Load embeddings from the saved dataset and upsert to Pinecone in batches.

        - Uses `save_embeddings_checkpoint()` to retrieve the latest data.
        - Converts DataFrame to Pinecone-compatible vectors.
        - Upserts vectors in batches.
        """
        summary_col_name = "text"
        uuid_col_name = "UUID"
        embeddings_col_name = "embeddings"

        print("Loading cleaned stats with embeddings...")
        cleaned_stats_with_embeddings = save_embeddings_checkpoint()

        print("Building vectors in Pinecone format...")
        vectors = [
            (
                row[uuid_col_name],
                row[embeddings_col_name],
                {"text": row[summary_col_name]},
            )
            for _, row in cleaned_stats_with_embeddings.iterrows()
        ]

        print(f"Uploading {len(vectors)} vectors to Pinecone...")

        for i in tqdm(
            range(0, len(vectors), batch_size), desc="Uploading batches to Pinecone"
        ):
            batch = vectors[i : i + batch_size]

            try:
                self.index.upsert(batch)
            except Exception as e:
                print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")

        print("Pinecone upload complete.")

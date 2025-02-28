import os
from pinecone import Pinecone, ServerlessSpec
from dotenv import load_dotenv
import pandas as pd
from preprocess_ufc_stats import preprocess_stats
from embeddings import create_embeddings_col
from tqdm import tqdm
import json


load_dotenv()


def load_clean_stats() -> pd.DataFrame:
    STATS_CLEANED_DATA_PATH = "./data/ufc_fight_stats_cleaned.csv"

    cleaned_stats = None
    if os.path.exists(STATS_CLEANED_DATA_PATH):
        print("Cleaned UFC Stats found. Loading data...")
        cleaned_stats = pd.read_csv(STATS_CLEANED_DATA_PATH)
    else:
        print("Cleaned UFC Stats not found. Generating cleaned data...")
        cleaned_stats = preprocess_stats()

    return cleaned_stats


def save_embeddings_checkpoint() -> pd.DataFrame:
    STATS_CLEANED_WITH_EMBEDDINGS_DATA_PATH = (
        "./data/ufc_fight_stats_cleaned_with_embeddings.csv"
    )

    cleaned_stats_with_embeddings = None
    if os.path.exists(STATS_CLEANED_WITH_EMBEDDINGS_DATA_PATH):
        print(
            "Embeddings for Cleaned UFC Stats found. Loading embeddings checkpoint..."
        )
        cleaned_stats_with_embeddings = pd.read_csv(
            STATS_CLEANED_WITH_EMBEDDINGS_DATA_PATH
        )
    else:
        cleaned_stats = load_clean_stats()
        summary_col_name = "text"

        print("Embeddings for Cleaned UFC Stats not found. Attempting to generate...")

        cleaned_stats_with_embeddings = create_embeddings_col(
            cleaned_stats, summary_col_name
        )

        cleaned_stats_with_embeddings.to_csv(
            STATS_CLEANED_WITH_EMBEDDINGS_DATA_PATH, index=False
        )

        print(
            "Saved embeddings checkpoint to " + STATS_CLEANED_WITH_EMBEDDINGS_DATA_PATH
        )

    cleaned_stats_with_embeddings["embeddings"] = cleaned_stats_with_embeddings[
        "embeddings"
    ].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    return cleaned_stats_with_embeddings


# Program entry-point
def main():
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

    summary_col_name = "text"
    uuid_col_name = "UUID"
    embeddings_col_name = "embeddings"

    cleaned_stats_with_embeddings = save_embeddings_checkpoint()

    print("Building vectors in Pinecone format...")
    vectors = [
        (row[uuid_col_name], row[embeddings_col_name], {"text": row[summary_col_name]})
        for _, row in cleaned_stats_with_embeddings.iterrows()
    ]

    index = pc.Index(INDEX_NAME)

    print(f"Uploading {len(vectors)} vectors to Pinecone...")

    batch_size = 2
    for i in tqdm(
        range(0, len(vectors), batch_size), desc="Uploading batches to Pinecone"
    ):
        batch = vectors[i : i + batch_size]

        print(
            f"Attempting to upload batch {i//batch_size + 1}/{(len(vectors) + batch_size - 1) // batch_size} with {len(batch)} vectors."
        )

        try:
            index.upsert(batch)
        except Exception as e:
            print(f"Error uploading batch {i//batch_size + 1}: {str(e)}")

    print("Pinecone upload complete.")


if __name__ == "__main__":
    main()

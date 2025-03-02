import os
import json
import pandas as pd
from preprocess import preprocess_stats
from embeddings import create_embeddings_col

STATS_CLEANED_DATA_PATH = "./data/ufc_fight_stats_cleaned.csv"
STATS_CLEANED_WITH_EMBEDDINGS_DATA_PATH = (
    "./data/ufc_fight_stats_cleaned_with_embeddings.csv"
)


def load_clean_stats() -> pd.DataFrame:
    """Load or generate cleaned UFC stats data."""
    cleaned_stats = None
    if os.path.exists(STATS_CLEANED_DATA_PATH):
        print("Cleaned UFC Stats found. Loading data...")
        cleaned_stats = pd.read_csv(STATS_CLEANED_DATA_PATH)
    else:
        print("Cleaned UFC Stats not found. Generating cleaned data...")
        cleaned_stats = preprocess_stats()

    return cleaned_stats


def save_embeddings_checkpoint() -> pd.DataFrame:
    """Load or generate UFC stats with embeddings."""
    cleaned_stats_with_embeddings = None
    if os.path.exists(STATS_CLEANED_WITH_EMBEDDINGS_DATA_PATH):
        print("Loading existing embeddings checkpoint...")
        cleaned_stats_with_embeddings = pd.read_csv(
            STATS_CLEANED_WITH_EMBEDDINGS_DATA_PATH
        )
    else:
        cleaned_stats = load_clean_stats()
        summary_col_name = "text"

        print("Generating embeddings for UFC Stats...")

        cleaned_stats_with_embeddings = create_embeddings_col(
            cleaned_stats, summary_col_name
        )

        cleaned_stats_with_embeddings.to_csv(
            STATS_CLEANED_WITH_EMBEDDINGS_DATA_PATH, index=False
        )

        print("Saved embeddings checkpoint.")

    cleaned_stats_with_embeddings["embeddings"] = cleaned_stats_with_embeddings[
        "embeddings"
    ].apply(lambda x: json.loads(x) if isinstance(x, str) else x)

    return cleaned_stats_with_embeddings

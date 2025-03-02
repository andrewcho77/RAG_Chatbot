from sentence_transformers import SentenceTransformer
import pandas as pd
import openai
from typing import List, Optional
from tqdm import tqdm


def fetch_openai_embeddings(
    texts: List[str], model: str = "text-embedding-ada-002"
) -> List[List[float]]:
    """
    Fetch embeddings from OpenAI's API for a list of texts.
    """
    response = openai.embeddings.create(input=texts, model=model)
    return [entry.embedding for entry in response.data]


def create_embeddings_col(
    data_frame: pd.DataFrame, col_name: str, num_rows: Optional[int] = None
) -> pd.DataFrame:
    """
    Create embeddings using OpenAI's API for a given DataFrame column.
    """
    tqdm.pandas()

    limited_data_frame = data_frame.head(num_rows) if num_rows else data_frame.copy()

    texts = limited_data_frame[col_name].tolist()

    batch_size = 25
    embeddings = []

    for i in tqdm(range(0, len(texts), batch_size), desc="Generating Embeddings"):
        batch = texts[i : i + batch_size]
        batch_embeddings = fetch_openai_embeddings(batch)
        embeddings.extend(batch_embeddings)

    limited_data_frame["embeddings"] = embeddings
    return limited_data_frame

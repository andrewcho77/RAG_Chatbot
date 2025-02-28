# RAG Chatbot

This project is a Retrieval-Augmented Generation (RAG) chatbot that provides insights based on UFC fight statistics. It uses an ETL pipeline to process and store fight data in a vector database, allowing for fast and relevant retrieval.

---

## System Overview

### ETL (Extract, Transform, Load) Pipeline

#### Entry Point
- The data pipeline starts with `load_ufc_stats.py`, which pulls UFC fight statistics from [this dataset](https://github.com/Greco1899/scrape_ufc_stats).
- The dataset is updated after each fight card and provides raw statistics for fighters, rounds, and events.

#### Processing Steps

##### 1. Data Preprocessing
Before storing the data, several transformations are applied:

- **Parsing numeric values**  
  Some columns contain numbers formatted as text, such as percentages, time durations, or "X of Y" stats. These are converted to numeric types for consistency.

- **Assigning unique identifiers**  
  Each row gets a UUID to make it easy to track fights across different processing stages.

- **Generating custom stat-line summaries**  
  A concise summary is created for each fight based on key stats. Example:

    ```
    In UFC 311: Makhachev vs. Moicano, Islam Makhachev fought in Islam Makhachev vs. Renato Moicano during round Round 1. They landed 6.0 significant strikes at an accuracy of 31.0%. Total strikes landed: 18.0, attempted: 31.0 | Knockdowns: 0.0. Takedowns: 1.0 of 2.0 (50.0% success rate). Submission attempts: 1.0. Control time: 1.0 min 27.0 sec. Head strikes landed: 5.0, attempted: 17.0. Body strikes landed: 1.0, attempted: 2.0. Leg strikes landed: 0.0, attempted: 0.0. Distance strikes landed: 5.0, attempted: 18.0. Clinch strikes landed: 0.0, attempted: 0.0. Ground strikes landed: 1.0, attempted: 1.0.
    ```

##### 2. Embedding Generation
- Each fight summary is converted into a vector embedding using OpenAI's `text-embedding-ada-002` model.
- API calls are batched to reduce cost and avoid hitting rate limits.

##### 3. Storing Data in a Vector Database
- The embeddings and metadata are stored in **Pinecone**, under the `ufc-stats` index.
- Each entry includes:
  - A UUID
  - The generated vector embedding
  - The original fight summary as metadata
- Upserts are also batched for efficiency.

---

## Caching and Checkpointing
- The pipeline maintains checkpoints so that expensive operations (like embedding generation) don’t have to be repeated unnecessarily.
- If a failure occurs, processing can resume from the last successful step instead of restarting from scratch.
- OpenAI API calls are minimized by caching results.

---

## Future Plans
- Improve the fight summaries by incorporating fighter history and trends.
- Experiment with different embedding strategies for better search relevance.
- Expand the dataset by pulling data from additional sources.
- Optimize retrieval performance for chatbot responses.
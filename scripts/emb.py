# This script embeds course information using a Gemini embedding model and stores the vectors with course metadata.
#
# Usage (from the project root):
#   uv run scripts/emb.py build fordham_course.xlsx
#   uv run scripts/emb.py search "machine learning with python" --difficulty beginner --max-hours 20

import os
import time

import numpy as np
import pandas as pd
import typer
from dotenv import load_dotenv
from google import genai
from google.genai import types

# Works both when run directly (uv run scripts/emb.py) and when imported (from scripts.emb import ...)
try:
    from scripts.preprocessing import data_import, preprocess_data, project_dir, data_dir
except ImportError:
    from preprocessing import data_import, preprocess_data, project_dir, data_dir


MODEL = 'gemini-embedding-001'
DIM = 768                     # smaller than the default 3072, little quality loss
BATCH_SIZE = 50               # texts per API request
MAX_RETRIES = 5

embeddings_dir = os.path.join(data_dir, 'embeddings')

# Metadata kept next to the vectors. Row i of the metadata belongs to row i of the vector matrix.
META_COLUMNS = ['course_id', 'course_name', 'university_industry_partner_name', 'difficulty_level',
                'avg_total_learning_hours', 'course_rating', 'course_url',
                'course_languages', 'skills', 'embedding_text']

app = typer.Typer(help = 'Embed course catalogs with Gemini and search them.')


# Create a Gemini client using the API key from env_vars.env or .env in the project root.
def get_client() -> genai.Client:

    for env_file in ('env_vars.env', '.env'):
        load_dotenv(os.path.join(project_dir, env_file))

    api_key = os.getenv('GEMINI_API_KEY')

    if api_key is None:
        raise RuntimeError('GEMINI_API_KEY not found. Add it to env_vars.env or .env in the project root.')

    return genai.Client(api_key = api_key)


# Turn a list of texts into a matrix of unit-length vectors (one row per text).
# Use task_type='RETRIEVAL_DOCUMENT' for courses and 'RETRIEVAL_QUERY' for user questions.
def embed_texts(client: genai.Client, texts: list[str], task_type: str = 'RETRIEVAL_DOCUMENT', verbose: bool = True) -> np.ndarray:

    vectors = []

    for start in range(0, len(texts), BATCH_SIZE):

        batch = texts[start:start + BATCH_SIZE]

        # Retry with increasing waits if the request fails (usually the free-tier rate limit)
        for attempt in range(MAX_RETRIES):
            try:
                result = client.models.embed_content(
                    model = MODEL,
                    contents = batch,
                    config = types.EmbedContentConfig(task_type = task_type, output_dimensionality = DIM),
                )
                break

            except Exception as err:
                wait = 5 * 2 ** attempt
                typer.echo(f'Batch at {start} failed ({err}); retrying in {wait}s', err = True)
                time.sleep(wait)

        else:
            raise RuntimeError(f'Batch at {start} failed after {MAX_RETRIES} tries.')

        vectors.extend(emb.values for emb in result.embeddings)

        if verbose:
            typer.echo(f'Embedded {min(start + BATCH_SIZE, len(texts))}/{len(texts)}')

    arr = np.array(vectors, dtype = np.float32)

    # Reduced-size Gemini vectors are not normalized, so scale each to length 1.
    # Then a dot product equals cosine similarity.
    arr /= np.linalg.norm(arr, axis = 1, keepdims = True)

    return arr


# File paths for one catalog, e.g. fordham_course.xlsx -> data/embeddings/fordham_course_vectors.npy
def index_paths(file_name: str) -> tuple[str, str]:

    stem = os.path.splitext(file_name)[0]

    return (os.path.join(embeddings_dir, f'{stem}_vectors.npy'),
            os.path.join(embeddings_dir, f'{stem}_meta.parquet'))


# Save vectors and metadata for one catalog.
def save_index(file_name: str, vectors: np.ndarray, courses: pd.DataFrame) -> None:

    os.makedirs(embeddings_dir, exist_ok = True)

    vectors_path, meta_path = index_paths(file_name)

    np.save(vectors_path, vectors)
    courses[META_COLUMNS].reset_index(drop = True).to_parquet(meta_path, index = False)


# Load vectors and metadata for one catalog.
def load_index(file_name: str) -> tuple[np.ndarray, pd.DataFrame]:

    vectors_path, meta_path = index_paths(file_name)

    if not (os.path.exists(vectors_path) and os.path.exists(meta_path)):
        raise FileNotFoundError(f'No embeddings for {file_name}. Run: uv run scripts/emb.py build {file_name}')

    vectors = np.load(vectors_path)
    meta = pd.read_parquet(meta_path)

    if len(vectors) != len(meta):
        raise ValueError(f'{file_name}: {len(vectors)} vectors but {len(meta)} metadata rows. Rebuild the embeddings.')

    return vectors, meta


# Rank courses by similarity to a query, with optional difficulty and hours filters.
def search_courses(client: genai.Client, query: str, vectors: np.ndarray, meta: pd.DataFrame,
                   k: int = 5, difficulty: str | None = None, max_hours: float | None = None) -> pd.DataFrame:

    q = embed_texts(client, [query], task_type = 'RETRIEVAL_QUERY', verbose = False)[0]

    results = meta.copy()
    results['score'] = vectors @ q

    if difficulty:
        results = results[results['difficulty_level'].eq(difficulty.upper()).fillna(False)]

    if max_hours:
        results = results[results['avg_total_learning_hours'].le(max_hours).fillna(False)]

    return results.sort_values('score', ascending = False).head(k)


@app.command()
def build(file_names: list[str] = typer.Argument(..., help = 'Names of CSV or Excel files inside the data folder.')):
    """Preprocess catalogs, embed every course, and save the vectors to data/embeddings/."""

    client = get_client()
    processed = preprocess_data(data_import(file_names))

    for file_name, courses in processed.items():

        vectors = embed_texts(client, courses['embedding_text'].tolist())
        save_index(file_name, vectors, courses)

        typer.echo(f'{file_name}: saved {vectors.shape[0]} vectors of size {vectors.shape[1]} to {embeddings_dir}')


@app.command()
def search(query: str = typer.Argument(..., help = 'What the learner wants to study.'),
           file_name: str = typer.Option('fordham_course.xlsx', help = 'Catalog to search.'),
           k: int = typer.Option(5, help = 'Number of results.'),
           difficulty: str = typer.Option(None, help = 'BEGINNER, INTERMEDIATE or ADVANCED.'),
           max_hours: float = typer.Option(None, help = 'Maximum average learning hours.')):
    """Search saved embeddings for the courses closest to a query."""

    vectors, meta = load_index(file_name)
    results = search_courses(get_client(), query, vectors, meta, k, difficulty, max_hours)

    columns = ['course_name', 'difficulty_level', 'avg_total_learning_hours', 'course_rating', 'score']

    with pd.option_context('display.max_colwidth', 60, 'display.width', 200):
        typer.echo(results[columns].to_string(index = False))


if __name__ == '__main__':
    app()

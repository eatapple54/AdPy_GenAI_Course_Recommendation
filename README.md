# GenAI Course Recommendation

Final course project for **Advanced Python with AI**. The goal is to recommend course packages based on a learner's goals, skill needs, experience, and constraints.

The project currently preprocesses course catalogs, embeds each course with Gemini, and supports similarity search over the saved vectors. User-feature extraction, ranking, and package assembly are planned work.

## Project structure

```text
AdPy_GenAI_Course_Recommendation/
├── data/
│   └── fordham_course.xlsx          # Source course catalog
├── scripts/
│   ├── preprocessing.py            # Data import, cleaning, and CLI
│   ├── emb.py                      # Gemini embeddings, storage, and search
│   └── main.py                     # Placeholder for the application workflow
├── src/
│   └── adpy_genai_course_recommendation/
│       └── __init__.py              # Generated greeting entry point
├── .python-version                 # Local Python version: 3.10
├── pyproject.toml                  # Project configuration and dependencies
├── uv.lock                         # Locked dependency versions
└── README.md
```

## Setup

Requirements: **Python 3.10 or newer** and **uv**. See the [official uv installation instructions](https://docs.astral.sh/uv/getting-started/installation/) if needed.

From the project root, install the locked dependencies:

```powershell
uv sync
```

uv creates or uses the project's `.venv`. Run scripts with `uv run`; manual environment activation is not required.

Dependencies are declared in `pyproject.toml`, with exact resolved versions in `uv.lock`. Add future dependencies with:

```powershell
uv add package-name
```

The preprocessing workflow uses pandas, openpyxl, and Typer and does not require API credentials. Embedding uses `google-genai`, `python-dotenv`, NumPy, and PyArrow, and requires a Gemini API key.

## Dataset

The included `data/fordham_course.xlsx` workbook contains **1,619 records and 25 columns** in the `Catalog_Python` sheet. Fields include course titles, descriptions, skills, difficulty, learning hours, ratings, languages, providers, specialization information, and URLs.

Place additional CSV or `.xlsx` files in `data/`. Each file is treated as a separate catalog. Excel import currently reads the first worksheet.

`Course Name` is required. Missing optional metadata is retained as missing values. The preprocessing function creates absent expected columns when needed.

## Run preprocessing

Run these commands from the project root. Pass filenames relative to `data/`:

```powershell
uv run scripts/preprocessing.py fordham_course.xlsx
```

Process multiple catalogs by listing their filenames:

```powershell
uv run scripts/preprocessing.py fordham_course.xlsx another_catalog.csv
```

The second example requires `data/another_catalog.csv` to exist. Quote filenames containing spaces.

View CLI help:

```powershell
uv run scripts/preprocessing.py --help
```

Expected output for the included workbook:

```text
Imported fordham_course.xlsx successfully.
Preprocessed fordham_course.xlsx: 1619 courses retained; 0 untitled or duplicate rows removed.
fordham_course.xlsx: 1619 courses processed.
```

**The preprocessing CLI prints processing counts. It does not save cleaned data or modify the source workbook.**

### What preprocessing does

- Copies each input DataFrame and normalizes column names to `snake_case`.
- Cleans whitespace, HTML entities, Excel character escapes, and recognizable encoding corruption.
- Removes rows without course names and completely duplicate rows.
- Checks numeric metadata and normalizes difficulty labels, treating `NOT CALIBRATED` as missing.
- Preserves existing course IDs and fills missing IDs using a SHA-256 hash of identifying information.
- Raises an error if conflicting rows share a course ID within one catalog.
- Combines skill fields into a case-normalized list without duplicates.
- Creates separate language lists for course language, subtitles, and translations.
- Builds one `embedding_text` value per course while keeping metadata in separate columns.

The new columns are:

| Column | Contents |
| --- | --- |
| `skills` | Combined unique values from `skills_learned` and `scorable_skills` |
| `course_languages` | Course language list |
| `subtitle_languages` | Subtitle language list |
| `translated_languages` | Machine-translated language list |
| `embedding_text` | Title, domain, sub-domain, skills, and description |

Skills and language lists use lowercase case-normalized values. Their order is not guaranteed because the current helper uses a set. Original skill and language columns are also retained.

### Use cleaned data in Python

From a Python session launched in the project root:

```python
from scripts.preprocessing import data_import, preprocess_data

catalogs = data_import(["fordham_course.xlsx"])
processed = preprocess_data(catalogs)
courses = processed["fordham_course.xlsx"]

print(courses[["course_id", "course_name", "difficulty_level"]].head())
print(courses["embedding_text"].iloc[0])
```

Both functions return dictionaries mapping filenames to DataFrames. `scripts/emb.py` uses `embedding_text` as its embedding input.

## Generate embeddings

`scripts/emb.py` embeds each course's `embedding_text` with Gemini (`gemini-embedding-001`, 768 dimensions, normalized to unit length) and saves the results to `data/embeddings/`:

| File | Contents |
| --- | --- |
| `<catalog>_vectors.npy` | One row per course: an N × 768 float32 matrix |
| `<catalog>_meta.parquet` | Course ID, name, provider, difficulty, hours, rating, URL, languages, skills, and embedding text, in the same row order as the vectors |

Put your Gemini API key in `env_vars.env` or `.env` in the project root (both are git-ignored):

```text
GEMINI_API_KEY=your-key-here
```

Build the embeddings (about 1,600 API inputs for the included catalog, sent in batches of 50 with automatic retries):

```powershell
uv run scripts/emb.py build fordham_course.xlsx
```

Search the saved embeddings:

```powershell
uv run scripts/emb.py search "machine learning with python" --difficulty beginner --max-hours 20
```

Courses are embedded with task type `RETRIEVAL_DOCUMENT` and queries with `RETRIEVAL_QUERY`. Because vectors are unit length, a dot product gives cosine similarity.

## Planned recommendation workflow

1. Preprocess the catalog and build course embedding text.
2. Generate a vector for each course and store it with course metadata.
3. Extract learning needs, preferences, and constraints from user input.
4. Search separately for each learning need and retrieve the top 5–10 eligible courses.
5. Rerank candidates and combine them according to a package template.
6. Check overall skill coverage, repeated content, difficulty progression, and the total time budget.

Difficulty, duration, and language should support filtering and package selection. Ratings, enrollment counts, URLs, and identifiers remain metadata rather than embedding content. Both duration fields are preserved separately; their definitions need to be confirmed before choosing a package time estimate.

Preprocessing (step 1) and embedding with similarity search (step 2) are implemented. The installed project command currently prints a greeting.

## Current limitations

- Text encoding repair is best effort and may not resolve every corrupted character.
- Non-numeric text in numeric columns currently raises a conversion error; numeric fields should contain numbers or missing values.
- The importer also accepts `.xls` extensions, but legacy Excel files may need an additional reader such as `xlrd`, which is not currently declared.
- Conflict detection applies within each catalog; cross-catalog deduplication is not implemented.
- Embeddings are rebuilt from scratch on each `build` run; there is no incremental update.
- Cleaned-data export, retrieval evaluation, and package templates are not implemented yet.

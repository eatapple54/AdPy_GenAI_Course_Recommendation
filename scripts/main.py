# The main execution script for the application.
# It asks the student a few questions and shows the best matching courses.
# Usage (from the project root):
#   uv run scripts/main.py

import pandas as pd
import typer

# Works both when run directly (uv run scripts/main.py) and when imported (from scripts.main import ...)
try:
    from scripts.emb import get_client, load_index, search_courses
except ImportError:
    from emb import get_client, load_index, search_courses


FILE_NAME = 'fordham_course.xlsx'
LEVELS = ['beginner', 'intermediate', 'advanced']


# Ask the student for a goal, level, and time limit, then print the best matching courses.
def main():

    typer.echo('Coursera course recommender for Fordham students\n')

    goal = typer.prompt('What do you want to learn? (for example: machine learning with python)')
    level = typer.prompt('Your level (beginner / intermediate / advanced)', default = 'beginner').strip().lower()
    max_hours = typer.prompt('Maximum hours you can spend on one course', default = 20.0, type = float)

    if level not in LEVELS:
        typer.echo(f'Level must be one of: {", ".join(LEVELS)}.', err = True)
        raise typer.Exit(code = 1)

    # Load the saved embeddings (built earlier with emb.py) and search them
    vectors, meta = load_index(FILE_NAME)
    results = search_courses(get_client(), goal, vectors, meta, k = 5, difficulty = level, max_hours = max_hours)

    if results.empty:
        typer.echo('\nNo courses matched. Try a higher hour limit or a different level.')
        return

    typer.echo(f'\nTop {len(results)} courses for "{goal}":')

    for number, (_, row) in enumerate(results.iterrows(), start = 1):

        rating = f"{row['course_rating']:.1f}" if pd.notna(row['course_rating']) else 'no rating'

        typer.echo(f"\n{number}. {row['course_name']}")
        typer.echo(f"   Provider: {row['university_industry_partner_name']}")
        typer.echo(f"   {row['difficulty_level']} | {row['avg_total_learning_hours']:.0f} hours | rating: {rating}")
        typer.echo(f"   Skills: {', '.join(list(row['skills'])[:5])}")
        typer.echo(f"   {row['course_url']}")


if __name__ == '__main__':
    typer.run(main)
# This script is responsible for preprocessing Coursera course catalog data.

import pandas as pd 
import typer
import os 
import re
import hashlib
import html
import unicodedata

project_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

data_dir = os.path.join(project_dir, 'data')


# Import specified csv files or excel files from the data directory and return a dictionary of DataFrames
def data_import(file_names: list[str] = typer.Argument(..., help = 'The list of csv file names or excel file names to import from the data directory.')) -> dict[str, pd.DataFrame]:

    data_dic = {}

    for file_name in file_names:

        file_path = os.path.join(data_dir, file_name)
        extension = os.path.splitext(file_name)[1].lower()

        if os.path.exists(file_path):

            if extension == '.csv':
                df = pd.read_csv(file_path)

            elif extension in ['.xlsx', '.xls']:
                df = pd.read_excel(file_path)

            else:
                typer.echo(f'Unsupported file format for {file_name}. Please change the file format to .csv or .xlsx/.xls.', err = True)
                continue

            data_dic[file_name] = df

            typer.echo(f'Imported {file_name} successfully.')

        else:

            typer.echo(f'File {file_name} does not exist in the data directory.', err = True)

    return data_dic

# Clean textual data and keep missing values
# This part is with thte help of codex
def clean_text(value):

    if pd.isna(value):
        return pd.NA

    # Convert to text and decode HTML entities
    text = html.unescape(str(value))

    # Decode excel escape sequences like _x000D_ and _x000A_ to their corresponding characters
    text = re.sub(r"_x([0-9a-fA-F]{4})_", lambda m: chr(int(m[1], 16)), text)

    # Repair common UTF-8 mis-encodings, including Windows-1252 bytes that were mis-decoded as ISO-8859-1 or CP1252.
    
    def repair_token(match):
        
        token = match[0]

        for _ in range(2):

            if not any(marker in token for marker in ("\u00c3", "\u00c2", "\u00e2\u20ac", "\u00ef\u00bb\u00bf")):
                break

            try:

                encoded = b"".join(
                    bytes([ord(char)]) if 0x80 <= ord(char) <= 0x9F
                    else char.encode("cp1252")
                    for char in token
                )

                repaired = encoded.decode("utf-8")

            except (UnicodeEncodeError, UnicodeDecodeError):
                break

            token = repaired

        return token

    text = re.sub(r"\S+", repair_token, text)
    text = unicodedata.normalize("NFC", text)
    text = " ".join(text.replace("\ufeff", "").split())

    return pd.NA if text in {"", "-"} else text

# Combine skills learned and scorable skills into a single list, removing duplicates and ignoring case.
# Combine language columns into lists, removing duplicates and ignoring case.
def split_values(*values) -> list[str]:

    result = set()

    for value in values:
        if pd.isna(value):
            continue

        for item in str(value).split(','):

            item = item.strip()
            if item:
                result.add(item.casefold())

    return list(result)


# Preprocess the course catalog data and return cleaned copies of DataFrames.
def preprocess_data(data_dic: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:

    processed_data_dic = {}

    text_columns = ['course_description', 'skills_learned', 'scorable_skills', 'domain', 'sub_domain', 'course_id', 'course_url', 
                    'course_language', 'subtitle_language', 'ml_translation', 'difficulty_level', 'university_industry_partner_name']

    for file_name, df in data_dic.items():

        cleaned = df.copy()

        cleaned.columns = [re.sub(r"[^a-z0-9]+", "_", str(col).strip().lower()).strip('_') for col in cleaned.columns]

        if 'course_name' not in cleaned.columns:
            raise ValueError(f'{file_name}: missing required \'Course Name\' column.')

        for column in cleaned.select_dtypes(include = ['object', 'string']).columns:

            cleaned[column] = cleaned[column].map(clean_text)

        for column in text_columns:
            if column not in cleaned.columns:
                cleaned[column] = pd.NA

        # Drop rows with missing course names and remove duplicates, resetting the index.
        cleaned = cleaned.dropna(subset = ['course_name']).drop_duplicates().reset_index(drop = True)
        
        # Check numerical features and set invalid values to NaN
        numeric_features = {'avg_total_learning_hours': (0, None),
                            'learning_hours_by_instructor': (0, None), 
                            'course_rating': (0, 5), 
                            'enrollment_count': (0, None)}

        for column, (minimum, maximum) in numeric_features.items():

            if column not in cleaned.columns:
                cleaned[column] = pd.NA

            values = pd.to_numeric(cleaned[column])
            valid = (values >= minimum) & (values < float('inf'))

            if maximum is not None:
                valid = valid & (values <= maximum)

            if 'hours' in column:
                valid = valid & values > 0

            cleaned[column] = values.where(valid)

        # Check difficulty level and set invalid values to NaN, converting to uppercase and replacing 'NOT CALIBRATED' with NaN.
        cleaned['difficulty_level'] = (cleaned['difficulty_level'].astype('string').str.upper().replace({'NOT CALIBRATED': pd.NA}))

        # Generate a unique course ID based on the course_id, course_url, or a hash of other identifying fields.
        # Some course ids are missing.
        def course_id(row):

            if pd.notna(row['course_id']):
                return str(row['course_id'])
            
            if pd.notna(row['course_url']):
                identity = str(row['course_url'])

            else:
                identity = '|'.join(str(row[column]) if pd.notna(row[column]) else '' for column in ('course_name', 'university_industry_partner_name', 'course_language'))

            # This part is with the help of codex
            return 'generated_' + hashlib.sha256(identity.encode("utf-8")).hexdigest()

        cleaned['course_id'] = pd.Series([course_id(row) for _, row in cleaned.iterrows()], index = cleaned.index, dtype = 'string')

        # Raise an error if there are conflicting rows sharing the same Course ID.
        if cleaned['course_id'].duplicated().any():
            raise ValueError(f'{file_name}: conflicting rows share a Course ID.')

        cleaned['skills'] = [split_values(learned, scorable) for learned, scorable in zip(cleaned['skills_learned'], cleaned['scorable_skills'])]

        for source, target in (
            ('course_language', 'course_languages'),
            ('subtitle_language', 'subtitle_languages'),
            ('ml_translation', 'translated_languages'),
        ):
            cleaned[target] = cleaned[source].map(split_values)

        def embedding_text(row):

            parts = [f"Course: {row['course_name']}"]

            for label, column in (('Domain', 'domain'), ('Sub-domain', 'sub_domain')):

                if pd.notna(row[column]):
                    parts.append(f'{label}: {row[column]}')

            if row['skills']:

                parts.append("Skills: " + ", ".join(row['skills']))

            if pd.notna(row['course_description']):
                parts.append("Description: " + row['course_description' ])

            return '\n\n'.join(parts)

        cleaned['embedding_text'] = pd.Series([embedding_text(row) for _, row in cleaned.iterrows()], index = cleaned.index)

        processed_data_dic[file_name] = cleaned

        typer.echo(
            f'Preprocessed {file_name}: {len(cleaned)} courses retained; '
            f'{len(df) - len(cleaned)} untitled or duplicate rows removed.'
        )

    return processed_data_dic

# Main execution function for the CLI, which imports and preprocesses the specified data files.
def main(file_names: list[str] = typer.Argument(..., help = 'Names of CSV or Excel files inside the data folder.')):

    imported = data_import(file_names)

    processed = preprocess_data(imported)

    for file_name, df in processed.items():

        typer.echo(f'{file_name}: {len(df)} courses processed.')


if __name__ == "__main__":
    typer.run(main)
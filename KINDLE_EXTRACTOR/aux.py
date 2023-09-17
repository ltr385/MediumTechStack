import pandas as pd
from typing import List

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Cleans the input dataframe based on the specified groupings and sorts.
    Retains the last entry if there are multiple records for the same 'Start_Pos' 
    after grouping by 'Title' and 'Author'.
    
    Parameters:
    - df (pd.DataFrame): The input dataframe to clean.
    
    Returns:
    - pd.DataFrame: The cleaned dataframe.
    """

    all_dfs = []

    for _, df_group in df.groupby(['Title', 'Author']):
        df_group = df_group.sort_values(by=['Start_Pos', 'Date'])
        df_clean = []

        for _, df_subgroup in df_group.groupby('Start_Pos'):
            df_clean.append(df_subgroup.iloc[-1])

        all_dfs.extend(df_clean)

    return pd.DataFrame(all_dfs).reset_index(drop=True)


def deduplicate_dataframe(df: pd.DataFrame) -> (pd.DataFrame, List, List):
    """Remove similar records in the dataframe based on text, title, and position."""

    # Make a copy of the dataframe to ensure the original data remains unchanged.
    df_copy = df.copy()

    # Create a TF-IDF vectorizer to convert the text data into vectors.
    vectorizer = TfidfVectorizer()

    # Fit and transform the texts into vectors for similarity comparison.
    tfidf_matrix = vectorizer.fit_transform(df_copy['Text'])

    # Calculate cosine similarity among text entries.
    # This helps determine how similar each text is to every other text.
    cosine_similarities = cosine_similarity(tfidf_matrix)

    # Define thresholds for similarity and positional difference.
    # Entries above the similarity threshold and below the position threshold are considered similar.
    SIMILARITY_THRESHOLD = 0.8
    POSITION_THRESHOLD = 20  # Allowed range of difference between positions.

    # Markers for rows to delete. 
    # This will store indices of the dataframe rows which are considered duplicates.
    rows_to_drop = set()

    # Logs for similar and deleted entries.
    # Helps in tracking which entries were found to be similar and which ones were deleted.
    log_similar_entries = []
    log_deleted_entries = []

    for i in range(cosine_similarities.shape[0]):
        for j in range(i + 1, cosine_similarities.shape[0]):
            # Checking similarity of title and positions.
            same_title = df_copy.at[i, 'Title'] == df_copy.at[j, 'Title']
            start_pos_close = abs(df_copy.at[i, 'Start_Pos'] - df_copy.at[
                j, 'Start_Pos']) <= POSITION_THRESHOLD
            end_pos_close = abs(df_copy.at[i, 'End_Pos'] - df_copy.at[
                j, 'End_Pos']) <= POSITION_THRESHOLD

            # If entries have the same title, close start and end positions, and text similarity above the threshold:
            # They are considered duplicates.
            if same_title and start_pos_close and end_pos_close and \
                    cosine_similarities[i, j] > SIMILARITY_THRESHOLD:
                log_similar_entries.append(
                    (df_copy.iloc[i].to_dict(), df_copy.iloc[j].to_dict()))

                # Of the similar entries, the one with the older date is dropped.
                if df_copy.at[i, 'Date'] > df_copy.at[j, 'Date']:
                    rows_to_drop.add(j)
                    log_deleted_entries.append(df_copy.iloc[j].to_dict())
                else:
                    rows_to_drop.add(i)
                    log_deleted_entries.append(df_copy.iloc[i].to_dict())

    # Remove identified duplicates from the copied dataframe and reset its index.
    df_cleaned = df_copy.drop(list(rows_to_drop)).reset_index(drop=True)

    # Return the cleaned dataframe along with logs of similar and deleted entries.
    return df_cleaned, log_similar_entries, log_deleted_entries

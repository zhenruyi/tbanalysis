import pandas as pd
import os
from src.utils.config import load_config
from src.utils.database import ensure_database_exists, write_to_database


def load_raw_data(file_path):
    return pd.read_csv(
        file_path,
        names=['user_id', 'item_id', 'category_id', 'behavior', 'timestamp']
    )


def sample(df, sampling_config):
    df['timestamp'] = pd.to_datetime(df['timestamp'], unit='s')
    df['date'] = df['timestamp'].dt.date
    df['hour'] = df['timestamp'].dt.hour
    df_sampled = df.sample(
        frac=sampling_config['frac'],
        random_state=sampling_config['random_state']
    )
    return df_sampled


def save_sample(df, file_path):
    df.to_csv(file_path, index=False)


if __name__ == "__main__":
    config = load_config()
    db_config = config['database']
    paths = config['paths']
    sampling_config = config['sampling']

    raw_path = os.path.join(os.path.dirname(__file__), f'../../{paths["raw_data"]}')
    processed_path = os.path.join(os.path.dirname(__file__), f'../../{paths["processed_data"]}')

    raw_df = load_raw_data(raw_path)
    sampled_df = sample(raw_df, sampling_config)
    save_sample(sampled_df, processed_path)

    try:
        ensure_database_exists(db_config)
        write_to_database(sampled_df, db_config, 'sampling')
    except Exception as e:
        print(f"Error writing to database: {e}")
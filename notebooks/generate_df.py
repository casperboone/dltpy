import os
import pandas as pd
import numpy as np

# CONFIG

# project root
PR = "../"
DATA_FILES_DIR = os.path.join(PR, "output/data")


def list_data_files(datadir, full=True):
    paths = []
    for file in os.listdir(datadir):
        if not os.path.isfile(os.path.join(datadir, file)):
            continue

        paths.append(os.path.join(datadir, file) if full else file)

    return paths


def parse_df(files, batch_size=10):
    df = None

    batch = []
    for i, file in enumerate(files):
        if i % 100 == 0:
            print(f"Loaded {i}/{len(files)} files")

        if df is None:
            df = pd.read_csv(file)
        else:
            batch.append(pd.read_csv(file))

        if i % batch_size == 0 and i > 0:
            df = pd.concat([df, *batch])
            batch = []

    if len(batch) > 0:
        df = pd.concat([df, *batch])

    return df


def format_df(df):
    """Add """
    df['arg_names'] = df['arg_names'].apply(lambda x: tuple(eval(x)))
    df['arg_types'] = df['arg_types'].apply(lambda x: np.asarray(eval(x)))

    return df


if __name__ == '__main__':
    DATA_FILES = list_data_files(DATA_FILES_DIR)
    print("Found %d datafiles" % len(DATA_FILES))

    df = parse_df(DATA_FILES, batch_size=32)
    print("Dataframe loaded!")

    print("Formatting dataframe")
    df = format_df(df)

    print("Dataframe formatted, writing it...")
    df.to_csv("_temp.csv")

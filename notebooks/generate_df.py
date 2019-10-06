import os
from typing import Tuple

import pandas as pd
import numpy as np

# CONFIG

# project root
from sklearn import preprocessing

PR = "../"
DATA_FILES_DIR = os.path.join(PR, "output/data")


def list_files(directory: str, full=True) -> list:
    """
    List all files in the given directory.
    
    :param directory: directory to search
    :param full: whether to return the full path
    :return: list of paths
    """
    paths = []
    for file in os.listdir(directory):
        if not os.path.isfile(os.path.join(directory, file)):
            continue

        paths.append(os.path.join(directory, file) if full else file)

    return paths


def parse_df(files: list, batch_size=10) -> pd.DataFrame:
    """
    Given a set of files, load each as a dataframe and concatenate the result
    into a large file dataframe containing the data from all files.

    :param files: the files to load
    :param batch_size: batch size used for concatenation
    :return: final dataframe
    """
    df_merged = None
    batch = []
    total = 0
    for i, file in enumerate(files):
        if i % 100 == 0:
            print(f"Loaded {i}/{len(files)} files")

        _loaded = pd.read_csv(file, index_col=0)
        total += len(_loaded)
        if df_merged is None:
            df_merged = _loaded
        else:
            batch.append(_loaded)

        if i % batch_size == 0 and i > 0:
            frames = [df_merged, *batch]
            df_merged = pd.concat(frames, ignore_index=True)
            batch = []

    if len(batch) > 0:
        df_merged = pd.concat([df_merged, *batch])

    df_merged = df_merged.reset_index(drop=True)
    assert total == len(df_merged), "Length of all loaded should be equal to merged"

    return df_merged


def format_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Formats the loaded dataframe.
    :type df: dataframe to use
    :returns: final dataframe
    """
    df['arg_names'] = df['arg_names'].apply(lambda x: eval(x))
    df['arg_types'] = df['arg_types'].apply(lambda x: eval(x))
    df['arg_descrs'] = df['arg_descrs'].apply(lambda x: eval(x))
    df['return_expr'] = df['return_expr'].apply(lambda x: eval(x))

    return df


def filter_functions(df: pd.DataFrame) -> pd.DataFrame:
    """
    Filters functions which are note useful.
    :param df: dataframe to use
    :return: filtered dataframe
    """
    print(f"Functions before dropping on return type {len(df)}")
    df = df.dropna(subset=['return_type'])
    print(f"Functions after dropping on return type {len(df)}")

    print(f"Functions before dropping nan return type {len(df)}")
    to_drop = np.invert((df['return_type'] == 'nan') | (df['return_type'] == 'None'))
    df = df[to_drop]
    print(f"Functions after dropping nan return type {len(df)}")

    print(f"Functions before dropping on empty docstring {len(df)}")
    df = df.dropna(subset=['docstring'])
    print(f"Functions after dropping on empty docstring {len(df)}")

    return df


def gen_argument_df(df: pd.DataFrame) -> pd.DataFrame:
    """
    Generates a new dataframe containing all argument data.
    :param df: dataframe for which to extract argument
    :return: argument dataframe
    """
    arguments = []
    for i, row in df.iterrows():
        for p_i, arg_name in enumerate(row['arg_names']):
            if arg_name != 'self':
                arguments.append([row['name'], arg_name, row['arg_types'][p_i], row['arg_descrs'][p_i]])

    return pd.DataFrame(arguments, columns=['func_name', 'arg_name', 'arg_type', 'arg_comment'])


def encode_types(df: pd.DataFrame, df_args: pd.DataFrame, threshold: int = 999) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Encode the dataframe types to integers.
    :param df: dataframe with function data
    :param df_args: dataframe with argument data
    :param threshold: number of common types to keep
    :return:
    """
    le = preprocessing.LabelEncoder()

    # All types
    return_types = df['return_type'].values
    arg_types = df_args['arg_type'].values
    all_types = np.concatenate((return_types, arg_types), axis=0)

    unique, counts = np.unique(all_types, return_counts=True)
    print(f"Found {len(unique)} unique types in a total of {len(all_types)} types.")

    # keep the threshold most common types rest is mapped to 'other'
    common_types = [unique[i] for i in np.argsort(counts)[::-1][:threshold]]

    print("Remapping uncommon types for functions")
    df['return_type_t'] = df['return_type'].apply(lambda x: x if x in common_types else 'other')

    print("Remapping uncommon types for arguments")
    df_args['arg_type_t'] = df_args['arg_type'].apply(lambda x: x if x in common_types else 'other')

    print("Fitting label encoder on transformed types")
    # All types transformed
    return_types = df['return_type_t'].values
    arg_types = df_args['arg_type_t'].values
    all_types = np.concatenate((return_types, arg_types), axis=0)
    le.fit(all_types)

    # transform all type
    print("Transforming return types")
    df['return_type_enc'] = le.transform(return_types)

    print("Transforming args types")
    df_args['arg_type_enc'] = le.transform(arg_types)

    return df, df_args


if __name__ == '__main__':
    DATA_FILE = "_temp_2.csv"
    FILTERED_DATA_FILE = "_temp_filtered.csv"
    CACHE = True

    if CACHE:
        if os.path.exists(FILTERED_DATA_FILE):
            print("Loading filtered cached copy")
            df = pd.read_csv(FILTERED_DATA_FILE)
        else:
            print("Loading cached copy")
            df = pd.read_csv(DATA_FILE)
            df = filter_functions(df)
            df.to_csv(FILTERED_DATA_FILE, index=False)
    else:
        DATA_FILES = list_files(DATA_FILES_DIR)
        print("Found %d datafiles" % len(DATA_FILES))

        df = parse_df(DATA_FILES, batch_size=128)
        print("Dataframe loaded!")

        print("Formatting dataframe")
        # df = format_df(df)
        print(df.head())

        print("Dataframe formatted, writing it...")
        df.to_csv(DATA_FILE, index=False)

    # Format dataframe
    print("Formatting dataframe")
    df = format_df(df)

    print(f"Functions before dropping on empty return expression {len(df)}")
    df = df[df['return_expr'].apply(len) > 0]
    print(f"Functions after dropping on empty return expression {len(df)}")

    # Split df
    print("Extracting arguments")
    df_params = gen_argument_df(df)
    print(f"Extracted a total of {len(df_params)} arguments with type {sum(df_params['arg_type'] != '')}.")
    df_params = df_params[df_params['arg_type'] != '']

    # Encode types as int
    df, df_params = encode_types(df, df_params)

    # Drop all columns useless for the ML algorithms
    df = df.drop(columns=['file', 'author'])

    df.to_csv("_ml_inputs.csv", index=False)
    df_params.to_csv("_ml_inputs_args.csv", index=False)

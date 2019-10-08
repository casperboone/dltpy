import os
import re
from typing import Tuple

import pandas as pd
import numpy as np
import pickle

from pandas import DataFrame
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder

# Configuration
PROJECT_ROOT = os.path.dirname(__file__)
DATA_FILES_DIR = os.path.join(PROJECT_ROOT, "output/data/")
ML_INPUTS_PATH = os.path.join(PROJECT_ROOT, "output/ml_inputs/")
ML_RETURN_DF_PATH = os.path.join(ML_INPUTS_PATH, "_ml_return.csv")
ML_PARAM_DF_PATH = os.path.join(ML_INPUTS_PATH, "_ml_param.csv")
LABEL_ENCODER_PATH = os.path.join(ML_INPUTS_PATH, "label_encoder.pkl")
DATA_FILE = os.path.join(ML_INPUTS_PATH, "_all_data.csv")
FILTERED_DATA_FILE = os.path.join(ML_INPUTS_PATH, "_data_filtered.csv")

CACHE = True


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

    print(f"Functions before dropping on empty return expression {len(df)}")
    df = df[df['return_expr'].apply(lambda x: len(eval(x))) > 0]
    print(f"Functions after dropping on empty return expression {len(df)}")

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


def encode_types(df: pd.DataFrame, df_args: pd.DataFrame, threshold: int = 999) -> Tuple[
    DataFrame, DataFrame, LabelEncoder]:
    """
    Encode the dataframe types to integers.
    :param df: dataframe with function data
    :param df_args: dataframe with argument data
    :param threshold: number of common types to keep
    :return: dataframe with types encoded and the labels encoder used for it.
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

    return df, df_args, le


if __name__ == '__main__':
    if not os.path.exists(ML_INPUTS_PATH):
        os.makedirs(ML_INPUTS_PATH)

    if CACHE and os.path.exists(FILTERED_DATA_FILE):
        print("Loading filtered cached copy")
        df = pd.read_csv(FILTERED_DATA_FILE)
    elif CACHE and os.path.exists(DATA_FILE):
        print("Loading cached copy")
        df = pd.read_csv(DATA_FILE)
        df = filter_functions(df)
        df.to_csv(FILTERED_DATA_FILE, index=False)
    else:
        DATA_FILES = list_files(DATA_FILES_DIR)
        print("Found %d datafiles" % len(DATA_FILES))

        df = parse_df(DATA_FILES, batch_size=128)

        print("Dataframe loaded writing it to CSV")
        df.to_csv(DATA_FILE, index=False)

        print("Filtering dataframe")
        df = filter_functions(df)

        print("Dataframe filtered writing cached version to CSV")
        df.to_csv(FILTERED_DATA_FILE, index=False)

    # Format dataframe
    print("Formatting dataframe")
    df = format_df(df)

    # Split df
    print("Extracting arguments")
    df_params = gen_argument_df(df)

    print(f"Extracted a total of {len(df_params)} arguments with type {sum(df_params['arg_type'] != '')}.")
    df_params = df_params[df_params['arg_type'] != '']

    # Encode types as int
    print("Encoding types")
    df, df_params, label_encoder = encode_types(df, df_params)

    print("Storing label encoder")
    with open(LABEL_ENCODER_PATH, 'wb') as file:
        pickle.dump(label_encoder, file)

    # Add argument names as a string except self
    df['arg_names_str'] = df['arg_names'].apply(lambda l: " ".join([v for v in l if v != 'self']))

    # Add return expressions as a string, replace self. and self within expressions
    df['return_expr_str'] = df['return_expr'].apply(lambda l: " ".join([re.sub(r"self\.?", '', v) for v in l]))

    # Drop all columns useless for the ML algorithms
    df = df.drop(columns=['file', 'author', 'repo', 'has_type', 'arg_names', 'arg_types', 'arg_descrs', 'return_expr'])

    # Store the dataframes
    df.to_csv(ML_RETURN_DF_PATH, index=False)
    df_params.to_csv(ML_PARAM_DF_PATH, index=False)

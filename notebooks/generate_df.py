import os
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


def encode_types(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode the dataframe types to integers.
    :param df:
    :return:
    """
    le = preprocessing.LabelEncoder()

    # All types
    return_types = df['return_type'].values
    arg_types = np.hstack(df['arg_types'].values)
    rt = np.concatenate((return_types, arg_types), axis=0)
    unique, counts = np.unique(rt, return_counts=True)
    print(f"Found {len(unique)} unique types in a total of {len(rt)} types.")

    # type we are going to keep
    print("Remapping uncommon types for functions")
    common_types = [unique[i] for i in np.argsort(counts)[::-1][:999]]
    df['return_type_t'] = df['return_type'].apply(lambda x: x if x in common_types else 'other')
    print("Remapping uncommon types for arguments")
    df['arg_types_t'] = df['arg_types'].apply(lambda x: [i if i in common_types else 'other' for i in x])

    print("Fitting label encoder")
    # All types transformed
    return_types = df['return_type_t'].values
    arg_types = np.hstack(df['arg_types_t'].values)
    rt = np.concatenate((return_types, arg_types), axis=0)
    le.fit(rt)

    #     print(le.classes_)

    # transform all type
    print("Transforming return types")
    df['return_type_enc'] = le.transform(return_types)

    print("Transforming args types")
    df['arg_types_enc'] = df['arg_types_t'].apply(lambda x: le.transform(x))

    return df


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

    # Split df
    print("Extracting arguments")
    df_params = gen_argument_df(df)
    print(f"Extracted a total of {len(df_params)} arguments.")

    # Encode types as int
    df = encode_types(df)

    print(f"Functions before dropping on empty return expression {len(df)}")
    df = df[df['return_expr'].apply(len) > 0]
    print(f"Functions after dropping on empty return expression {len(df)}")

    # Drop all columns useless for the ML algorithms
    df = df.drop(columns=['file', 'author'])

    df.to_csv("_ml_inputs.csv", index=False)

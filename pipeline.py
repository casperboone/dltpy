import argparse
import json
import os
import shutil
import time
import traceback

import pandas as pd
from joblib import delayed

from cloner import Cloner
from extractor import Extractor, ParseError
from nl_preprocessing import NLPreprocessor
from project_filter import ProjectFilter
from utils import ParallelExecutor

cloner = Cloner()
project_filter = ProjectFilter()
extractor = Extractor()
preprocessor = NLPreprocessor()

# Create output directory
if not os.path.isdir('./output'):
    os.mkdir('./output')


# CONFIG
OUTPUT_DIRECTORY = os.path.join('./output', str(int(time.time())))
USE_CACHE = True


def list_files(directory: str) -> list:
    """
    List all files in the given directory (recursively)
    """
    filenames = []

    for root, dirs, files in os.walk(directory):
        for filename in files:
            filenames.append(os.path.join(root, filename))

    return filenames


def read_file(filename: str) -> str:
    """
    Open a file and return its contents as a string
    """
    with open(filename) as file:
        return file.read()


def get_project_filename(project) -> str:
    """
    Return the filename at which a project datafile should be stored.
    :param project: the project dict
    :return: return filename
    """
    return os.path.join(OUTPUT_DIRECTORY, f"{project['author']}{project['repo']}-functions.csv")


def write_project(project) -> None:
    functions = []
    columns = None

    if 'files' in project:
        for file in project['files']:
            for function in file['functions']:
                if columns is None:
                    columns = ['author', 'repo', 'file', 'has_type'] + list(function.tuple_keys())

                function_metadata = (
                                        project['author'],
                                        project['repo'],
                                        file['filename'],
                                        function.has_types()
                                    ) + function.as_tuple()

                functions.append(function_metadata)

                assert len(function_metadata) == len(columns), \
                    f"Assertion failed size of columns should be same as the size of the data tuple."

    if len(functions) == 0:
        return
    function_df = pd.DataFrame(functions, columns=columns)
    function_df['arg_names_len'] = function_df['arg_names'].apply(len)
    function_df['arg_types_len'] = function_df['arg_types'].apply(len)
    function_df.to_csv(get_project_filename(project))


def run_pipeline(projects: list) -> None:
    """
    Run the pipeline (clone, filter, extract, remove) for all given projects
    """
    ParallelExecutor(n_jobs=args.jobs)(total=len(projects))(
        delayed(process_project)(i, project) for i, project in enumerate(projects, start=args.start))


def process_project(i, project):
    try:
        project_id = f'{project["author"]}/{project["repo"]}'
        print(f'Running pipeline for project {i} {project_id}')

        if os.path.exists(get_project_filename(project)) and USE_CACHE:
            print(f"Found cached copy for project {project_id}")
            return

        project['files'] = []

        if 'repoUrl' in project:
            print(f'Cloning for {project_id}...')
            raw_project_directory = cloner.clone(project["author"], project["repo"])
        else:
            raw_project_directory = project["directory"]

        print(f'Filtering for {project_id}...')
        filtered_project_directory = project_filter.filter_directory(raw_project_directory)

        print(f'Extracting for {project_id}...')
        extracted_functions = {}
        for filename in list_files(filtered_project_directory):
            try:
                functions = extractor.extract(read_file(filename))
                extracted_functions[filename] = functions
            except ParseError:
                print(f"Could not parse file {filename}")
            except UnicodeDecodeError:
                print(f"Could not read file {filename}")

        print(f'Preprocessing for {project_id}...')
        preprocessed_functions = {}
        for filename, functions in extracted_functions.items():
            preprocessed_functions[filename] = [preprocessor.preprocess(function) for function in functions]

        project['files'] = [{'filename': filename, 'functions': functions}
                            for filename, functions in preprocessed_functions.items()]

        if 'repoUrl' in project:
            print(f'Remove project files for {project_id}...')
            shutil.rmtree(raw_project_directory)
    except KeyboardInterrupt:
        quit(1)
    except Exception:
        print(f'Running pipeline for project {i} failed')
        traceback.print_exc()
    finally:
        write_project(project)


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--projects_file',
                    help='json file containing GitHub projects',
                    type=str,
                    default='./resources/mypy-dependents-by-stars.json')
parser.add_argument('--limit',
                    help='limit the number of projects for which the pipeline should run',
                    type=int,
                    default=0)
parser.add_argument("--jobs",
                    help="number of jobs to use for pipeline.",
                    type=int,
                    default=-1)
parser.add_argument("--output_dir",
                    help="output dir for the pipeline",
                    type=str,
                    default=os.path.join('./output', str(int(time.time()))))
parser.add_argument('--start',
                    help='start position within projects list',
                    type=int,
                    default=0)

if __name__ == '__main__':
    # Parse args
    args = parser.parse_args()

    # Create output dir
    OUTPUT_DIRECTORY = args.output_dir
    if not os.path.exists(OUTPUT_DIRECTORY):
        os.mkdir(OUTPUT_DIRECTORY)

    # Open projects file and run pipeline
    with open(args.projects_file) as json_file:
        projects = json.load(json_file)

        if args.limit > 0:
            projects = projects[:args.limit]

        run_pipeline(projects)
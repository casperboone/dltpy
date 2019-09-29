import argparse
import json
import os
import shutil
import time
import traceback

import pandas as pd
from joblib import Parallel, delayed

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
output_directory = os.path.join('./output', str(int(time.time())))
os.mkdir(output_directory)


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

    function_df = pd.DataFrame(functions, columns=columns)
    function_df['arg_names_len'] = function_df['arg_names'].apply(len)
    function_df['arg_types_len'] = function_df['arg_types'].apply(len)
    function_df.to_csv(os.path.join(output_directory, f"{project['author']}{project['repo']}-functions.csv"))


def run_pipeline(projects: list) -> None:
    """
    Run the pipeline (clone, filter, extract, remove) for all given projects
    """
    ParallelExecutor(n_jobs=args.jobs)(total=len(projects))(delayed(process_project)(i, project) for i, project in enumerate(projects, start=1))


def process_project(i, project):
    try:
        project_id = f'{project["author"]}/{project["repo"]}'
        print(f'Running pipeline for project {i} {project_id}')

        project['files'] = []

        print(f'Cloning for {project_id}...')
        cloned_project_directory = cloner.clone(project["author"], project["repo"])

        print(f'Filtering for {project_id}...')
        filtered_project_directory = project_filter.filter_directory(cloned_project_directory)

        print(f'Extracting for {project_id}...')
        extracted_functions = {}
        for filename in list_files(filtered_project_directory):
            try:
                functions = extractor.extract(read_file(filename))
                extracted_functions[filename] = functions
            except ParseError:
                print(f"Could not parse file {filename}")
                # statistics['unparsabl_files'] += 1

        print(f'Preprocessing for {project_id}...')
        preprocessed_functions = {}
        for filename, functions in extracted_functions.items():
            preprocessed_functions[filename] = [preprocessor.preprocess(function) for function in functions]

        project['files'] = [{'filename': filename, 'functions': functions}
                            for filename, functions in preprocessed_functions.items()]

        print(f'Remove project files for {project_id}...')
        shutil.rmtree(cloned_project_directory)
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

args = parser.parse_args()

# Open projects file and run pipeline
with open(args.projects_file) as json_file:
    projects = json.load(json_file)

    if args.limit > 0:
        projects = projects[:args.limit]

    run_pipeline(projects)

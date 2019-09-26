import argparse
import json
import os
import shutil
import time
from pprint import pprint
import traceback
from pprint import pprint

import pandas as pd

from cloner import Cloner
from extractor import Extractor, ParseError
from nl_preprocessing import NLPreprocessor
from project_filter import ProjectFilter

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


def write_project_output(projects: list) -> None:
    """
    Write a list of all found functions and a list of all found functions with types to the output directory
    """
    functions = []
    columns = None

    for project in projects:
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

                    assert len(function_metadata) == len(columns), f"Assertion failed size of columns should be same " \
                                                                   f"as the size of the data tuple."

    function_df = pd.DataFrame(functions, columns=columns)
    function_df.to_csv(os.path.join(output_directory, "functions.csv"))


def write_statistics(statistics: dict) -> None:
    """
    Write statistics to the output directory
    """
    with open(os.path.join(output_directory, 'statistics.json'), 'w') as file:
        json.dump(statistics, file, default=lambda o: o.__dict__, indent=4)
        file.close()


def run_pipeline(projects: list) -> None:
    """
    Run the pipeline (clone, filter, extract, remove) for all given projects
    """
    statistics = {
        'projects': len(projects),
        'failed_projects': 0,
        'files': 0,
        'unparsable_files': 0,
        'functions': 0,
        'functions_with_types': 0
    }

    for i, project in enumerate(projects, start=1):
        try:
            print(f'Running pipeline for project {i}/{len(projects)}: {project["author"]}/{project["repo"]}')

            project['files'] = []

            print('Cloning...')
            cloned_project_directory = cloner.clone(project["author"], project["repo"])

            print('Filtering...')
            filtered_project_directory = project_filter.filter_directory(cloned_project_directory)

            print('Extracting...')
            extracted_functions = {}
            for filename in list_files(filtered_project_directory):
                statistics['files'] += 1
                try:
                    functions = extractor.extract(read_file(filename))
                    statistics['functions'] += len(functions)
                    statistics['functions_with_types'] += sum(function.has_types() for function in functions)
                    extracted_functions[filename] = functions
                except ParseError:
                    statistics['unparsable_files'] += 1

            print('Preprocessing...')
            preprocessed_functions = {}
            for filename, functions in extracted_functions.items():
                preprocessed_functions[filename] = [preprocessor.preprocess(function) for function in functions]

            project['files'] = [{'filename': filename, 'functions': functions}
                                for filename, functions in preprocessed_functions.items()]

            print('Remove project files...')
            shutil.rmtree(cloned_project_directory)
        except KeyboardInterrupt:
            quit(1)
        except Exception:
            statistics['failed_projects'] += 1
            print(f'Running pipeline for project {i}/{len(projects)} failed')
            traceback.print_exc()
        finally:
            write_project_output(projects)
            write_statistics(statistics)
    pprint(statistics)


# Parse command line arguments
parser = argparse.ArgumentParser()
parser.add_argument('--projects_file',
                    help='json file containing GitHub projects',
                    type=str,
                    default='./resources/mypy-dependents-by-stars.json')
parser.add_argument('--limit',
                    help='limit the number of projects for which the pipeline should run',
                    type=int)
args = parser.parse_args()

# Open projects file and run pipeline
with open(args.projects_file) as json_file:
    projects = json.load(json_file)

    if args.limit > 0:
        projects = projects[:args.limit]

    run_pipeline(projects)

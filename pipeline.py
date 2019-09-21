import argparse
import json
import os
import shutil
import time
from pprint import pprint
import traceback

from cloner import Cloner
from project_filter import ProjectFilter
from extractor import Extractor, ParseError

cloner = Cloner()
project_filter = ProjectFilter()
extractor = Extractor()

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
    functions_with_types = []

    for project in projects:
        if 'files' in project:
            for file in project['files']:
                for function in file['functions']:
                    function_metadata = {
                        'author': project['author'],
                        'repo': project['repo'],
                        'filename': file['filename'],
                        'signature': function
                    }
                    functions.append(function_metadata)
                    if function.has_types():
                        functions_with_types.append(function_metadata)

    with open(os.path.join(output_directory, 'functions.json'), 'w') as file:
        json.dump(functions, file, default=lambda o: o.__dict__, indent=4)
        file.close()

    with open(os.path.join(output_directory, 'functions_with_types.json'), 'w') as file:
        json.dump(functions_with_types, file, default=lambda o: o.__dict__, indent=4)
        file.close()


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
            for filename in list_files(filtered_project_directory):
                statistics['files'] += 1
                try:
                    functions = extractor.extract(read_file(filename))
                    statistics['functions'] += len(functions)
                    statistics['functions_with_types'] += sum(1 for function in functions if function.has_types())
                    project['files'].append({
                        'filename': filename,
                        'functions': functions
                    })
                except ParseError:
                    statistics['unparsable_files'] += 1

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

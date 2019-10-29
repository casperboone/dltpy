import os


class ProjectFilter:
    def filter_directory(self, directory: str, extension: str = '.py') -> str:
        """
        Delete all files within the given directory with filenames not ending in the given extension
        """
        for root, dirs, files in os.walk(directory):
            [os.remove(os.path.join(root, fi)) for fi in files if not fi.endswith(extension)]

        return directory

from git import Repo
import os
import shutil


class Cloner:
    def clone(self, author: str, repo: str, destination: str = './projects') -> str:
        """
        Clone projects from GitHub

        If the project already exists at the destination location, we replace it by the new clone
        """
        repo_url = 'https://github.com/' + author + '/' + repo + '.git'
        destination_path = os.path.join(destination, author + '__' + repo)

        if os.path.isdir(destination_path):
            shutil.rmtree(destination_path)

        Repo.clone_from(repo_url, destination_path, depth=1)

        return destination_path

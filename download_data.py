"""
Run this file to download data from Dryad. Downloaded files end up in this repostitory's
dryad_files/ directory.

First create the venv using the instructions in this repo's README.md. Then in a
Terminal, at this repository's top-level directory
(BCI-cursor-control-from-speech-motor-cortex/), run:

source ./venv/bin/activate
python download_data.py
"""

import sys
import os
import urllib.request
import json


########################################################################################
#
# Helpers.
#
########################################################################################


def display_progress_bar(block_num, block_size, total_size, message=""):
    """"""
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    MB_total = total_size / 1e6
    sys.stdout.write(
        f"\r{message}\t\t{MB_downloaded_so_far:.1f} MB / {MB_total:.1f} MB"
    )
    sys.stdout.flush()


########################################################################################
#
# Main function.
#
########################################################################################


def main():
    """"""
    DRYAD_DOI = "10.5061/dryad.prr4xgxzq"

    ## Make sure the command is being run from the right place and we can see the data
    ## directory.

    assert os.getcwd().endswith(
        "BCI-cursor-control-from-speech-motor-cortex"
    ), f"Please run the download command from the repo's top-level directory (instead of {os.getcwd()})"

    DATA_DIR = "dryad_files/"
    data_dirpath = os.path.abspath(DATA_DIR)
    os.makedirs(data_dirpath, exist_ok=True)
    assert os.path.exists(
        data_dirpath
    ), "Cannot find the data directory to download into."

    ## Get the list of files from the latest version on Dryad.

    DRYAD_ROOT = "https://datadryad.org"
    urlified_doi = DRYAD_DOI.replace("/", "%2F")

    versions_url = f"{DRYAD_ROOT}/api/v2/datasets/doi:{urlified_doi}/versions"
    with urllib.request.urlopen(versions_url) as response:
        versions_info = json.loads(response.read().decode())

    files_url_path = versions_info["_embedded"]["stash:versions"][-1]["_links"][
        "stash:files"
    ]["href"]

    # Loop through the pages of file infos until there is no "next" page.
    all_file_infos = []
    while True:
        files_url = f"{DRYAD_ROOT}{files_url_path}"
        with urllib.request.urlopen(files_url) as response:
            files_info = json.loads(response.read().decode())
        all_file_infos.extend(files_info["_embedded"]["stash:files"])
        files_url_path = files_info["_links"].get("next", {}).get("href")

        if files_url_path is None:
            break

    total_files = len(all_file_infos)

    ## Download each file into the data directory (and unzip for certain files).

    for file_num, file_info in enumerate(all_file_infos):
        filename = file_info["path"]
        download_path = file_info["_links"]["stash:download"]["href"]
        download_url = f"{DRYAD_ROOT}{download_path}"

        download_to_filepath = os.path.join(data_dirpath, filename)

        urllib.request.urlretrieve(
            download_url,
            download_to_filepath,
            reporthook=lambda *args: display_progress_bar(
                *args,
                message=f"({file_num + 1} / {total_files}) Downloading {filename}",
            ),
        )
        sys.stdout.write("\n")

    print(f"\nDownload complete. See data files in {data_dirpath}\n")


if __name__ == "__main__":
    main()

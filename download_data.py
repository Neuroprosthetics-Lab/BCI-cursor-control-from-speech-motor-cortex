import urllib.request
import sys
import os
import zipfile


########################################################################################
#
# Helpers.
#
########################################################################################


def display_progress_bar(block_num, block_size, total_size):
    """"""
    bytes_downloaded_so_far = block_num * block_size
    MB_downloaded_so_far = bytes_downloaded_so_far / 1e6
    sys.stdout.write(f"\r\t{MB_downloaded_so_far:.1f} MB")
    sys.stdout.flush()


########################################################################################
#
# Main function.
#
########################################################################################


def main():
    """"""
    DRYAD_DOI = "10.5061/dryad.prr4xgxzq"

    ## Make the directory for the data if it doesn't exist.

    DATA_DIR = "dryad_files/"
    os.makedirs(DATA_DIR, exist_ok=True)
    data_dirpath = os.path.abspath(DATA_DIR)

    ## Download the data as a zip file into the directory.

    urlified_doi = DRYAD_DOI.replace("/", "%2F")
    zip_url = f"https://datadryad.org/api/v2/datasets/doi:{urlified_doi}/download"

    ZIP_FILENAME = "dataset.zip"
    print(f"Downloading {ZIP_FILENAME} ...")

    zip_filepath = os.path.join(data_dirpath, ZIP_FILENAME)
    urllib.request.urlretrieve(zip_url, zip_filepath, reporthook=display_progress_bar)
    sys.stdout.write("\n")

    print("Download complete.")

    ## Unzip the data files into the directory.

    print(f"Extracting files from {ZIP_FILENAME} ...")

    with zipfile.ZipFile(zip_filepath, "r") as zf:
        zf.extractall(data_dirpath)

    os.remove(zip_filepath)

    print(f"Extraction complete. See data files in {data_dirpath}\n")


if __name__ == "__main__":
    main()

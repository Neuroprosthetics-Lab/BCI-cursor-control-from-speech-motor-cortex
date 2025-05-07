# BCI Cursor Control From Speech Motor Cortex

This repo is contains code and data for [Singer-Clark et al. 2025, Speech motor cortex enables BCI cursor control and click](https://doi.org/10.1088/1741-2552/add0e5).

## Data

### Downloading the data

Data can be found [here, hosted on Dryad](TODO: put Dryad doi here).

In a Terminal, navigate to this repo's root directory (`BCI-cursor-control-from-speech-motor-cortex/`), and run the command `TODO` to download the data from Dryad into a folder named `dryad_files/` inside this repo. The Python scripts in this repo assume that is the location of the data.

### Data format

TODO: copy the relevant section from the Dryad README once that is finalized.

## Code

### Setup

Install the relevant dependencies (`numpy`, `scipy`, `matplotlib`, etc.) to your Python environment using `pip install -r requirements.txt`. It is recommended to use an environment manager such as `venv` or `conda` to create a standalone environment before installing dependencies.

### Running examples

This repo contains example scripts (e.g., `example_figure1_first_ever_cursor_BCI_usage.py`) which generate some figures from the paper. In addition to generating figures, these scripts also serve as boilerplate code from which you can develop your own analyses. These examples demonstrate how to load the data, extract relevant fields, and visualize things.

1. In a Terminal, navigate to this repo's root directory (`BCI-cursor-control-from-speech-motor-cortex/`).
2. If using a `venv` or `conda` environment, activate it (e.g., `conda activate your_env_name`).
3. Run `python example_figure1_first_ever_cursor_BCI_usage.py`.
4. If you prefer interactive notebooks, you can instead run the corresponding notebook `example_figure1_first_ever_cursor_BCI_usage.ipynb` using the notebook tool of your choice.

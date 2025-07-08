# BCI Cursor Control From Speech Motor Cortex

This repo contains code and data for [Singer-Clark et al. 2025, Speech motor cortex enables BCI cursor control and click](https://doi.org/10.1088/1741-2552/add0e5).

Contact Tyler at tsingerclark@ucdavis.edu if you have any questions.

![Radial8 cursor trajectories.](./images/radial8_trajectories.png | width=100) ![Radial8 cursor trajectories.](./images/electrode_227.png | width=100)

## Code

### Setup

Install the relevant dependencies (`numpy`, `scipy`, `matplotlib`, etc.) to your Python environment using `pip install -r requirements.txt`. It is recommended to use an environment manager such as `venv` or `conda` to create a standalone environment before installing dependencies.

### Running examples

This repo contains example scripts (e.g., `example_figure1_first_ever_cursor_BCI_usage.py`) which generate some figures from the paper. In addition to generating figures, these scripts also serve as boilerplate code from which you can develop your own analyses. These examples demonstrate how to load the data, extract relevant fields, and visualize things.

1. In a Terminal, navigate to this repo's root directory (`BCI-cursor-control-from-speech-motor-cortex/`).
2. If using a `venv` or `conda` environment, activate it (e.g., `conda activate your_env_name`).
3. Run an example script (e.g., `python example_figure1_first_ever_cursor_BCI_usage.py`).
4. (Optional) If you prefer interactive notebooks, you can instead run the corresponding notebook (e.g., `example_figure1_first_ever_cursor_BCI_usage.ipynb`) using the notebook tool of your choice.

## Data

### Downloading the data

Data can be found [here, hosted on Dryad](https://doi.org/10.5061/dryad.prr4xgxzq). (NOTE: Dryad publication pending. Code available for now without the data itself. Data will be available soon.)

Download it automatically by running `python download_data.py`.

Alternatively, you can manually download and unzip the data from Dryad into a folder named `dryad_files/` inside this repo. The Python scripts in this repo assume that is the location of the data.

```
BCI-cursor-control-from-speech-motor-cortex/
--> dryad_files/
    --> t15_day00039_block00_radial8_calibration_task.mat
    --> t15_day00039_block01_radial8_calibration_task.mat
    ...
--> example_figure1_first_ever_cursor_BCI_usage.ipynb
--> example_figure1_first_ever_cursor_BCI_usage.py
...
```

### Data format

#### Naming convention for `.mat` files

Filenames start with the participant (e.g., `t15`), followed by a day identifier (e.g., `day00039`), a block identifier (e.g., `block02`), and a task name (e.g., `radial8_calibration_task`).

Each `.mat` file represents one task block, ~3-5 minutes in length.

#### Contents of `.mat` files

Streams of data (e.g., task state, neural data) are contained in `.mat` files (e.g., `t15_day00039_block00_radial8_calibration_task.mat`), which can be loaded with Python's [`scipy.io.loadmat`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.io.loadmat.html).

To inspect the structure of these files, install `scipy` to your Python environment (e.g., `pip install scipy`), and follow the example below.

```
import scipy.io

block_data = scipy.io.loadmat("./t15_day00039_block00_radial8_calibration_task.mat")

print(block_data.keys())
# dict_keys(['__header__', '__version__', '__globals__', 'timestamp_sec', 'threshold_crossings', 'spike_band_power', 'assist_amount', 'cursor_position', 'target_position', 'trial_idx', 'cursor_decoder_output', 'trial_start_bin', 'target_radius', 'cursor_radius', 'array_label_by_electrode', 'dwell_requirement_sec'])
print(block_data["threshold_crossings"].shape)
# (17986, 256)
print(block_data["spike_band_power"].shape)
# (17986, 256)
print(block_data["cursor_position"].shape)
# (17986, 2)
print(block_data["target_position"].shape)
# (17986, 2)
```

- **Per-bin fields**
  - `timestamp_sec`
    - Each time bin's timestamp (in seconds) from the start of the block. Each time bin represents 10 ms.
  - `threshold_crossings`
    - A neural feature field. The number of times (within each time bin) each electrode's signal (after filtering) crossed an electrode-specific threshold. A proxy for action potentials (without fully spike sorting). First dimension is time bin and second dimension is electrode, so the shape of this array is something like `(18000, 256)`.
  - `spike_band_power`
    - A neural feature field. The average squared value (within each time bin) of each electrode's signal (after filtering). First dimension is bin and second dimension is electrode. First dimension is time bin and second dimension is electrode, so the shape of this array is something like `(18000, 256)`.
  - `assist_amount`
    - Number between `0.0` and `1.0`. A value of `0.0` means the cursor movements being determined fully from the neural decoder ("closed-loop control"). A value of `1.0` means the cursor movements were being determined fully by computer assistance which moves the cursor toward the cued target ("open-loop control"). A value in between means the cursor movements were a weighted average of the neural decoder's vector and the computer assistance vector which points directly toward the target at a constant speed. This value varies during some calibration blocks, starting at `1.0` and ending up at `0.0`.
  - `click_assist`
    - Boolean, where `false` means the target selection clicks were being determined from the neural decoder, and `true` means the target selections were being determined by computer assistance after the cursor touched the target for a certain amount of time. Sessions with no click involved do not have this field.
  - `cursor_position`
    - The location of the center of the cursor on the screen. `(0, 0)` is the center of the screen, positive `x` is to the right, positive `y` is up, and a distance of `1.0` equals the shortest screen dimension (in this case, the screen height). For the remaining fields, all positions, velocities, and lengths also use this unit system.
  - `target_position`
    - Center of the cued target for the current trial, whether the target is a circle or a square. It should be constant during each trial, and then change when a new trial begins.
  - `trial_idx`
    - A counter that identifies the current trial number. It should be constant during each trial, and then increment by 1 when a new trial begins.
  - `cursor_decoder_output`
    - The output of the neural cursor decoder that was used online during the task. These are velocities in units / second. When `assist_amount` is `0.0`, these are fully what is determining the cursor movement, so the cursor position should move each bin by this amount times the bin width (10 ms, or 0.01 seconds).
  - `click_decoder_output`
    - The output of the neural click decoder that was used online during the task. Values are either `0` (for no click) or `1` (for click). Sessions with no click involved do not have this field.
- **Per-trial fields**
  - `trial_start_bin`
    - The bin index (i.e., index into the binned field arrays above) of when each new trial begins.
  - `target_presentation_bin`
    - In the Simultaneous Speech and Cursor Task, the bin index of when the cursor target and prompted word are displayed.
  - `cursor_go_cue_bin`
    - In the Simultaneous Speech and Cursor Task, the bin index of when the cursor target and prompted word are displayed.
  - `speech_go_cue_bin`
    - In the Simultaneous Speech and Cursor Task, the bin index of when the audible beep played. Not all trials have a beep. For trials without a beep, this field's value is `-1`.
  - `trial_end_bin`
    - In the Simultaneous Speech and Cursor Task, the bin index of when the trial ends, either by target selection or trial timeout.
  - `speech_prompt`
    - In the Simultaneous Speech and Cursor Task, the word that was prompted above the target when the target was presented.
- **Per-block fields**
  - `array_label_by_electrode`
    - Identifier of the microelectrode array location in the brain (one of `v6v`, `d6v`, `4`, or `55b`) that each electrode belongs to. This lets you look at the neural features fields (`threshold_crossings` and `spike_band_power`) and know which brain area each signal is coming from.
  - `cursor_radius`
    - Radius of the cursor circle, in the same length units as `cursor_position`, `target_position`, etc. In the Radial8 Calibration Task as well as the Simultaneous Speech and Cursor Task (which both use circular targets), for the cursor to be considered "touching" the cued target, the distance from `cursor_position` to `target_position` must be within the `cursor_radius` plus the `target_radius`.
  - `target_radius`
    - In the Radial8 Calibration Task as well as the Simultaneous Speech and Cursor Task, this is the radius of each target, in the same length units as `cursor_position`, `target_position`, etc. For the cursor to be considered "touching" the cued target, the distance from `cursor_position` to `target_position` must be within the `cursor_radius` plus the `target_radius`.
  - `dwell_requirement_sec`
    - For days 39 and 202 (when click is not involved), this field is how long the cursor must be touching the cued target to be considered a selection and end the trial.
  - `grid_num_rows`
    - In the Grid Evaluation Task, this is the number of rows (and columns) in the grid of square targets. In combination with `grid_total_height`, this tells you the height (and width) of each grid target. For the cursor to be considered "touching" the cued target, `cursor_position` must be within the grid cell's width and height.
  - `grid_total_height`
    - In the Grid Evaluation Task, this is the height (and width) of the full grid of targets, in the same length units as `cursor_position`, `target_position`, etc. In combination with `grid_num_rows`, this tells you the height (and width) of each grid target. For the cursor to be considered "touching" the cued target, `cursor_position` must be within the grid cell's width and height.
  - `is_control_block`
    - In the Simultaneous Speech and Cursor Task, this boolean says whether the block was a "control" block or "verbal" block. During "verbal" blocks the participant spoke after each speech go cue (a beep). During "control" blocks, the speech go cue (a beep) still occurred, but the participant did not speak.

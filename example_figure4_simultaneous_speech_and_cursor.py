import numpy as np
import scipy.io
from scipy.ndimage import gaussian_filter1d
from matplotlib.patches import Circle
import matplotlib.pyplot as plt


########################################################################################
#
# Constants.
#
########################################################################################

TARGET_COLORS = [
    (0.9254902, 0.12156863, 0.14117647),
    (0.98431373, 0.72941176, 0.07058824),
    (0.57254902, 0.78431373, 0.24313725),
    (0.384, 0.682, 0.2),
    (0.43137255, 0.79607843, 0.85490196),
    (0.26529412, 0.40686275, 0.72490196),
    (0.45568627, 0.31764706, 0.63529412),
    (0.84705882, 0.2627451, 0.59215686),
]
# Lighter versions of the target colors.
TRAJECTORY_COLORS = [
    (0.9627451, 0.22156863, 0.24117647),
    (0.992156865, 0.829411759, 0.17058824),
    (0.672549019, 0.88431373, 0.34313725),
    (0.45352941, 0.78117647, 0.31058824),
    (0.53137255, 0.89607843, 0.927450979),
    (0.36529412, 0.52686275, 0.84490196),
    (0.55568627, 0.417647059, 0.735294119),
    (0.92352941, 0.36274510, 0.69215686),
]
PROMPTS = ["bah", "though", "day", "kite", "choice", "veto", "were"]
PROMPT_COLORS = {
    "bah": (0.71, 0.84, 0.44),
    "though": (0.75, 0.5, 0.75),
    "day": (0.88, 0.88, 0.33),
    "kite": (0.55, 0.81, 0.77),
    "choice": (0.67, 0.67, 0.67),
    "veto": (1.0, 0.5, 0.5),
    "were": (0.9, 0.7, 0.4),
}


########################################################################################
#
# Helpers.
#
########################################################################################


def get_direction_idx_from_vector(vector):
    """
    Given a 2D vector, get an integer from 0 through 7 corresponding to the 1/8th slice
    of the unit circle it falls in. Good for identifying a radial8 target, or a position
    near a radial8 target.
    """
    x, y = vector

    target_angle = np.arctan2(y, x)
    if target_angle < 0:
        target_angle += 2 * np.pi

    direction_idx = int(np.round(target_angle / (np.pi / 4))) % 8

    return direction_idx


########################################################################################
#
# Main function.
#
########################################################################################


def main():
    """"""

    ## Load the simul task blocks of data from the Simultaneous Speech and Cursor
    ## Session.

    filepaths = [
        "./dryad_files/t15_day00202_block02_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block03_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block04_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block05_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block06_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block07_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block10_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block11_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block12_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block13_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block14_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block15_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block16_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block17_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block18_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block21_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block22_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block23_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block24_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block25_simultaneous_speech_and_cursor_task.mat",
        "./dryad_files/t15_day00202_block26_simultaneous_speech_and_cursor_task.mat",
    ]
    try:
        data = [scipy.io.loadmat(filepath) for filepath in filepaths]
    except FileNotFoundError:
        print(
            "ERROR: Data files not found. Follow steps in the README to download data."
        )

    ## Calculate target acquisition times and group them by task condition.

    verbal_blocks_beep_trials = []
    verbal_blocks_nobeep_trials = []
    control_blocks_beep_trials = []
    control_blocks_nobeep_trials = []

    # The 21 simul blocks (A = verbal, B = control) were collected in 3 sets as follows:
    # 1. A A B A A B
    # 2. A A B A A B A A B
    # 3. A A B A A B
    # with a break and some cursor calibration before each set.

    # To allow for a fair comparison between verbal and control blocks, we should have
    # an A B A structure from each set. Achieve this by excluding the first A in each
    # set and the last A B in each set.
    ABA_data = np.concatenate([data[1:4], data[7:13], data[16:19]])

    for block_data in ABA_data:
        timestamps = block_data["timestamp_sec"].flatten()
        cursor_go_cue_bins = block_data["cursor_go_cue_bin"].flatten()
        speech_go_cue_bins = block_data["speech_go_cue_bin"].flatten()
        trial_end_bins = block_data["trial_end_bin"].flatten()
        is_control_block = block_data["is_control_block"].item()
        is_verbal_block = not is_control_block

        for trial_idx in range(len(cursor_go_cue_bins)):
            cursor_go_cue_bin = cursor_go_cue_bins[trial_idx]
            speech_go_cue_bin = speech_go_cue_bins[trial_idx]
            trial_end_bin = trial_end_bins[trial_idx]

            # If there was no beep in this trial, the speech go cue bin is -1.
            is_beep_trial = speech_go_cue_bin != -1

            trial_end_timestamp = timestamps[trial_end_bin]
            cursor_go_cue_timestamp = timestamps[cursor_go_cue_bin]
            target_acquisition_time = trial_end_timestamp - cursor_go_cue_timestamp

            if is_verbal_block:
                if is_beep_trial:
                    verbal_blocks_beep_trials.append(target_acquisition_time)
                else:
                    verbal_blocks_nobeep_trials.append(target_acquisition_time)
            else:
                if is_beep_trial:
                    control_blocks_beep_trials.append(target_acquisition_time)
                else:
                    control_blocks_nobeep_trials.append(target_acquisition_time)

    ## Plot a boxplot of trial lengths for each condition.

    fig, ax = plt.subplots()

    condition_trial_times = [
        control_blocks_nobeep_trials,
        control_blocks_beep_trials,
        verbal_blocks_nobeep_trials,
        verbal_blocks_beep_trials,
    ]
    condition_x_positions = [0.5, 1.5, 3.0, 4.0]
    condition_trial_types = [
        "no beep\ntrials",
        "beep\ntrials",
        "no beep\ntrials",
        "beep\ntrials",
    ]
    condition_colors = [
        (0.6, 0.6, 0.9),
        (1.0, 0.55, 0.15),
        (0.2, 0.27, 0.64),
        (1.0, 0.4, 0.0),
    ]

    boxplot = ax.boxplot(
        condition_trial_times,
        positions=condition_x_positions,
        tick_labels=condition_trial_types,
        widths=0.5,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 4},
        flierprops={
            "markerfacecolor": (0.5, 0.5, 0.5),
            "markersize": 6,
            "markeredgecolor": "none",
            "clip_on": False,
        },
        capprops={"linewidth": 3, "color": (0.3, 0.3, 0.3)},
        whiskerprops={"linewidth": 3, "color": (0.3, 0.3, 0.3)},
    )
    for box_idx, box in enumerate(boxplot["boxes"]):
        box.set_facecolor(condition_colors[box_idx])
        box.set_linestyle("none")

    # Style the plots.

    ax.set_xlim(-0.25, 4.6)
    ax.tick_params(axis="x", labelsize=16, length=0, pad=-55)
    ax.set_xticklabels(ax.get_xticklabels(), weight="bold")
    for tick_idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(condition_colors[tick_idx])
    ax.text(
        np.mean(condition_x_positions[:2]),
        -0.5,
        "control\nblocks",
        color=(0.4, 0.4, 0.4),
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )
    ax.text(
        np.mean(condition_x_positions[-2:]),
        -0.5,
        "verbal\nblocks",
        color=(0.1, 0.1, 0.1),
        ha="center",
        va="center",
        fontsize=18,
        fontweight="bold",
    )

    ax.set_ylim(0, 10)
    ax.set_yticks(
        np.arange(11), labels=[i if i in [0, 5, 10] else "" for i in np.arange(11)]
    )
    ax.tick_params(axis="y", labelsize=18)
    ax.set_ylabel("target acquisition time (s)", fontsize=20, labelpad=10)

    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.tick_params(axis="y", length=7, width=3)

    plt.tight_layout()
    plt.show()

    ## Trial-average the neural activity, aligned to different stages of the trial.

    presentation_windows_grouped_by_direction = {
        direction_idx: [] for direction_idx in range(8)
    }
    cursor_go_cue_windows_grouped_by_direction = {
        direction_idx: [] for direction_idx in range(8)
    }
    speech_go_cue_windows_grouped_by_prompt = {prompt: [] for prompt in PROMPTS}

    PRE_GO_CUE_sec = 0.5
    POST_GO_CUE_sec = 1.0

    BIN_WIDTH_sec = 0.01
    PRE_GO_CUE_bins = int(PRE_GO_CUE_sec / BIN_WIDTH_sec)
    POST_GO_CUE_bins = int(POST_GO_CUE_sec / BIN_WIDTH_sec)

    for block_data in data:
        threshold_crossings = block_data["threshold_crossings"]
        target_positions = block_data["target_position"]
        target_presentation_bins = block_data["target_presentation_bin"].flatten()
        cursor_go_cue_bins = block_data["cursor_go_cue_bin"].flatten()
        speech_go_cue_bins = block_data["speech_go_cue_bin"].flatten()
        speech_prompts = [s.item() for s in block_data["speech_prompt"].flatten()]
        is_control_block = block_data["is_control_block"].item()
        is_verbal_block = not is_control_block

        # Scale threshold crossings values to represent firing rates in Hz.
        firing_rates = threshold_crossings / BIN_WIDTH_sec
        # Apply smoothing.
        SMOOTHING_SIGMA = 5
        firing_rates = gaussian_filter1d(firing_rates, sigma=SMOOTHING_SIGMA, axis=0)

        total_bins = len(firing_rates)

        ## Windows aligned to target presentation and to cursor go cue.

        for trial_idx in range(len(target_presentation_bins)):
            target_presentation_bin = target_presentation_bins[trial_idx]
            cursor_go_cue_bin = cursor_go_cue_bins[trial_idx]
            speech_go_cue_bin = speech_go_cue_bins[trial_idx]

            # Skip trials with a beep.
            if speech_go_cue_bin != -1:
                continue

            trial_target = target_positions[target_presentation_bin]

            # Skip trials toward the center target (the user can anticipate the target).
            CENTER_TARGET = np.array([0.0, 0.0])
            is_toward_center_target = np.all(trial_target == CENTER_TARGET)
            if is_toward_center_target:
                continue

            direction_idx = get_direction_idx_from_vector(trial_target)

            # Window aligned to target presentation.

            presentation_window_start_bin = target_presentation_bin - PRE_GO_CUE_bins
            presentation_window_end_bin = target_presentation_bin + POST_GO_CUE_bins

            presentation_window = firing_rates[
                presentation_window_start_bin:presentation_window_end_bin
            ]

            presentation_windows_grouped_by_direction[direction_idx].append(
                presentation_window
            )

            # Window aligned to cursor go cue.

            cursor_go_cue_window_start_bin = cursor_go_cue_bin - PRE_GO_CUE_bins
            cursor_go_cue_window_end_bin = cursor_go_cue_bin + POST_GO_CUE_bins

            cursor_go_cue_window = firing_rates[
                cursor_go_cue_window_start_bin:cursor_go_cue_window_end_bin
            ]

            cursor_go_cue_windows_grouped_by_direction[direction_idx].append(
                cursor_go_cue_window
            )

        ## Windows aligned to speech go cue.

        for trial_idx in range(len(speech_go_cue_bins)):
            speech_go_cue_bin = speech_go_cue_bins[trial_idx]
            speech_prompt = speech_prompts[trial_idx]

            # Skip trials with no beep.
            if speech_go_cue_bin == -1:
                continue

            # Skip trials in control blocks (since control blocks don't have speech).
            if is_control_block:
                continue

            speech_go_cue_window_start_bin = speech_go_cue_bin - PRE_GO_CUE_bins
            speech_go_cue_window_end_bin = speech_go_cue_bin + POST_GO_CUE_bins

            # Skip windows at the end of the block which go outside the block.
            if speech_go_cue_window_end_bin > total_bins:
                continue

            speech_go_cue_window = firing_rates[
                speech_go_cue_window_start_bin:speech_go_cue_window_end_bin
            ]

            speech_go_cue_windows_grouped_by_prompt[speech_prompt].append(
                speech_go_cue_window
            )

    # Average across trials.
    presentation_trial_averaged_by_direction = {
        direction_idx: np.mean(neural_windows, axis=0)
        for direction_idx, neural_windows in presentation_windows_grouped_by_direction.items()
    }
    cursor_go_cue_trial_averaged_by_direction = {
        direction_idx: np.mean(neural_windows, axis=0)
        for direction_idx, neural_windows in cursor_go_cue_windows_grouped_by_direction.items()
    }
    speech_go_cue_trial_averaged_by_prompt = {
        prompt: np.mean(neural_windows, axis=0)
        for prompt, neural_windows in speech_go_cue_windows_grouped_by_prompt.items()
    }
    # Get the standard error of the mean.
    presentation_sem_by_direction = {
        direction_idx: np.std(neural_windows, axis=0) / np.sqrt(len(neural_windows))
        for direction_idx, neural_windows in presentation_windows_grouped_by_direction.items()
    }
    cursor_go_cue_sem_by_direction = {
        direction_idx: np.std(neural_windows, axis=0) / np.sqrt(len(neural_windows))
        for direction_idx, neural_windows in cursor_go_cue_windows_grouped_by_direction.items()
    }
    speech_go_cue_sem_by_direction = {
        prompt: np.std(neural_windows, axis=0) / np.sqrt(len(neural_windows))
        for prompt, neural_windows in speech_go_cue_windows_grouped_by_prompt.items()
    }

    ## Plot individual channels' trial-averaged firing rates aligned to different stages
    ## of the trial.

    SELECTED_ELECTRODES = [229, 165, 247]
    num_bins_in_window = int((PRE_GO_CUE_sec + POST_GO_CUE_sec) / BIN_WIDTH_sec)
    relative_timestamps = np.linspace(
        -PRE_GO_CUE_sec, POST_GO_CUE_sec, num_bins_in_window
    )

    for electrode_idx in SELECTED_ELECTRODES:
        fig, (presentation_ax, cursor_go_cue_ax, speech_go_cue_ax) = plt.subplots(1, 3)

        # Plot activity aligned to target presentation.
        for direction_idx in range(8):
            trial_averaged = presentation_trial_averaged_by_direction[direction_idx][
                :, electrode_idx
            ]
            sem = presentation_sem_by_direction[direction_idx][:, electrode_idx]
            color = TARGET_COLORS[direction_idx]
            presentation_ax.plot(
                relative_timestamps, trial_averaged, color=color, linewidth=2
            )
            presentation_ax.fill_between(
                relative_timestamps,
                trial_averaged - sem,
                trial_averaged + sem,
                color=color,
                alpha=0.1,
                edgecolor="none",
            )

            # Add a dot for the target presentation.
            presentation_ax.scatter(
                [0.0], [-4.0], marker="o", s=95, color=(0.2, 0.2, 0.2), clip_on=False
            )
            presentation_ax.text(
                0.0, -9.0, "target\npresentation", ha="center", va="top", fontsize=16
            )

            # Add a scale bar for time.
            presentation_ax.hlines(
                -4.0,
                POST_GO_CUE_sec - 0.5,
                POST_GO_CUE_sec,
                color=(0.2, 0.2, 0.2),
                linewidth=3,
                clip_on=False,
            )
            presentation_ax.text(
                POST_GO_CUE_sec,
                -7.0,
                f"{int(0.5 * 1000)} ms",
                ha="right",
                va="top",
                fontsize=14,
            )

            # Style the plot.
            presentation_ax.set_xlim(-PRE_GO_CUE_sec, POST_GO_CUE_sec)
            presentation_ax.tick_params(bottom=False, labelbottom=False)
            presentation_ax.set_ylim(0, 85)
            presentation_ax.set_yticks([0, 85])
            presentation_ax.tick_params(axis="y", width=3, length=8, labelsize=20)
            presentation_ax.set_ylabel("firing rate (Hz)", fontsize=24, labelpad=10)
            presentation_ax.spines["top"].set_visible(False)
            presentation_ax.spines["right"].set_visible(False)
            presentation_ax.spines["bottom"].set_visible(False)
            presentation_ax.spines["left"].set_position(("data", -PRE_GO_CUE_sec - 0.1))
            presentation_ax.spines["left"].set_linewidth(3)

        # Plot activity aligned to cursor go cue.
        for direction_idx in range(8):
            trial_averaged = cursor_go_cue_trial_averaged_by_direction[direction_idx][
                :, electrode_idx
            ]
            sem = cursor_go_cue_sem_by_direction[direction_idx][:, electrode_idx]
            color = TARGET_COLORS[direction_idx]
            cursor_go_cue_ax.plot(
                relative_timestamps, trial_averaged, color=color, linewidth=2
            )
            cursor_go_cue_ax.fill_between(
                relative_timestamps,
                trial_averaged - sem,
                trial_averaged + sem,
                color=color,
                alpha=0.1,
                edgecolor="none",
            )

            # Add a dot for the cursor go cue.
            cursor_go_cue_ax.scatter(
                [0.0], [-4.0], marker="o", s=95, color=(0.2, 0.2, 0.2), clip_on=False
            )
            cursor_go_cue_ax.text(
                0.0, -9.0, "cursor\ngo cue", ha="center", va="top", fontsize=16
            )

            # Style the plot.
            cursor_go_cue_ax.set_xlim(-PRE_GO_CUE_sec, POST_GO_CUE_sec)
            cursor_go_cue_ax.set_ylim(0, 85)
            cursor_go_cue_ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            cursor_go_cue_ax.spines["top"].set_visible(False)
            cursor_go_cue_ax.spines["right"].set_visible(False)
            cursor_go_cue_ax.spines["bottom"].set_visible(False)
            cursor_go_cue_ax.spines["left"].set_visible(False)

        # Plot activity aligned to speech go cue.
        for prompt in PROMPTS:
            trial_averaged = speech_go_cue_trial_averaged_by_prompt[prompt][
                :, electrode_idx
            ]
            sem = speech_go_cue_sem_by_direction[prompt][:, electrode_idx]
            color = PROMPT_COLORS[prompt]
            speech_go_cue_ax.plot(
                relative_timestamps, trial_averaged, color=color, linewidth=2
            )
            speech_go_cue_ax.fill_between(
                relative_timestamps,
                trial_averaged - sem,
                trial_averaged + sem,
                color=color,
                alpha=0.1,
                edgecolor="none",
            )

            # Add a dot for the speech go cue.
            speech_go_cue_ax.scatter(
                [0.0], [-4.0], marker="o", s=95, color=(0.2, 0.2, 0.2), clip_on=False
            )
            speech_go_cue_ax.text(
                0.0, -9.0, "speech\ngo cue", ha="center", va="top", fontsize=16
            )

            # Style the plot.
            speech_go_cue_ax.set_xlim(-PRE_GO_CUE_sec, POST_GO_CUE_sec)
            speech_go_cue_ax.set_ylim(0, 85)
            speech_go_cue_ax.tick_params(
                left=False, bottom=False, labelleft=False, labelbottom=False
            )
            speech_go_cue_ax.spines["top"].set_visible(False)
            speech_go_cue_ax.spines["right"].set_visible(False)
            speech_go_cue_ax.spines["bottom"].set_visible(False)
            speech_go_cue_ax.spines["left"].set_visible(False)

        array_label = data[0]["array_label_by_electrode"][electrode_idx].strip()
        fig.suptitle(f"electrode {electrode_idx}\n(array {array_label})", fontsize=20)
        fig.set_figwidth(13)
        fig.set_figheight(5)
        fig.subplots_adjust(bottom=0.2, left=0.16, wspace=0.12)

        plt.show()


if __name__ == "__main__":
    main()

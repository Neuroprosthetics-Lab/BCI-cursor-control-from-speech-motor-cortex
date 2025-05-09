import numpy as np
import scipy.io
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

    ## Load the radial8 blocks of data from the Simultaneous Speech and Cursor Session.

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
            is_beep_trial = speech_go_cue_bin > 0

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
    condition_x_positions = [0.5, 1.6, 3.0, 4.1]
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
        widths=0.7,
        patch_artist=True,
        medianprops={"color": "white", "linewidth": 4},
        flierprops={
            "markerfacecolor": (0.5, 0.5, 0.5),
            "markersize": 9,
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
    ax.tick_params(axis="x", labelsize=30, length=0, pad=-120)
    ax.set_xticklabels(ax.get_xticklabels(), weight="bold")
    for tick_idx, tick in enumerate(ax.get_xticklabels()):
        tick.set_color(condition_colors[tick_idx])
    ax.text(
        np.mean(condition_x_positions[:2]),
        0.1,
        "control\nblocks",
        color=(0.4, 0.4, 0.4),
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
    )
    ax.text(
        np.mean(condition_x_positions[-2:]),
        0.1,
        "verbal\nblocks",
        color=(0.1, 0.1, 0.1),
        ha="center",
        va="center",
        fontsize=30,
        fontweight="bold",
    )

    ax.set_ylim(0, 10)
    ax.set_yticks(
        np.arange(11), labels=[i if i in [0, 5, 10] else "" for i in np.arange(11)]
    )
    ax.tick_params(axis="y", labelsize=40, pad=10)
    ax.set_ylabel("target acquisition time (s)", fontsize=40, labelpad=10)

    ax.spines[["top", "right", "bottom"]].set_visible(False)
    ax.spines["left"].set_linewidth(3)
    ax.tick_params(axis="y", length=7, width=3)

    fig.set_figheight(14)
    fig.set_figwidth(11)
    fig.subplots_adjust(top=0.95, bottom=0.2, left=0.2)

    plt.show()


if __name__ == "__main__":
    main()

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

    ## Load the blocks of data from the First-ever Usage Session.

    filepaths = [
        "./dryad_files/t15_day00039_block00_radial8_calibration_task.mat",
        "./dryad_files/t15_day00039_block01_radial8_calibration_task.mat",
        "./dryad_files/t15_day00039_block02_radial8_calibration_task.mat",
        "./dryad_files/t15_day00039_block03_radial8_calibration_task.mat",
        "./dryad_files/t15_day00039_block04_radial8_calibration_task.mat",
        "./dryad_files/t15_day00039_block05_radial8_calibration_task.mat",
    ]
    try:
        data = [scipy.io.loadmat(filepath) for filepath in filepaths]
    except FileNotFoundError:
        print(
            "ERROR: Data files not found. Follow steps in the README to download data."
        )

    ## Draw the cursor trajectories for the center-out-and-back movements for all the
    ## specified blocks.

    fig, ax = plt.subplots()

    unique_target_positions = set()

    for block_data in data:
        cursor_positions = block_data["cursor_position"]
        target_positions = block_data["target_position"]
        assist_amounts = block_data["assist_amount"].flatten()
        trial_start_bins = block_data["trial_start_bin"].flatten()

        for trial_idx, trial_start_bin in enumerate(trial_start_bins):
            # If this trial used any assist, don't draw it. Only draw fully closed-loop
            # trials.
            starting_assist_amount = assist_amounts[trial_start_bin]
            if starting_assist_amount > 0.0:
                continue

            # Each trajectory we draw will start with a movement toward an outer target.
            trial_target = target_positions[trial_start_bin]
            CENTER_TARGET = np.array([0.0, 0.0])
            is_toward_center_target = np.all(trial_target == CENTER_TARGET)
            if is_toward_center_target:
                continue

            # If the block ends during this center-out-and-back, don't draw it.
            if trial_idx + 2 >= len(trial_start_bins):
                continue

            ## Draw the center-out-and-back trajectory.

            # Get the range of bins representing the full center-out-and-back, which
            # includes the center-out trial plus the following trial back to center.
            trajectory_start_bin = trial_start_bin
            trajectory_end_bin = trial_start_bins[trial_idx + 2]

            trajectory = cursor_positions[trajectory_start_bin:trajectory_end_bin]

            # Color the trajectory based on the outer target.
            direction_idx = get_direction_idx_from_vector(trial_target)
            trajectory_color = TRAJECTORY_COLORS[direction_idx]

            ax.plot(trajectory[:, 0], trajectory[:, 1], color=trajectory_color)

            unique_target_positions.add(tuple(trial_target))

    # Draw the target circles.
    target_radius = data[0]["target_radius"].item()
    cursor_radius = data[0]["cursor_radius"].item()
    touching_radius = target_radius + cursor_radius
    for target_position in unique_target_positions:
        if not np.array_equal(target_position, np.array([0, 0])):
            direction_idx = get_direction_idx_from_vector(target_position)
            target_color = TARGET_COLORS[direction_idx]
            target_circle = Circle(
                target_position,
                touching_radius,
                edgecolor=target_color,
                facecolor=target_color,
            )
            ax.add_patch(target_circle)

    # Style the plot.
    ax.set_xlim(-0.65, 0.65)
    ax.set_ylim(-0.65, 0.65)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_axis_off()
    # Scale bar.
    scale_bar_length_px = 200
    px_per_bgunit = 1080  # screen height in pixels
    scale_bar_length_bgunits = scale_bar_length_px / px_per_bgunit
    ax.hlines(
        -0.5,
        -0.45,
        -0.45 + scale_bar_length_bgunits,
        linewidth=5,
        color=(0.1, 0.1, 0.1),
    )
    ax.text(-0.45, -0.45, f"{scale_bar_length_px} px", fontsize=12)

    plt.tight_layout()
    plt.show()

    ## Trial-average the neural activity for each direction of outer target.

    neural_windows_grouped_by_direction = {
        direction_idx: [] for direction_idx in range(8)
    }

    PRE_GO_CUE_sec = 0.5
    POST_GO_CUE_sec = 1.0

    BIN_WIDTH_sec = 0.01
    PRE_GO_CUE_bins = int(PRE_GO_CUE_sec / BIN_WIDTH_sec)
    POST_GO_CUE_bins = int(POST_GO_CUE_sec / BIN_WIDTH_sec)

    for block_data in data:
        threshold_crossings = block_data["threshold_crossings"]
        target_positions = block_data["target_position"]
        trial_start_bins = block_data["trial_start_bin"].flatten()

        # Scale threshold crossings values to represent firing rates in Hz.
        firing_rates = threshold_crossings / BIN_WIDTH_sec
        # Apply smoothing.
        SMOOTHING_SIGMA = 5
        firing_rates = gaussian_filter1d(firing_rates, sigma=SMOOTHING_SIGMA, axis=0)

        for trial_start_bin in trial_start_bins:
            # Only include center-out trials (the user cannot anticipate the target).
            trial_target = target_positions[trial_start_bin]
            is_toward_center_target = np.all(trial_target == CENTER_TARGET)
            if is_toward_center_target:
                continue

            neural_window_start_bin = trial_start_bin - PRE_GO_CUE_bins
            neural_window_end_bin = trial_start_bin + POST_GO_CUE_bins

            # Skip trials at the start or end of the block which go outside the block.
            total_bins = len(firing_rates)
            if neural_window_start_bin < 0 or neural_window_end_bin > total_bins:
                continue

            neural_window = firing_rates[neural_window_start_bin:neural_window_end_bin]

            direction_idx = get_direction_idx_from_vector(trial_target)
            neural_windows_grouped_by_direction[direction_idx].append(neural_window)

    # Average across trials for each direction.
    trial_averaged_by_direction = {
        direction_idx: np.mean(neural_windows, axis=0)
        for direction_idx, neural_windows in neural_windows_grouped_by_direction.items()
    }
    # Get the standard error of the mean for each direction.
    sem_by_direction = {
        direction_idx: np.std(neural_windows, axis=0) / np.sqrt(len(neural_windows))
        for direction_idx, neural_windows in neural_windows_grouped_by_direction.items()
    }

    ## Plot the trial-averaged firing rates for a select few electrodes.

    SELECTED_ELECTRODES = [227, 236, 122]
    num_bins_in_window = int((PRE_GO_CUE_sec + POST_GO_CUE_sec) / BIN_WIDTH_sec)
    relative_timestamps = np.linspace(
        -PRE_GO_CUE_sec, POST_GO_CUE_sec, num_bins_in_window
    )

    for electrode_idx in SELECTED_ELECTRODES:
        fig, ax = plt.subplots()

        for direction_idx in range(8):
            trial_averaged = trial_averaged_by_direction[direction_idx][
                :, electrode_idx
            ]
            sem = sem_by_direction[direction_idx][:, electrode_idx]
            color = TARGET_COLORS[direction_idx]
            ax.plot(relative_timestamps, trial_averaged, color=color, linewidth=2)
            ax.fill_between(
                relative_timestamps,
                trial_averaged - sem,
                trial_averaged + sem,
                color=color,
                alpha=0.1,
                edgecolor="none",
            )

        # Add a dot for the go cue.
        ax.scatter(
            [0.0], [-4.0], marker="o", s=95, color=(0.2, 0.2, 0.2), clip_on=False
        )
        ax.text(0.0, -9.0, "go cue", ha="center", va="top", fontsize=18)

        # Add a scale bar for time.
        ax.hlines(
            -4.0,
            POST_GO_CUE_sec - 0.5,
            POST_GO_CUE_sec,
            color=(0.2, 0.2, 0.2),
            linewidth=3,
            clip_on=False,
        )
        ax.text(
            POST_GO_CUE_sec,
            -7.0,
            f"{int(0.5 * 1000)} ms",
            ha="right",
            va="top",
            fontsize=16,
        )

        # Style the plot.
        ax.set_xlim(-PRE_GO_CUE_sec, POST_GO_CUE_sec)
        ax.tick_params(bottom=False, labelbottom=False)
        ax.set_ylim(0, 100)
        ax.set_yticks([0, 100])
        ax.tick_params(axis="y", width=3, length=8, labelsize=20)
        ax.set_ylabel("firing rate (Hz)", fontsize=24, labelpad=10)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["bottom"].set_visible(False)
        ax.spines["left"].set_position(("data", -PRE_GO_CUE_sec - 0.1))
        ax.spines["left"].set_linewidth(3)

        array_label = data[0]["array_label_by_electrode"][electrode_idx].strip()
        fig.suptitle(f"electrode {electrode_idx}\n(array {array_label})", fontsize=20)

        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    main()

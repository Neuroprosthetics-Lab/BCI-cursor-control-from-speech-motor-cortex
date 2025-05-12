import numpy as np
import scipy.io
import matplotlib.pyplot as plt


########################################################################################
#
# Main function.
#
########################################################################################


def main():
    """"""

    ## Load the Grid Evaluation Task blocks of data from the last Evaluation Session.
    ## This session used the improved decoder and denser grid.

    filepaths = [
        "./dryad_files/t15_day00468_block03_grid_evaluation_task.mat",
        "./dryad_files/t15_day00468_block04_grid_evaluation_task.mat",
        "./dryad_files/t15_day00468_block05_grid_evaluation_task.mat",
        "./dryad_files/t15_day00468_block09_grid_evaluation_task.mat",
        "./dryad_files/t15_day00468_block10_grid_evaluation_task.mat",
        "./dryad_files/t15_day00468_block11_grid_evaluation_task.mat",
        "./dryad_files/t15_day00468_block15_grid_evaluation_task.mat",
        "./dryad_files/t15_day00468_block16_grid_evaluation_task.mat",
        "./dryad_files/t15_day00468_block17_grid_evaluation_task.mat",
    ]
    try:
        data = [scipy.io.loadmat(filepath) for filepath in filepaths]
    except FileNotFoundError:
        print(
            "ERROR: Data files not found. Follow steps in the README to download data."
        )

    ## Plot a timeline of the evaluation blocks.

    fig, axs = plt.subplots(1, len(data))

    # Keep track of bitrates for plotting later.
    bitrates = []

    for ax_idx, (block_ax, block_data) in enumerate(zip(axs, data)):
        timestamps = block_data["timestamp_sec"].flatten()
        cursor_positions = block_data["cursor_position"]
        target_positions = block_data["target_position"]
        trial_start_bins = block_data["trial_start_bin"].flatten()
        grid_num_rows = block_data["grid_num_rows"].item()
        grid_total_height = block_data["grid_total_height"].item()

        # Get which bins the cursor was on the cued target.
        row_height = column_width = grid_total_height / grid_num_rows
        target_distances = np.abs(target_positions - cursor_positions)
        is_within_x = target_distances[:, 0] < (column_width / 2)
        is_within_y = target_distances[:, 1] < (row_height / 2)
        is_on_cued_target = is_within_x & is_within_y

        # Get which trial-ending clicks were on the cued target and which were not.
        trial_results = []
        trial_ending_click_bins = trial_start_bins - 1
        trial_ending_click_bins = trial_ending_click_bins[1:]
        trial_ending_click_timestamps = np.array(
            [timestamps[bin_idx] for bin_idx in trial_ending_click_bins]
        )
        for trial_ending_click_bin in trial_ending_click_bins:
            is_success = is_on_cued_target[trial_ending_click_bin]
            trial_results.append(is_success)
        trial_results = np.array(trial_results)

        # Get trial lengths.
        trial_start_timestamps = np.array(
            [timestamps[bin_idx] for bin_idx in trial_start_bins]
        )
        trial_lengths = np.diff(trial_start_timestamps)

        # Calculate bitrate, for plotting later.
        total_length = timestamps[-1] - timestamps[0]
        num_success = sum(trial_results)
        num_fail = len(trial_results) - num_success
        net_target_selections = num_success - num_fail
        total_target_options = 14 * 14
        bits_per_selection = np.log2(total_target_options - 1)
        bitrate = (net_target_selections * bits_per_selection) / total_length
        bitrates.append(bitrate)

        ## Plot this block's trial results on the corresponding subplot.

        # Success points.
        block_ax.scatter(
            trial_ending_click_timestamps[trial_results],
            trial_lengths[trial_results],
            marker="o",
            color=(0.0, 0.52, 0.60),
            s=20,
            clip_on=False,
            label="success",
        )
        # Failure points.
        block_ax.scatter(
            trial_ending_click_timestamps[~trial_results],
            trial_lengths[~trial_results],
            marker="^",
            color=(0.95, 0.33, 0.0),
            s=26,
            clip_on=False,
            label="failure",
        )
        # Style the plot.
        block_ax.set_xlim(0, 180)
        block_ax.set_xticks([0, 180])
        block_ax.tick_params(axis="x", labelsize=11)
        block_ax.xaxis.get_majorticklabels()[0].set_horizontalalignment("left")
        block_ax.xaxis.get_majorticklabels()[-1].set_horizontalalignment("right")
        block_ax.set_ylim(0, 10)
        block_ax.set_yticks([0, 2, 4, 6, 8, 10])
        block_ax.set_yticklabels([0, "", "", "", "", 10])
        block_ax.spines["top"].set_visible(False)
        block_ax.spines["right"].set_visible(False)
        block_ax.spines["bottom"].set_linewidth(2)
        block_ax.spines["left"].set_linewidth(2)
        block_ax.tick_params(width=2, labelsize=14)
        block_ax.spines["bottom"].set_position(("data", -0.5))
        block_ax.spines["left"].set_position(("data", -35))
        block_ax.set_facecolor((0.93, 0.93, 0.93))
        # Add a legend to only the first subplot.
        if ax_idx == 0:
            block_ax.set_zorder(2)
            block_ax.legend(loc=(0.14, 0.80), prop={"size": 12}, markerscale=2.0)
        # Remove certain parts for all subplots except the first.
        if ax_idx != 0:
            block_ax.set_xticks([])
            block_ax.spines["bottom"].set_visible(False)
            block_ax.set_yticks([])
            block_ax.spines["left"].set_visible(False)

    fig.supxlabel("time (s)", x=0.56, ha="center", fontsize=16)
    fig.supylabel(
        "target acquisition\ntime (s)",
        fontsize=16,
        x=0.05,
        va="center",
        ha="center",
    )
    fig.subplots_adjust(wspace=0.18, top=0.85, bottom=0.15, right=0.98, left=0.14)
    fig.set_figwidth(8)

    plt.show()

    ## Plot the bitrates during the evaluation blocks.

    fig, ax = plt.subplots()

    bitrate_avg = np.mean(bitrates)

    ax.scatter(
        range(len(data)),
        bitrates,
        marker="o",
        color=(0.5, 0.3, 0.7),
        s=70,
        edgecolors="none",
    )
    ax.scatter(
        [11],
        [bitrate_avg],
        marker="<",
        color=(0.5, 0.3, 0.7),
        s=80,
        clip_on=False,
    )
    ax.text(
        13,
        bitrate_avg,
        f"T15 ({bitrate_avg:.2f})",
        color=(0.5, 0.3, 0.7),
        fontsize=14,
        fontweight="bold",
        verticalalignment="center",
    )
    # Style the plot.
    ax.set_xlim(-3, 10)
    ax.set_xticks([])
    ax.set_ylim(0, 5)
    ax.set_yticks([0, 1, 2, 3, 4, 5])
    ax.tick_params(axis="y", labelsize=16)
    ax.set_ylabel("bitrate (bps)", fontsize=18, labelpad=15)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.tick_params(width=2)
    fig.subplots_adjust(right=0.55, left=0.25)
    fig.set_figwidth(4)

    plt.show()


if __name__ == "__main__":
    main()

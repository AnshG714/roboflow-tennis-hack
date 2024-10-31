import json
import matplotlib
import matplotlib.pyplot as plt
from scipy.stats import gaussian_kde
import numpy as np
from pathlib import Path
import io


# Output from running run_inference_on_reference_image in inference.py
REFERENCE_DEPTH_HEIGHT = 402
REFERENCE_BASELINE = REFERENCE_DEPTH_HEIGHT / 261


def read_results(filename="results.json"):
    with open(filename) as f:
        results = json.load(f)
    return results


def clamp_outliers(depths, factor=3):

    mean = np.mean(depths)
    std_dev = np.std(depths)
    lower_bound = mean - factor * std_dev
    upper_bound = mean + factor * std_dev

    # Clamp depths to within bounds
    clamped_depths = np.clip(depths, lower_bound, upper_bound)

    return clamped_depths


def get_depth_estimate(bounding_box_height):
    depth = REFERENCE_DEPTH_HEIGHT / bounding_box_height

    return depth


def get_player_bbox_centers(results):

    positions = []
    hits = []
    for result in results["tennis-annotations"]:
        predictions = result["predictions"]
        if len(predictions) > 0:
            prediction = predictions[0]
            class_name = prediction["class"]

            x, y = (
                prediction["x"],
                prediction["y"],
            )

            relative_depth = get_depth_estimate(prediction["height"])

            if class_name in [
                "forehand-ready",
                "forehand-stroke",
                "forehand-finish",
                "backhand-ready",
                "backhand-stroke",
                "backhand-finish",
            ]:
                hits.append((x, y, relative_depth))

            positions.append((x, y, relative_depth))

    return positions, hits


def plot_heatmap(positions, label="Position"):
    positions = np.array(positions)
    x = positions[:, 0]
    depth = positions[:, 2]

    # make sure we're not recording outliers -- seems like sometimes we
    # detect people on the other side of the net
    depth = clamp_outliers(depth)

    k = gaussian_kde(np.vstack([x, depth]))

    xi, yi = np.mgrid[
        x.min() : x.max() : 50 * 1j,
        depth.min() : depth.max() : 50 * 1j,
    ]
    zi = k(np.vstack([xi.flatten(), yi.flatten()]))
    fig = plt.figure(figsize=(8.54, 4.8))
    ax = fig.add_subplot(111)
    ax.contourf(xi, yi, zi.reshape(xi.shape))
    ax.axhline(
        REFERENCE_BASELINE, color="white", linestyle="--", linewidth=2, label="Baseline"
    )
    ax.set_xlabel("Court X Position (pixels)")
    ax.set_ylabel("Court Depth (Relative))")

    plt.xlim(x.min() - 30, x.max() + 30)
    plt.ylim(depth.min(), depth.max())

    plt.title("Player Position Heatmap")

    return plt


def process_results(filename):
    matplotlib.use("Agg")
    filename_without_extension = Path(filename).stem
    results = {}
    with open(f"./outputs/{filename_without_extension}.json") as f:
        results = json.load(f)

    if results is None:
        return None

    centers, _ = get_player_bbox_centers(results)
    plot = plot_heatmap(centers)

    bytes_image = io.BytesIO()
    plot.savefig(bytes_image, format="png")
    bytes_image.seek(0)
    return bytes_image


if __name__ == "__main__":
    # for testing.
    results = read_results()
    centers, hits = get_player_bbox_centers(results)
    plot_heatmap(centers)
    plt.show()

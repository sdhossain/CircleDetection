import os
from skimage import io
import matplotlib.pyplot as plt


def plot_and_save_circles(df, directory, radius_ranges):
    if not os.path.exists(directory):
        os.makedirs(directory)

    for min_radius, max_radius in radius_ranges:
        range_df = df[
            (df['radius'] >= min_radius) & (df['radius'] < max_radius)]
        examples = range_df.sample(n=min(3, len(range_df)))  # No. of Samples

        for index, row in examples.iterrows():
            img = io.imread(row['filepath'])
            plt.figure()
            ax = plt.gca()

            draw_circle(img=img, 
                        row=row['row'],
                        col=row['col'],
                        radius=row['radius'],
                        color='green',
                        ax=ax)  # Ground truth in green
            draw_circle(img=img,
                        row=row['predicted_row'],
                        row=row['predicted_col'],
                        row=row['predicted_radius'],
                        color='red',
                        ax=ax)  # Prediction in red

            plt.legend(['Ground Truth', 'Prediction'])
            plt.title(
                f"Radius Range {min_radius}-{max_radius}, Size {row['radius']}"
            )
            plt.savefig(
                os.path.join(
                    directory,
                    f"radius_range_{min_radius}_{max_radius}_size" + \
                    f"_{row['radius']}_index_{index}.png")
            )
            plt.close()


def draw_circle(img, row, col, radius, color='red', ax=None):
    """
    Function to draw a circle on the image.
    """
    if ax is None:
        ax = plt.gca()
    circle = plt.Circle((col, row), radius, color=color, fill=False)
    ax.add_artist(circle)
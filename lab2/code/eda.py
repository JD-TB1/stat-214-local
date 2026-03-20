from utils_npz import NPZUtils
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from pathlib import Path

DOCS_PATH = Path(__file__).parent.parent / "documents"
DOCS_PATH.mkdir(exist_ok=True)


def visualize_radiance_angles(utils: NPZUtils, files_to_visualize: list[str], angles: list[int]) -> None:
    """
    Visualize images for all radiance angles for each file.

    Args:
        utils: An instance of NPZUtils to load and visualize data.
        files_to_visualize: List of .npz filenames to visualize.
        angles: List of radiance angles to visualize.
    """
    dict_radiance_angles = {k: v for k, v in utils.dict_radiance_angles.items() if k in angles}

    for filename in files_to_visualize:
        df = utils.load_img_to_df(filename)

        fig, axes = plt.subplots(
            1,
            len(dict_radiance_angles),
            figsize=(5 * len(dict_radiance_angles), 4),
            constrained_layout=True
        )

        if len(dict_radiance_angles) == 1:
            axes = [axes]

        for idx, (radiance_angle, angle_name) in enumerate(dict_radiance_angles.items()):
            ax = axes[idx]
            img = utils.prepare_image_from_df(df, radiance_angle)

            ax.imshow(img, cmap='gray')
            ax.set_title(angle_name, fontsize=10)
            ax.axis('off')

        fig.suptitle(f'Radiance Angles for {filename}', fontsize=16)
        plt.savefig(DOCS_PATH / f'radiance_angles_{filename}.png')
        plt.show()

def visualise_comparisons(utils: NPZUtils, files_to_visualize, feature1 ,feature2):
    """
    Visualize comparative features for each file

    Args:
        utils: An instance of NPZUtils to load and visualize data.
        files_to_visualize: List of .npz filenames to visualize.
    """
    fig, axes = plt.subplots(1,3,figsize=(18,6))
    for ax, img in zip(axes,files_to_visualize):
        df = utils.load_img_to_df(img)
        sns.scatterplot(
            data=df[df.label != 0],
            x=feature1,
            y=feature2,
            hue="label",
            alpha=0.5,
            ax = ax
        )
        ax.set_title(img)
        ax.set_xlabel(utils.column_map[feature1])
        ax.set_ylabel(utils.column_map[feature2])
        ax.legend()

    plt.tight_layout()
    plt.savefig(DOCS_PATH / f'{feature1}_vs_{feature2}.png')
    plt.show()


if __name__ == "__main__":
    npz_utils = NPZUtils()
    files_to_visualize = ['O013257.npz', 'O013490.npz', 'O012791.npz']
    visualize_radiance_angles(npz_utils, files_to_visualize, angles=[5, 6, 7, 8, 9])
    #print(npz_utils.load_npz(files_to_visualize[0])['arr_0'].shape)
    visualise_comparisons(npz_utils,files_to_visualize,"corr","ndai")
    visualise_comparisons(npz_utils,files_to_visualize,"corr","sd")
    visualise_comparisons(npz_utils,files_to_visualize,"ra_df","ra_an")
    visualise_comparisons(npz_utils,files_to_visualize,"ra_bf","ra_an")
    visualise_comparisons(npz_utils,files_to_visualize,"ra_cf","ra_an")

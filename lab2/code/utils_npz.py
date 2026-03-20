import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import glob

class NPZUtils:
    def __init__(self, data_path=None):
        """
        Initialize the utility class with the path to the directory containing .npz files.
        """
        self.data_path = data_path or Path(__file__).parent.parent / "data"
        self.dict_radiance_angles = {
            5: "DF",
            6: "CF",
            7: "BF",
            8: "AF",
            9: "AN",
            10: "Expert Labels"
        }
        self.columns = [
            "y", "x", "ndai", "sd","corr",
            "ra_df", "ra_cf","ra_bf","ra_af","ra_an","label"
        ]
        self.column_map = {
            "y":"Y", "x": "X", 
            "ndai": "NDAI", "sd": "SD",
            "corr": "CORR", "ra_df": "Radiance angle DF", 
            "ra_cf": "Radiance angle CF","ra_bf": "Radiance angle BF",
            "ra_af": "Radiance angle AF","ra_an": "Radiance angle AN","label": "Expert Label"
        }

    def angle_to_column(self, radiance_angle: int) -> str:
        mapping = {
            5: "ra_df",
            6: "ra_cf",
            7: "ra_bf",
            8: "ra_af",
            9: "ra_an",
            10: "label",
        }
        return mapping[radiance_angle]


    def prepare_image_from_df(self, df: pd.DataFrame, radiance_angle: int) -> np.ndarray:
        """
        Prepare a 2D image from a dataframe for a specific radiance angle.
        """
        value_column = self.angle_to_column(radiance_angle)

        x_coords = df["x"].astype(int).to_numpy()
        y_coords = df["y"].astype(int).to_numpy()
        values = df[value_column].to_numpy()

        x_min, x_max = x_coords.min(), x_coords.max()
        y_min, y_max = y_coords.min(), y_coords.max()

        width = x_max - x_min + 1
        height = y_max - y_min + 1

        img = np.zeros((height, width), dtype=float)
        img[y_coords - y_min, x_coords - x_min] = values

        img_min = img.min()
        img_max = img.max()
        if img_max > img_min:
            img = (img - img_min) / (img_max - img_min)

        return img
        
    def load_img_to_df(self, img):
        data = np.load("../data/"+img)
        img = data[data.files[0]]
        df = pd.DataFrame(img, columns = self.columns)
        return df
    
    def load_all(self):
        all_data = []
        for fp in sorted(glob.glob("../data/*.npz")):
            z = np.load(fp)
            arr = z[z.files[0]]
            # If label column missing
            if arr.shape[1] == 10:
                label_col = np.zeros((arr.shape[0], 1))
                arr = np.hstack([arr, label_col])
            all_data.append(arr)
        all_data = np.vstack(all_data)
        df = pd.DataFrame(all_data, columns=self.columns)
        return df.shape
    

# test
# if __name__ == "__main__":
#     npz_utils = NPZUtils()
#     files_to_visualize = ['O013257.npz', 'O013490.npz', 'O012791.npz']
#     img = npz_utils.prepare_image(files_to_visualize[0], radiance_angle=10)
#     plt.imshow(img, cmap='gray', vmin=-1, vmax=1)
#     plt.savefig(f'documents/image_2D_{files_to_visualize[0]}_{npz_utils.dict_radiance_angles[10]}.png')
#     plt.show()


# check if we want to remove these
# since we are now using dataframes
# def load_npz(self, npz_filename: str) -> dict[str, np.ndarray]:
#         """
#         Load a specific .npz file from the data directory.

#         Args:
#             npz_filename: Name of the .npz file to load.

#         Returns:
#             dict: Data from the .npz file.
#         """
#         npz_path = self.data_path / npz_filename
#         npz_data = np.load(npz_path, allow_pickle=True)
#         return {key: npz_data[key] for key in npz_data.files}

# def load_all_npz(self) -> dict[str, np.ndarray]:
#     """
#     Load all .npz files from the data directory.

#     Returns:
#         dict: A dictionary with filenames as keys and their data as values.
#     """
#     data_dict = {}
#     for npz_file in self.data_path.glob("*.npz"):
#         npz_data = np.load(npz_file, allow_pickle=True)
#         data_dict[npz_file.name] = npz_data['arr_0']
#     return data_dict

# def prepare_image(self, filename: str, radiance_angle: int) -> np.ndarray:
#     """
#     Prepare a single image from a .npz file for a specific radiance angle.

#     Args:
#         filename: The .npz filename to prepare.
#         radiance_angle: The angle of radiance to prepare (5 to 9), or expert labels (10).

#     Returns:
#         np.ndarray: The prepared image.
#     """
#     print(f"Loading {filename}...")
#     data_dict = self.load_npz(filename)
#     data = data_dict['arr_0']

#     print(f"  Number of pixels: {data.shape[0]}")

#     # x, y coordinates
#     x_coords = data[:, 0].astype(int)
#     y_coords = data[:, 1].astype(int)

#     # Image dimensions
#     x_min, y_min = x_coords.min(), y_coords.min()
#     print(f"  Coordinate range: x [{x_min}, {x_coords.max()}], y [{y_min}, {y_coords.max()}]")
#     height = int(y_coords.max() - y_min + 1)
#     width = int(x_coords.max() - x_min + 1)
#     print(f"  Dimensions: {height} x {width}")

#     # Reconstruct the 2D image using the specified radiance angle
#     img = np.zeros((height, width))
#     for i in range(len(x_coords)):
#         img[y_coords[i] - y_min, x_coords[i] - x_min] = data[i, radiance_angle]

#     # Normalize for display
#     img_min = img.min()
#     img_max = img.max()
#     if img_max > img_min:
#         img_norm = (img - img_min) / (img_max - img_min)
#     else:
#         img_norm = img

#     return img_norm
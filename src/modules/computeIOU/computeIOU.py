from logs import logDecorator as lD
from lib.unet import unetVgg16
import jsonref, os
import numpy as np
from pathlib import Path
import cv2
import pandas as pd

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.computeIOU.computeIOU"

config_iou = jsonref.load(open("../config/modules/computeIOU.json"))


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for computeIOU.

    This function computes IOU for a given ground truth mask and its
    corresponding predicted mask.

    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dintionary containing information about the
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    """

    print("=" * 30)
    print("Main function of computeIOU.")
    print("=" * 30)

    # Get parameters from .json files.
    y_true_dir = config_iou["y_true_dir"]
    y_pred_dir = config_iou["y_pred_dir"]
    extension = config_iou["extension"]
    target_size = (config_iou["target_size"], config_iou["target_size"])
    save_iou_dict_dir = config_iou["save_iou_dict_dir"]

    # ------------

    # Get paths.
    y_true_paths_list = []
    y_pred_paths_list = []

    for full in os.listdir(y_true_dir):
        if full.endswith(extension):
            y_true_paths_list.append(os.path.join(y_true_dir, full))

    for full in os.listdir(y_pred_dir):
        if full.endswith(extension):
            y_pred_paths_list.append(os.path.join(y_pred_dir, full))

    y_true_paths_list.sort()
    y_pred_paths_list.sort()

    # ------------

    # Load y_true masks.
    y_true_arrays = [
        unetVgg16.unetVgg16().loadMaskImg(path=path, dsize=target_size)
        for path in y_true_paths_list
    ]

    y_true_arrays = np.array(y_true_arrays, dtype=np.float64)

    # Load y_pred masks.
    y_pred_arrays = [
        unetVgg16.unetVgg16().loadMaskImg(path=path, dsize=target_size)
        for path in y_pred_paths_list
    ]

    y_pred_arrays = np.array(y_pred_arrays, dtype=np.float64)

    # ------------

    iou_dict = {}

    # Iterate and compute IOU.
    for i in range(len(y_true_paths_list)):

        # Compute IOU and save to dict
        iou_tensor = unetVgg16.unetVgg16().iouMetric(
            y_true=y_true_arrays[i], y_pred=y_pred_arrays[i]
        )

        iou = iou_tensor.numpy()

        # Get patient ID from y_true masks.
        filename = os.path.basename(y_true_paths_list[i])
        filename_split = filename.split("_")
        patientID = "_".join([filename_split[i] for i in range(4)])

        # Save to dict.
        iou_dict[patientID] = iou

    # ------------

    # Save to DataFrame and csv.
    df = pd.DataFrame(list(iou_dict.items()), columns=["patient_id", "iouMetric"])

    save_path = os.path.join(save_iou_dict_dir, "iou.csv")
    df.to_csv(save_path, index=False)
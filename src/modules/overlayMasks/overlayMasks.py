from logs import logDecorator as lD
from lib.unet import unetVgg16
import jsonref, os
import numpy as np
from pathlib import Path
import cv2
import pandas as pd

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.overlayMasks.overlayMasks"

config_overlay = jsonref.load(open("../config/modules/overlayMasks.json"))


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for overlayMasks.

    This function overlays a given ground truth mask with its
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
    print("Main function of overlayMasks.")
    print("=" * 30)

    # Get parameters from .json files.
    full_img_dir = config_overlay["full_img_dir"]
    y_true_dir = config_overlay["y_true_dir"]
    y_pred_dir = config_overlay["y_pred_dir"]
    extension = config_overlay["extension"]
    target_size = (config_overlay["target_size"], config_overlay["target_size"])
    save_maskoverlay_dir = config_overlay["save_maskoverlay_dir"]
    save_fulloverlay_dir = config_overlay["save_fulloverlay_dir"]

    # ------------

    # Get paths.
    full_img_paths_list = []
    y_true_paths_list = []
    y_pred_paths_list = []

    for full in os.listdir(full_img_dir):
        if full.endswith(extension):
            full_img_paths_list.append(os.path.join(full_img_dir, full))

    for full in os.listdir(y_true_dir):
        if full.endswith(extension):
            y_true_paths_list.append(os.path.join(y_true_dir, full))

    for full in os.listdir(y_pred_dir):
        if full.endswith(extension):
            y_pred_paths_list.append(os.path.join(y_pred_dir, full))

    full_img_paths_list.sort()
    y_true_paths_list.sort()
    y_pred_paths_list.sort()

    # ------------

    # Load full_img.
    full_img_arrays = [
        cv2.resize(src=cv2.imread(path, cv2.IMREAD_GRAYSCALE), dsize=target_size)
        for path in full_img_paths_list
    ]

    # Load y_true masks.
    y_true_arrays = [
        cv2.resize(src=cv2.imread(path, cv2.IMREAD_GRAYSCALE), dsize=target_size)
        for path in y_true_paths_list
    ]

    # Load y_pred masks.
    y_pred_arrays = [
        cv2.resize(src=cv2.imread(path, cv2.IMREAD_GRAYSCALE), dsize=target_size)
        for path in y_pred_paths_list
    ]

    print(full_img_arrays[0].min(), full_img_arrays[0].max())
    print(y_true_arrays[0].min(), y_true_arrays[0].max())
    print(y_pred_arrays[0].min(), y_pred_arrays[0].max())

    # ------------

    # Stack to create RGB version of grayscale images.
    full_img_rgb = [np.stack([img, img, img], axis=-1) for img in full_img_arrays]

    # Green true mask. Note OpenCV uses BGR.
    y_true_rgb = [
        np.stack([np.zeros_like(img), img, np.zeros_like(img)], axis=-1)
        for img in y_true_arrays
    ]

    # Red predicted mask. Note OpenCV uses BGR.
    y_pred_rgb = [
        np.stack([np.zeros_like(img), np.zeros_like(img), img], axis=-1)
        for img in y_pred_arrays
    ]

    # ------------

    for i in range(len(full_img_rgb)):

        # First overlay true and predicted masks.
        overlay_masks = cv2.addWeighted(
            src1=y_true_rgb[i], alpha=0.5, src2=y_pred_rgb[i], beta=1, gamma=0
        )

        # Then overlay full_img and masks.
        overlay_all = cv2.addWeighted(
            src1=full_img_rgb[i], alpha=1, src2=overlay_masks, beta=0.5, gamma=0
        )

        # Save.

        # Get patient ID from y_true masks.
        filename = os.path.basename(y_true_paths_list[i])
        filename_split = filename.split("_")
        patientID = "_".join([filename_split[i] for i in range(4)])

        masks_filename = patientID + "___MasksOverlay.png"
        all_filename = patientID + "___AllOverlay.png"

        save_path_masks = os.path.join(save_maskoverlay_dir, masks_filename)
        save_path_all = os.path.join(save_fulloverlay_dir, all_filename)

        print(save_path_masks)
        print(save_path_all)

        cv2.imwrite(filename=save_path_masks, img=overlay_masks)
        cv2.imwrite(filename=save_path_all, img=overlay_all)

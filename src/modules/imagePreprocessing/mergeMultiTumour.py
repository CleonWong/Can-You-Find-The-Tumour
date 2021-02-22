import os
import pprint
import jsonref
import numpy as np
import pandas as pd
import cv2
from logs import logDecorator as lD

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.imagePreprocessing.mergeMultiTumour"
config_mmt = jsonref.load(open("../config/modules/mergeMultiTumour.json"))


@lD.log(logBase + ".findMultiTumour")
def findMultiTumour(logger, csv_path, abnormality_col):

    """
    This function returns a set of patientID_leftOrRight_imageView
    that have more than 1 abnormality.

    Parameters
    ----------
    csv_path : {str}
        The relative (or absolute) path to the
        Mass-Training-Description-UPDATED.csv or
        Mass-Test-Description-UPDATED.csv
    abnormality_col: {str}
        The name of the column that counts the number of
        abnormalities.

    Returns
    -------
    multi_tumour_set: {set}
        A set of all the patient IDs that have more than one
        abnormality (a.k.a tumour).
    """

    try:
        # Read .csv
        df = pd.read_csv(csv_path, header=0)

        # Get rows with more than 1 abnormality.
        multi_df = df.loc[df[abnormality_col] > 1]

        multi_tumour_list = []
        for row in multi_df.itertuples():

            # Get patient ID, image view and description.
            patient_id = row.patient_id
            lr = row.left_or_right_breast
            img_view = row.image_view

            # Join to get filename identifier.
            identifier = "_".join([patient_id, lr, img_view])
            multi_tumour_list.append(identifier)

        # Get unique set.
        multi_tumour_set = set(multi_tumour_list)

    except Exception as e:
        # logger.error(f'Unable to findMultiTumour!\n{e}')
        print((f"Unable to findMultiTumour!\n{e}"))

    return multi_tumour_set


@lD.log(logBase + ".masksToSum")
def masksToSum(logger, img_path, multi_tumour_set, extension):

    """
    This function gets the relative (or absolute, depending
    on `img_path`) path of the masks that needs to be summed.

    Parameters
    ----------
    img_path : {str}
        The relative (or absolute) path that contains all the
        images and its masks.
    multi_tumour_set: {set}
        The set that contains all the patient id with that
        needs their masks summed.
    extension : {str}
        The filetype of the mask image. e.g. ".png", ".jpg".

    Returns
    -------
    masks_to_sum_dict: {dict}
        A dict where (key, value) = (patient id, paths of the
        masks to sum).
    """

    try:

        # Get filenames of all images in `img_path`.
        images = [
            f
            for f in os.listdir(img_path)
            if (not f.startswith(".") and f.endswith(extension))
        ]

        # Get filenames of all maskes that needs to be summed.
        masks_to_sum = [
            m
            for m in images
            if ("MASK" in m and any(multi in m for multi in multi_tumour_set))
        ]

        # Create dict.
        masks_to_sum_dict = {patient_id: [] for patient_id in multi_tumour_set}

        for k, _ in masks_to_sum_dict.items():
            v = [os.path.join(img_path, m) for m in masks_to_sum if k in m]
            masks_to_sum_dict[k] = sorted(v)

        # Remove items that have only one mask to smum (i.e. don't need to sum)
        to_pop = [k for k, v in masks_to_sum_dict.items() if len(v) == 1]

        for k in to_pop:
            masks_to_sum_dict.pop(k)

    except Exception as e:
        # logger.error(f'Unable to findMultiTumour!\n{e}')
        print((f"Unable to get findMultiTumour!\n{e}"))

    return masks_to_sum_dict


@lD.log(logBase + ".sumMasks")
def sumMasks(logger, mask_list):

    """
    This function sums a list of given masks.

    Parameters
    ----------
    mask_list : {list of numpy.ndarray}
        A list of masks (numpy.ndarray) that needs to be summed.

    Returns
    -------
    summed_mask_bw: {numpy.ndarray}
        The summed mask, ranging from [0, 1].
    """

    try:

        summed_mask = np.zeros(mask_list[0].shape)

        for arr in mask_list:
            summed_mask = np.add(summed_mask, arr)

        # Binarise (there might be some overlap, resulting in pixels with
        # values of 510, 765, etc...)
        _, summed_mask_bw = cv2.threshold(
            src=summed_mask, thresh=1, maxval=255, type=cv2.THRESH_BINARY
        )

    except Exception as e:
        # logger.error(f'Unable to findMultiTumour!\n{e}')
        print((f"Unable to get findMultiTumour!\n{e}"))

    return summed_mask_bw


# ----------------------------------


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for imagePreprocessing module.

    This function takes a path of the raw image folder,
    iterates through each image and executes the necessary
    image preprocessing steps on each image, and saves
    preprocessed images in the output path specified.

    The hyperparameters in this module can be tuned in
    the "../config/modules/mergeMultiTumour.json" file.

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
    print("Main function of mergeMultiTumour.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Get params.
    img_path = config_mmt["paths"]["images"]
    csv_path = config_mmt["paths"]["csv"]
    abnormality_col = config_mmt["abnormality_col"]
    extension = config_mmt["extension"]
    output_path = config_mmt["paths"]["output"]
    save = config_mmt["save"]

    multi_tumour_set = findMultiTumour(
        csv_path=csv_path, abnormality_col=abnormality_col
    )

    masks_to_sum_dict = masksToSum(
        img_path=img_path, multi_tumour_set=multi_tumour_set, extension=extension
    )

    # Sum!
    for k, v in masks_to_sum_dict.items():

        # Get image as arrays.
        mask_list = [cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE) for mask_path in v]

        # Sum masks
        summed_mask = sumMasks(mask_list=mask_list)

        # Save summed mask
        if save:
            if "train" in img_path.lower():
                save_path = os.path.join(
                    output_path, ("_".join(["Mass-Training", k, "MASK___PRE.png"]))
                )
            elif "test" in img_path.lower():
                save_path = os.path.join(
                    output_path, ("_".join(["Mass-Test", k, "MASK___PRE.png"]))
                )
            cv2.imwrite(save_path, summed_mask)

    print()
    print("Getting out of mergeMultiTumour module.")
    print("-" * 30)
    print()

    return
import os
import pprint
import jsonref
import numpy as np
import cv2
import pydicom
from pathlib import Path
from logs import logDecorator as lD

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.cropPreprocessing.cropPreprocessing"
config_cropPre = jsonref.load(open("../config/modules/cropPreprocessing.json"))


@lD.log(logBase + ".minMaxNormalise")
def minMaxNormalise(logger, img):

    """
    This function does min-max normalisation on
    the given image.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to normalise.

    Returns
    -------
    norm_img: {numpy.ndarray}
        The min-max normalised image.
    """

    try:
        norm_img = (img - img.min()) / (img.max() - img.min())

    except Exception as e:
        # logger.error(f'Unable to minMaxNormalise!\n{e}')
        print((f"Unable to get minMaxNormalise!\n{e}"))

    return norm_img


@lD.log(logBase + ".clahe")
def clahe(logger, img, clip=2.0, tile=(8, 8)):

    """
    This function applies the Contrast-Limited Adaptive
    Histogram Equalisation filter to a given image.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to edit.
    clip : {int or floa}
        Threshold for contrast limiting.
    tile : {tuple (int, int)}
        Size of grid for histogram equalization. Input
        image will be divided into equally sized
        rectangular tiles. `tile` defines the number of
        tiles in row and column.

    Returns
    -------
    clahe_img : {numpy.ndarray, np.uint8}
        The CLAHE edited image, with values ranging from [0, 255]
    """

    try:
        # Convert to uint8.
        # img = skimage.img_as_ubyte(img)
        img = cv2.normalize(
            img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        img_uint8 = img.astype("uint8")
        #  img = cv2.normalize(
        #     img, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8U
        # )

        clahe_create = cv2.createCLAHE(clipLimit=clip, tileGridSize=tile)
        clahe_img = clahe_create.apply(img_uint8)

    except Exception as e:
        # logger.error(f'Unable to clahe!\n{e}')
        print((f"Unable to get clahe!\n{e}"))

    return clahe_img


@lD.log(logBase + ".pad")
def pad(logger, img):

    """
    This function pads a given image with black pixels,
    along its shorter side, into a square and returns
    the square image.

    If the image is portrait, black pixels will be
    padded on the right to form a square.

    If the image is landscape, black pixels will be
    padded on the bottom to form a square.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to pad.

    Returns
    -------
    padded_img : {numpy.ndarray}
        The padded square image, if padding was required
        and done.
    img : {numpy.ndarray}
        The original image, if no padding was required.
    """

    try:
        nrows, ncols = img.shape

        # If padding is required...
        if nrows != ncols:

            # Take the longer side as the target shape.
            if ncols < nrows:
                target_shape = (nrows, nrows)
            elif nrows < ncols:
                target_shape = (ncols, ncols)

            # pad.
            padded_img = np.zeros(shape=target_shape)
            padded_img[:nrows, :ncols] = img

        # If padding is not required...
        elif nrows == ncols:

            # Return original image.
            padded_img = img

    except Exception as e:
        # logger.error(f'Unable to pad!\n{e}')
        print((f"Unable to pad!\n{e}"))

    return padded_img


@lD.log(logBase + ".centerCrop")
def centerCrop(logger, img):

    """
    This function takes a center square crop of a given image, with the length
    of the square equal to the shorter length of the original image.

    e.g. The original image (height, width) = (x, y), where x < y.
    Then the square crop will be of sides with length (x, x).

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to crop.

    Returns
    -------
    sq_img : {numpy.ndarray}
        The cropped square image.
    img : {numpy.ndarray}
        The original image, if no padding was required.
    """

    try:
        h, w = img.shape

        # If cropping is required...
        if h != w:

            # Take the shorter side as the square length.
            if w < h:  # Vertical rectangle, use w as square length.
                start_w = 0
                end_w = w
                start_h = h // 2 - w // 2
                end_h = start_h + w

            elif h < w:  # Horizontal rectangle, use h as square length.
                start_h = 0
                end_h = h
                start_w = w // 2 - h // 2
                end_w = start_w + h

            # Crop.
            sq_img = img[start_h:end_h, start_w:end_w]

            return sq_img

        # If padding is not required...
        elif w == h:

            # Return original image.
            return img

    except Exception as e:
        # logger.error(f'Unable to centerCrop!\n{e}')
        print((f"Unable to centerCrop!\n{e}"))


@lD.log(logBase + ".cropPreprocess")
def cropPreprocess(
    logger,
    img,
    clip,
    tile,
):

    """
    This function chains and executes all the preprocessing
    steps for a cropped ROI image, in the following order:

    Step 1 - Min-max normalise
    Step 2 - CLAHE enchancement
    Step 3 - Center square crop
    Step 4 - Min-max normalise

    Parameters
    ----------
    img : {numpy.ndarray}
        The cropped ROI image to preprocess.

    Returns
    -------
    img_pre : {numpy.ndarray}
        The preprocessed cropped ROI image.
    """

    try:

        # Step 1: Min-max normalise.
        norm_img = minMaxNormalise(img=img)
        # cv2.imwrite("../data/preprocessed/Mass/testing/normed.png", norm_img)

        # Step 2: CLAHE enhancement.
        clahe_img = clahe(img=norm_img, clip=clip, tile=(tile, tile))
        # cv2.imwrite("../data/preprocessed/Mass/testing/clahe_img.png", clahe_img)

        # Step 3: Crop.
        sq_img = centerCrop(img=clahe_img)

        # Step 4: Min-max normalise.
        img_pre = minMaxNormalise(img=sq_img)
        # cv2.imwrite("../data/preprocessed/Mass/testing/img_pre.png", img_pre)

        # padded_img = cv2.normalize(
        #     padded_img,
        #     None,
        #     alpha=0,
        #     beta=255,
        #     norm_type=cv2.NORM_MINMAX,
        #     dtype=cv2.CV_32F,
        # )
        # cv2.imwrite("../data/preprocessed/Mass/testing/padded_img.png", padded_img)

    except Exception as e:
        # logger.error(f'Unable to cropPreprocess!\n{e}')
        print((f"Unable to cropPreprocess!\n{e}"))

    return img_pre


# ----------------------------------


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for cropPreprocessing module.

    The hyperparameters in this module can be tuned in
    the "../config/modules/cropPreprocessing.json" file.

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
    print("Main function of cropPreprocessing.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Get path of the folder containing .dcm files.
    input_path = config_cropPre["paths"]["input"]
    output_path = config_cropPre["paths"]["output"]

    # Output format.
    output_format = config_cropPre["output_format"]

    # Get individual .dcm paths.
    dcm_paths = []
    for curdir, dirs, files in os.walk(input_path):
        files.sort()
        for f in files:
            if f.endswith(".dcm"):
                dcm_paths.append(os.path.join(curdir, f))

    # Get paths of full mammograms and ROI masks.
    crop_paths = [f for f in dcm_paths if ("CROP" in f)]

    count = 0
    for crop_path in crop_paths:

        # Read full mammogram .dcm file.
        ds = pydicom.dcmread(crop_path)

        # Get relevant metadata from .dcm file.
        patient_id = ds.PatientID

        # Calc-Test masks do not have the "Calc-Test_" suffix
        # when it was originally downloaded (masks from Calc-Train,
        # Mass-Test and Mass-Train all have their corresponding suffices).
        patient_id = patient_id.replace(".dcm", "")
        patient_id = patient_id.replace("Calc-Test_", "")

        print(patient_id)

        cropImg = ds.pixel_array

        # ===================
        # Preprocess Crop Img
        # ===================

        # Get all hyperparameters.
        clip = config_cropPre["clahe"]["clip"]
        tile = config_cropPre["clahe"]["tile"]

        # Preprocess cropped ROI image.
        cropImg_pre = cropPreprocess(img=cropImg, clip=clip, tile=tile)

        # Need to normalise to [0, 255] before saving as .png.
        cropImg_pre_norm = cv2.normalize(
            cropImg_pre,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # Save preprocessed crop image.
        save_filename = (
            os.path.basename(crop_path).replace(".dcm", "") + "___PRE" + output_format
        )
        save_path = os.path.join(output_path, save_filename)

        print(save_path)
        cv2.imwrite(save_path, cropImg_pre_norm)
        print(f"DONE FULL: {crop_path}")

        count += 1

        # if count == 10:
        #     break

    print(f"Total count = {count}")
    print()
    print("Getting out of cropPreprocessing module.")
    print("-" * 30)

    return
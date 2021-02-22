import os
import pprint
import jsonref
import numpy as np
import cv2
import pydicom
from pathlib import Path
from logs import logDecorator as lD


# import skimage

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = (
    config["logging"]["logBase"] + ".modules.imagePreprocessing.imagePreprocessing"
)
config_imgPre = jsonref.load(open("../config/modules/imagePreprocessing.json"))


@lD.log(logBase + ".cropBorders")
def cropBorders(logger, img, l=0.01, r=0.01, u=0.04, d=0.04):

    """
    This function crops a specified percentage of border from
    each side of the given image. Default is 1% from the top,
    left and right sides and 4% from the bottom side.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to crop.

    Returns
    -------
    cropped_img: {numpy.ndarray}
        The cropped image.
    """

    try:
        nrows, ncols = img.shape

        # Get the start and end rows and columns
        l_crop = int(ncols * l)
        r_crop = int(ncols * (1 - r))
        u_crop = int(nrows * u)
        d_crop = int(nrows * (1 - d))

        cropped_img = img[u_crop:d_crop, l_crop:r_crop]

    except Exception as e:
        # logger.error(f'Unable to cropBorders!\n{e}')
        print((f"Unable to get cropBorders!\n{e}"))

    return cropped_img


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


@lD.log(logBase + ".globalBinarise")
def globalBinarise(logger, img, thresh, maxval):

    """
    This function takes in a numpy array image and
    returns a corresponding mask that is a global
    binarisation on it based on a given threshold
    and maxval. Any elements in the array that is
    greater than or equals to the given threshold
    will be assigned maxval, else zero.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to perform binarisation on.
    thresh : {int or float}
        The global threshold for binarisation.
    maxval : {np.uint8}
        The value assigned to an element that is greater
        than or equals to `thresh`.


    Returns
    -------
    binarised_img : {numpy.ndarray, dtype=np.uint8}
        A binarised image of {0, 1}.
    """

    try:
        binarised_img = np.zeros(img.shape, np.uint8)
        binarised_img[img >= thresh] = maxval

    except Exception as e:
        # logger.error(f'Unable to globalBinarise!\n{e}')
        print((f"Unable to globalBinarise!\n{e}"))

    return binarised_img


@lD.log(logBase + ".editMask")
def editMask(logger, mask, ksize=(23, 23), operation="open"):

    """
    This function edits a given mask (binary image) by performing
    closing then opening morphological operations.

    Parameters
    ----------
    mask : {numpy.ndarray}
        The mask to edit.
    ksize : {tuple}
        Size of the structuring element.
    operation : {str}
        Either "open" or "close", each representing open and close
        morphological operations respectively.

    Returns
    -------
    edited_mask : {numpy.ndarray}
        The mask after performing close and open morphological
        operations.
    """

    try:
        kernel = cv2.getStructuringElement(shape=cv2.MORPH_RECT, ksize=ksize)

        if operation == "open":
            edited_mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        elif operation == "close":
            edited_mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)

        # Then dilate
        edited_mask = cv2.morphologyEx(edited_mask, cv2.MORPH_DILATE, kernel)

    except Exception as e:
        # logger.error(f'Unable to editMask!\n{e}')
        print((f"Unable to get editMask!\n{e}"))

    return edited_mask


@lD.log(logBase + ".sortContoursByArea")
def sortContoursByArea(logger, contours, reverse=True):

    """
    This function takes in list of contours, sorts them based
    on contour area, computes the bounding rectangle for each
    contour, and outputs the sorted contours and their
    corresponding bounding rectangles.

    Parameters
    ----------
    contours : {list}
        The list of contours to sort.

    Returns
    -------
    sorted_contours : {list}
        The list of contours sorted by contour area in descending
        order.
    bounding_boxes : {list}
        The list of bounding boxes ordered corresponding to the
        contours in `sorted_contours`.
    """

    try:
        # Sort contours based on contour area.
        sorted_contours = sorted(contours, key=cv2.contourArea, reverse=reverse)

        # Construct the list of corresponding bounding boxes.
        bounding_boxes = [cv2.boundingRect(c) for c in sorted_contours]

    except Exception as e:
        # logger.error(f'Unable to sortContourByArea!\n{e}')
        print((f"Unable to get sortContourByArea!\n{e}"))

    return sorted_contours, bounding_boxes


@lD.log(logBase + ".xLargestBlobs")
def xLargestBlobs(logger, mask, top_x=None, reverse=True):

    """
    This function finds contours in the given image and
    keeps only the top X largest ones.

    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to get the top X largest blobs.
    top_x : {int}
        The top X contours to keep based on contour area
        ranked in decesnding order.


    Returns
    -------
    n_contours : {int}
        The number of contours found in the given `mask`.
    X_largest_blobs : {numpy.ndarray}
        The corresponding mask of the image containing only
        the top X largest contours in white.
    """
    try:
        # Find all contours from binarised image.
        # Note: parts of the image that you want to get should be white.
        contours, hierarchy = cv2.findContours(
            image=mask, mode=cv2.RETR_EXTERNAL, method=cv2.CHAIN_APPROX_NONE
        )

        n_contours = len(contours)

        # Only get largest blob if there is at least 1 contour.
        if n_contours > 0:

            # Make sure that the number of contours to keep is at most equal
            # to the number of contours present in the mask.
            if n_contours < top_x or top_x == None:
                top_x = n_contours

            # Sort contours based on contour area.
            sorted_contours, bounding_boxes = sortContoursByArea(
                contours=contours, reverse=reverse
            )

            # Get the top X largest contours.
            X_largest_contours = sorted_contours[0:top_x]

            # Create black canvas to draw contours on.
            to_draw_on = np.zeros(mask.shape, np.uint8)

            # Draw contours in X_largest_contours.
            X_largest_blobs = cv2.drawContours(
                image=to_draw_on,  # Draw the contours on `to_draw_on`.
                contours=X_largest_contours,  # List of contours to draw.
                contourIdx=-1,  # Draw all contours in `contours`.
                color=1,  # Draw the contours in white.
                thickness=-1,  # Thickness of the contour lines.
            )

    except Exception as e:
        # logger.error(f'Unable to xLargestBlobs!\n{e}')
        print((f"Unable to get xLargestBlobs!\n{e}"))

    return n_contours, X_largest_blobs


@lD.log(logBase + ".applyMask")
def applyMask(logger, img, mask):

    """
    This function applies a mask to a given image. White
    areas of the mask are kept, while black areas are
    removed.

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to mask.
    mask : {numpy.ndarray, dtype=np.uint8}
        The mask to apply.

    Returns
    -------
    masked_img: {numpy.ndarray}
        The masked image.
    """

    try:
        masked_img = img.copy()
        masked_img[mask == 0] = 0

    except Exception as e:
        # logger.error(f'Unable to applyMask!\n{e}')
        print((f"Unable to get applyMask!\n{e}"))

    return masked_img


@lD.log(logBase + ".checkLRFlip")
def checkLRFlip(logger, mask):

    """
    This function checks whether or not an image needs to be
    flipped horizontally (i.e. left-right flip). The correct
    orientation is the breast being on the left (i.e. facing
    right).

    Parameters
    ----------
    mask : {numpy.ndarray, dtype=np.uint8}
        The corresponding mask of the image to flip.

    Returns
    -------
    LR_flip : {boolean}
        True means need to flip horizontally,
        False means otherwise.
    """

    try:
        # Get number of rows and columns in the image.
        nrows, ncols = mask.shape
        x_center = ncols // 2
        y_center = nrows // 2

        # Sum down each column.
        col_sum = mask.sum(axis=0)
        # Sum across each row.
        row_sum = mask.sum(axis=1)

        left_sum = sum(col_sum[0:x_center])
        right_sum = sum(col_sum[x_center:-1])

        if left_sum < right_sum:
            LR_flip = True
        else:
            LR_flip = False

    except Exception as e:
        # logger.error(f'Unable to checkLRFlip!\n{e}')
        print((f"Unable to get checkLRFlip!\n{e}"))

    return LR_flip


@lD.log(logBase + ".makeLRFlip")
def makeLRFlip(logger, img):

    """
    This function flips a given image horizontally (i.e. left-right).

    Parameters
    ----------
    img : {numpy.ndarray}
        The image to flip.

    Returns
    -------
    flipped_img : {numpy.ndarray}
        The flipped image.
    """

    try:
        flipped_img = np.fliplr(img)
    except Exception as e:
        # logger.error(f'Unable to makeLRFlip!\n{e}')
        print((f"Unable to get makeLRFlip!\n{e}"))

    return flipped_img


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


@lD.log(logBase + ".fullMammoPreprocess")
def fullMammoPreprocess(
    logger,
    img,
    l,
    r,
    d,
    u,
    thresh,
    maxval,
    ksize,
    operation,
    reverse,
    top_x,
    clip,
    tile,
):

    """
    This function chains and executes all the preprocessing
    steps for a full mammogram, in the following order:

    Step 1 - Initial crop
    Step 2 - Min-max normalise
    Step 3 - Remove artefacts
    Step 4 - Horizontal flip (if needed)
    Step 5 - CLAHE enchancement
    Step 6 - Pad
    Step 7 - Downsample (?)
    Step 8 - Min-max normalise

    Parameters
    ----------
    img : {numpy.ndarray}
        The full mammogram image to preprocess.

    Returns
    -------
    img_pre : {numpy.ndarray}
        The preprocessed full mammogram image.
    lr_flip : {boolean}
        If True, the corresponding ROI mask needs to be
        flipped horizontally, otherwise no need to flip.
    """

    try:
        # Step 1: Initial crop.
        cropped_img = cropBorders(img=img, l=l, r=r, d=d, u=u)
        # cv2.imwrite("../data/preprocessed/Mass/testing/cropped.png", cropped_img)

        # Step 2: Min-max normalise.
        norm_img = minMaxNormalise(img=cropped_img)
        # cv2.imwrite("../data/preprocessed/Mass/testing/normed.png", norm_img)

        # Step 3: Remove artefacts.
        binarised_img = globalBinarise(img=norm_img, thresh=thresh, maxval=maxval)
        edited_mask = editMask(
            mask=binarised_img, ksize=(ksize, ksize), operation=operation
        )
        _, xlargest_mask = xLargestBlobs(mask=edited_mask, top_x=top_x, reverse=reverse)
        # cv2.imwrite(
        # "../data/preprocessed/Mass/testing/xLargest_mask.png", xlargest_mask
        # )
        masked_img = applyMask(img=norm_img, mask=xlargest_mask)
        # cv2.imwrite("../data/preprocessed/Mass/testing/masked_img.png", masked_img)

        # Step 4: Horizontal flip.
        lr_flip = checkLRFlip(mask=xlargest_mask)
        if lr_flip:
            flipped_img = makeLRFlip(img=masked_img)
        elif not lr_flip:
            flipped_img = masked_img
        # cv2.imwrite("../data/preprocessed/Mass/testing/flipped_img.png", flipped_img)

        # Step 5: CLAHE enhancement.
        clahe_img = clahe(img=flipped_img, clip=clip, tile=(tile, tile))
        # cv2.imwrite("../data/preprocessed/Mass/testing/clahe_img.png", clahe_img)

        # Step 6: pad.
        padded_img = pad(img=clahe_img)
        padded_img = cv2.normalize(
            padded_img,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )
        # cv2.imwrite("../data/preprocessed/Mass/testing/padded_img.png", padded_img)

        # Step 7: Downsample.
        # Not done yet.

        # Step 8: Min-max normalise.
        img_pre = minMaxNormalise(img=padded_img)
        # cv2.imwrite("../data/preprocessed/Mass/testing/img_pre.png", img_pre)

    except Exception as e:
        # logger.error(f'Unable to fullMammPreprocess!\n{e}')
        print((f"Unable to fullMammPreprocess!\n{e}"))

    return img_pre, lr_flip


@lD.log(logBase + ".maskPreprocess")
def maskPreprocess(logger, mask, lr_flip):

    """
    This function chains and executes all the preprocessing
    steps necessary for a ROI mask image.

    Step 1 - Initial crop
    Step 2 - Horizontal flip (if needed)
    Step 3 - Pad
    Step 4 - Downsample (?)

    Parameters
    ----------
    mask : {numpy.ndarray}
        The ROI mask image to preprocess.
    lr_flip : {boolean}
        If True, the ROI mask needs to be
        flipped horizontally, otherwise no need to flip.

    Returns
    -------
    mask_pre : {numpy.ndarray}
        The preprocessed ROI mask image.
    """

    # Step 1: Initial crop.
    mask = cropBorders(img=mask)

    # Step 2: Horizontal flip.
    if lr_flip:
        mask = makeLRFlip(img=mask)

    # Step 3: Pad.
    mask_pre = pad(img=mask)

    # Step 4: Downsample.

    return mask_pre


# ----------------------------------


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for imagePreprocessing module.

    This function takes a path of the raw image folder,
    iterates through each image and executes the necessary
    image preprocessing steps on each image, and saves
    preprocessed images (in the specified file extension)
    in the output paths specified. FULL and MASK images are
    saved in their specified separate folders.

    The hyperparameters in this module can be tuned in
    the "../config/modules/imagePreprocessing.json" file.

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
    print("Main function of imagePreprocessing.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Get path of the folder containing .dcm files.
    input_path = config_imgPre["paths"]["input"]
    output_full_path = config_imgPre["paths"]["output_full"]
    output_mask_path = config_imgPre["paths"]["output_mask"]

    # Output format.
    output_format = config_imgPre["output_format"]

    # Get individual .dcm paths.
    dcm_paths = []
    for curdir, dirs, files in os.walk(input_path):
        files.sort()
        for f in files:
            if f.endswith(".dcm"):
                dcm_paths.append(os.path.join(curdir, f))

    # Get paths of full mammograms and ROI masks.
    fullmamm_paths = [f for f in dcm_paths if ("FULL" in f)]
    mask_paths = [f for f in dcm_paths if ("MASK" in f)]

    count = 0
    for fullmamm_path in fullmamm_paths:

        # Read full mammogram .dcm file.
        ds = pydicom.dcmread(fullmamm_path)

        # Get relevant metadata from .dcm file.
        patient_id = ds.PatientID

        # Calc-Test masks do not have the "Calc-Test_" suffix
        # when it was originally downloaded (masks from Calc-Train,
        # Mass-Test and Mass-Train all have their corresponding suffices).
        patient_id = patient_id.replace(".dcm", "")
        patient_id = patient_id.replace("Calc-Test_", "")

        fullmamm = ds.pixel_array

        # =========================
        # Preprocess Full Mammogram
        # =========================

        # Get all hyperparameters.
        l = config_imgPre["cropBorders"]["l"]
        r = config_imgPre["cropBorders"]["r"]
        u = config_imgPre["cropBorders"]["u"]
        d = config_imgPre["cropBorders"]["d"]
        thresh = config_imgPre["globalBinarise"]["thresh"]
        maxval = config_imgPre["globalBinarise"]["maxval"]
        ksize = config_imgPre["editMask"]["ksize"]
        operation = config_imgPre["editMask"]["operation"]
        reverse = config_imgPre["sortContourByArea"]["reverse"]
        top_x = config_imgPre["xLargestBlobs"]["top_x"]
        clip = config_imgPre["clahe"]["clip"]
        tile = config_imgPre["clahe"]["tile"]

        # Preprocess full mammogram images.
        fullmamm_pre, lr_flip = fullMammoPreprocess(
            img=fullmamm,
            l=l,
            r=r,
            u=u,
            d=d,
            thresh=thresh,
            maxval=maxval,
            ksize=ksize,
            operation=operation,
            reverse=reverse,
            top_x=top_x,
            clip=clip,
            tile=tile,
        )

        # Need to normalise to [0, 255] before saving as .png.
        fullmamm_pre_norm = cv2.normalize(
            fullmamm_pre,
            None,
            alpha=0,
            beta=255,
            norm_type=cv2.NORM_MINMAX,
            dtype=cv2.CV_32F,
        )

        # Save preprocessed full mammogram image.
        save_filename = (
            os.path.basename(fullmamm_path).replace(".dcm", "")
            + "___PRE"
            + output_format
        )
        save_path = os.path.join(output_full_path, save_filename)
        cv2.imwrite(save_path, fullmamm_pre_norm)
        print(f"DONE FULL: {fullmamm_path}")

        # ================================
        # Preprocess Corresponding Mask(s)
        # ================================

        # Get the path of corresponding ROI mask(s) .dcm file(s).
        mask_path = [mp for mp in mask_paths if patient_id in mp]

        for mp in mask_path:

            # Read mask(s) .dcm file(s).
            mask_ds = pydicom.dcmread(mp)
            mask = mask_ds.pixel_array

            # Preprocess.
            mask_pre = maskPreprocess(mask=mask, lr_flip=lr_flip)

            # Save preprocessed mask.
            save_filename = (
                os.path.basename(mp).replace(".dcm", "") + "___PRE" + output_format
            )
            save_path = os.path.join(output_mask_path, save_filename)
            cv2.imwrite(save_path, mask_pre)

            print(f"DONE MASK: {mp}")

        count += 1

        # if count == 1:
        #     break

    print(f"Total count = {count}")
    print()
    print("Getting out of imagePreprocessing module.")
    print("-" * 30)

    return
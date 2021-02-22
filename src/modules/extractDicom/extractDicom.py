import os
import shutil
import pprint
import jsonref
import pandas as pd
import numpy as np
import pydicom
from pathlib import Path

from logs import logDecorator as lD

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.extractDicom.extractDicom"
config_extractDicom = jsonref.load(open("../config/modules/extractDicom.json"))


@lD.log(logBase + ".new_name_dcm")
def new_name_dcm(logger, dcm_path):

    """
    This function takes the absolute path of a .dcm file
    and renames it according to the convention below:

    1. Full mammograms:
        - Mass-Training_P_00001_LEFT_CC_FULL.dcm
    2. Cropped image:
        - Mass-Training_P_00001_LEFT_CC_CROP_1.dcm
        - Mass-Training_P_00001_LEFT_CC_CROP_2.dcm
        - ...
    3. Mask image:
        - Mass-Training_P_00001_LEFT_CC_MASK_1.dcm
        - Mass-Training_P_00001_LEFT_CC_MASK_2.dcm
        - ...


    Parameters
    ----------
    dcm_path : {str}
        The relative (or absolute) path of the .dcm file
        to rename, including the .dcm filename.
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC/1-1.dcm"

    Returns
    -------
    new_name : {str}
        The new name that the .dcm file should have
        WITH the ".dcm" extention WITHOUT its relative
        (or absolute) path.
        e.g. "Mass-Training_P_00001_LEFT_CC_FULL.dcm"
    False : {boolean}
        False is returned if the new name of the .dcm
        file cannot be determined.
    """

    try:
        # Read dicom.
        ds = pydicom.dcmread(dcm_path)

        # Get information.
        patient_id = ds.PatientID
        patient_id = patient_id.replace(".dcm", "")

        try:
            # If ds contains SeriesDescription attribute...
            img_type = ds.SeriesDescription

            # === FULL ===
            if "full" in img_type:
                new_name = patient_id + "_FULL" + ".dcm"
                print(f"FULL --- {new_name}")
                return new_name

            # === CROP ===
            elif "crop" in img_type:

                # Double check if suffix is integer.
                suffix = patient_id.split("_")[-1]

                if suffix.isdigit():
                    new_patient_id = patient_id.split("_" + suffix)[0]
                    new_name = new_patient_id + "_CROP" + "_" + suffix + ".dcm"
                    print(f"CROP --- {new_name}")
                    return new_name

                elif not suffix.isdigit():
                    print(f"CROP ERROR, {patient_id}")
                    return False

            # === MASK ===
            elif "mask" in img_type:

                # Double check if suffix is integer.
                suffix = patient_id.split("_")[-1]

                if suffix.isdigit():
                    new_patient_id = patient_id.split("_" + suffix)[0]
                    new_name = new_patient_id + "_MASK" + "_" + suffix + ".dcm"
                    print(f"MASK --- {new_name}")
                    return new_name

                elif not suffix.isdigit():
                    print(f"MASK ERROR, {patient_id}")
                    return False

        except:
            # If ds does not contain SeriesDescription...
            # === FULL ===
            if "full" in dcm_path:
                new_name = patient_id + "_FULL" + ".dcm"
                return new_name

            else:
                # Read the image to decide if its a mask or crop.
                # MASK only has pixel values {0, 1}
                arr = ds.pixel_array
                unique = np.unique(arr).tolist()

                if len(unique) != 2:

                    # === CROP ===
                    # Double check if suffix is integer.
                    suffix = patient_id.split("_")[-1]

                    if suffix.isdigit():
                        new_patient_id = patient_id.split("_" + suffix)[0]
                        new_name = new_patient_id + "_CROP" + "_" + suffix + ".dcm"
                        print(f"CROP --- {new_name}")
                        return new_name

                    elif not suffix.isdigit():
                        print(f"CROP ERROR, {patient_id}")
                        return False

                elif len(unique) == 2:

                    # === MASK ===
                    # Double check if suffix is integer.
                    suffix = patient_id.split("_")[-1]

                    if suffix.isdigit():
                        new_patient_id = patient_id.split("_" + suffix)[0]
                        new_name = new_patient_id + "_MASK" + "_" + suffix + ".dcm"
                        print(f"MASK --- {new_name}")
                        return new_name

                    elif not suffix.isdigit():
                        print(f"MASK ERROR, {patient_id}")
                        return False
                else:
                    return img_type

    except Exception as e:
        # logger.error(f'Unable to new_name_dcm!\n{e}')
        print((f"Unable to new_name_dcm!\n{e}"))


@lD.log(logBase + ".move_dcm_up")
def move_dcm_up(logger, dest_dir, source_dir, dcm_filename):

    """
    This function move a .dcm file from its given source
    directory into the given destination directory. It also
    handles conflicting filenames by adding "___a" to the
    end of a filename if the filename already exists in the
    destination directory.

    Parameters
    ----------
    dest_dir : {str}
        The relative (or absolute) path of the folder that
        the .dcm file needs to be moved to.
    source_dir : {str}
        The relative (or absolute) path where the .dcm file
        needs to be moved from, including the filename.
        e.g. "source_folder/Mass-Training_P_00001_LEFT_CC_FULL.dcm"
    dcm_filename : {str}
        The name of the .dcm file WITH the ".dcm" extension
        but WITHOUT its (relative or absolute) path.
        e.g. "Mass-Training_P_00001_LEFT_CC_FULL.dcm".

    Returns
    -------
    None
    """

    try:
        dest_dir_with_new_name = os.path.join(dest_dir, dcm_filename)

        # If the destination path does not exist yet...
        if not os.path.exists(dest_dir_with_new_name):
            shutil.move(source_dir, dest_dir)

        # If the destination path already exists...
        elif os.path.exists(dest_dir_with_new_name):
            # Add "_a" to the end of `new_name` generated above.
            new_name_2 = dcm_filename.strip(".dcm") + "___a.dcm"
            # This moves the file into the destination while giving the file its new name.
            shutil.move(source_dir, os.path.join(dest_dir, new_name_2))

    except Exception as e:
        # logger.error(f'Unable to move_dcm_up!\n{e}')
        print((f"Unable to move_dcm_up!\n{e}"))


@lD.log(logBase + ".delete_empty_folders")
def delete_empty_folders(logger, top, error_dir):

    """
    This function recursively walks through a given directory
    (`top`) using depth-first search (bottom up) and deletes
    any directory that is empty (ignoring hidden files).
    If there are directories that are not empty (except hidden
    files), it will save the absolute directory in a Pandas
    dataframe and export it as a `not-empty-folders.csv` to
    `error_dir`.

    Parameters
    ----------
    top : {str}
        The directory to iterate through.
    error_dir : {str}
        The directory to save the `not-empty-folders.csv` to.

    Returns
    -------
    None
    """

    try:
        curdir_list = []
        files_list = []

        for (curdir, dirs, files) in os.walk(top=top, topdown=False):

            if curdir != str(top):

                dirs.sort()
                files.sort()

                print(f"WE ARE AT: {curdir}")
                print("=" * 10)

                print("List dir:")

                directories_list = [
                    f for f in os.listdir(curdir) if not f.startswith(".")
                ]
                print(directories_list)

                if len(directories_list) == 0:
                    print("DELETE")
                    shutil.rmtree(curdir, ignore_errors=True)

                elif len(directories_list) > 0:
                    print("DON'T DELETE")
                    curdir_list.append(curdir)
                    files_list.append(directories_list)

                print()
                print("Moving one folder up...")
                print("-" * 40)
                print()

        if len(curdir_list) > 0:
            not_empty_df = pd.DataFrame(
                list(zip(curdir_list, files_list)), columns=["curdir", "files"]
            )
            to_save_path = os.path.join(error_dir, "not-empty-folders.csv")
            not_empty_df.to_csv(to_save_path, index=False)

    except Exception as e:
        # logger.error(f'Unable to delete_empty_folders!\n{e}')
        print((f"Unable to delete_empty_folders!\n{e}"))


@lD.log(logBase + ".count_dcm")
def count_dcm(logger, top):

    """
    This function recursively walks through a given directory
    (`top`) using depth-first search (bottom up) and counts the
    number of .dcm files present.

    Parameters
    ----------
    path : {str}
        The directory to count.

    Returns
    -------
    count : {int}
        The number of .dcm files in `path`.
    """

    try:
        count = 0

        # Count number of .dcm files in ../data/Mass/Test.
        for _, _, files in os.walk(top):
            for f in files:
                if f.endswith(".dcm"):
                    count += 1

    except Exception as e:
        # logger.error(f'Unable to count_dcm!\n{e}')
        print((f"Unable to count_dcm!\n{e}"))

    return count


# ----------------------------------


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for extractDicom module.

    >>>>>>> This function takes a path of the raw image folder,
    iterates through each image and executes the necessary
    image preprocessing steps on each image, and saves
    preprocessed images in the output path specified.

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
    print("Main function of extractDicom module.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Get the path to the folder that contains all the nested .dcm files.
    top = Path(config_extractDicom["top"])

    # ==============================================
    # 1. Count number of .dcm files BEFORE executing
    # ==============================================
    before = count_dcm(top=top)

    # ==========
    # 2. Execute
    # ==========

    # 2.1. Rename and move .dcm files.
    # --------------------------------
    for (curdir, dirs, files) in os.walk(top=top, topdown=False):

        dirs.sort()
        files.sort()

        for f in files:

            # === Step 1: Rename .dcm file ===
            if f.endswith(".dcm"):

                old_name_path = os.path.join(curdir, f)
                new_name = new_name_dcm(dcm_path=old_name_path)

                if new_name:
                    new_name_path = os.path.join(curdir, new_name)
                    os.rename(old_name_path, new_name_path)

                    # === Step 2: Move RENAMED .dcm file ===
                    move_dcm_up(
                        dest_dir=top, source_dir=new_name_path, dcm_filename=new_name
                    )

    # 2.2. Delete empty folders.
    # --------------------------
    delete_empty_folders(top=top, error_dir=top)

    # =============================================
    # 3. Count number of .dcm files AFTER executing
    # =============================================
    after = count_dcm(top=top)

    print(f"BEFORE --> Number of .dcm files: {before}")
    print(f"AFTER --> Number of .dcm files: {after}")
    print()
    print("Getting out of extractDicom.")
    print("-" * 30)

    return

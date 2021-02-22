import os
import pprint
import jsonref
import numpy as np
import pandas as pd
from pathlib import Path

from logs import logDecorator as lD

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.updateDcmPath.updateDcmPath"
config_updateDcmPath = jsonref.load(open("../config/modules/updateDcmPath.json"))


@lD.log(logBase + ".cropBorders")
def updateDcmPath(logger, og_df, dcm_folder):

    """
    This function updates paths to the full mammogram scan,
    cropped image and ROI mask of each row (.dcm file) of the
    given DataFrame.

    Parameters
    ----------
    og_df : {pd.DataFrame}
        The original Pandas DataFrame that needs to be updated.
    dcm_folder : {str}
        The relative (or absolute) path to the folder that conrains
        all the .dcm files to get the path.

    Returns
    -------
    og_df: {pd.DataFrame}
        The Pandas DataFrame with all the updated .dcm paths.
    """

    try:

        # Creat new columns in og_df.
        og_df["full_path"] = np.nan
        og_df["crop_path"] = np.nan
        og_df["mask_path"] = np.nan

        # Get list of .dcm paths.
        dcm_paths_list = []
        for _, _, files in os.walk(dcm_folder):
            for f in files:
                if f.endswith(".dcm"):
                    dcm_paths_list.append(os.path.join(dcm_folder, f))

        for row in og_df.itertuples():

            row_id = row.Index

            # Get identification details.
            patient_id = row.patient_id
            img_view = row.image_view
            lr = row.left_or_right_breast
            abnormality_id = row.abnormality_id

            # Use this list to match DF row with .dcm path.
            info_list = [patient_id, img_view, lr]

            crop_suffix = "CROP_" + str(abnormality_id)
            mask_suffix = "MASK_" + str(abnormality_id)

            # Get list of relevant paths to this patient.
            full_paths = [
                path
                for path in dcm_paths_list
                if all(info in path for info in info_list + ["FULL"])
            ]

            crop_paths = [
                path
                for path in dcm_paths_list
                if all(info in path for info in info_list + [crop_suffix])
            ]

            mask_paths = [
                path
                for path in dcm_paths_list
                if all(info in path for info in info_list + [mask_suffix])
            ]

            # full_paths_str = ",".join(full_paths)
            # crop_paths_str = ",".join(crop_paths)
            # mask_paths_str = ",".join(mask_paths)

            # Update paths.
            if len(full_paths) > 0:
                og_df.loc[row_id, "full_path"] = full_paths
            if len(crop_paths) > 0:
                og_df.loc[row_id, "crop_path"] = crop_paths
            if len(mask_paths) > 0:
                og_df.loc[row_id, "mask_path"] = mask_paths

    except Exception as e:
        # logger.error(f'Unable to updateDcmPath!\n{e}')
        print((f"Unable to get updateDcmPath!\n{e}"))

    return og_df


# ----------------------------------


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for updateDcmPath module.

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
    print("Main function of updateDcmPath.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Get the input paths to the required files.
    mass_train_csv_path = config_updateDcmPath["input"]["mass"]["train"]["csv"]
    mass_train_dcm_folder = config_updateDcmPath["input"]["mass"]["train"]["dcm_folder"]
    mass_test_csv_path = config_updateDcmPath["input"]["mass"]["test"]["csv"]
    mass_test_dcm_folder = config_updateDcmPath["input"]["mass"]["test"]["dcm_folder"]

    # Read the .csv files.
    og_mass_train_df = pd.read_csv(mass_train_csv_path)
    og_mass_test_df = pd.read_csv(mass_test_csv_path)

    # Rename columns.
    new_train_cols = [col.replace(" ", "_") for col in og_mass_train_df.columns]
    og_mass_train_df.columns = new_train_cols

    new_test_cols = [col.replace(" ", "_") for col in og_mass_test_df.columns]
    og_mass_test_df.columns = new_test_cols

    # Update .dcm paths.
    updated_mass_train_df = updateDcmPath(
        og_df=og_mass_train_df, dcm_folder=mass_train_dcm_folder
    )

    updated_mass_test_df = updateDcmPath(
        og_df=og_mass_test_df, dcm_folder=mass_test_dcm_folder
    )

    # Get output folder path and save updated .csv.
    train_output_path = Path(config_updateDcmPath["output"]["mass"]["train"]["csv"])
    test_output_path = Path(config_updateDcmPath["output"]["mass"]["test"]["csv"])

    if config_updateDcmPath["save_output"]:
        updated_mass_train_df.to_csv(train_output_path, index=False)
        updated_mass_test_df.to_csv(test_output_path, index=False)

    print("Getting out of updateDcmPath.")
    print("-" * 30)

    return

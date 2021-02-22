import os
import pprint
import jsonref
from pathlib import Path

from logs import logDecorator as lD

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.countFileType.countFileType"
config_countFileType = jsonref.load(open("../config/modules/countFileType.json"))


@lD.log(logBase + ".count_file_type")
def countFileType(logger, top, extension):

    """
    This function recursively walks through a given directory
    (`top`) using depth-first search (bottom up) and counts the
    number of files present.

    Parameters
    ----------
    top : {str}
        The directory to count.
    extension : {str}
        The type of file to count.

    Returns
    -------
    full_count : {int}
        The number of full mammogram .dcm files in `top`.
    crop_count : {int}
        The number of cropped mammogram .dcm files in `top`.
    mask_count : {int}
        The number of ROI mask .dcm files in `top`.
    unknown_count : {int}
        The number of .dcm files in `top` that are unidentified.
    """

    try:
        full_count = 0
        crop_count = 0
        mask_count = 0
        unknown_count = 0

        # Count number of .dcm files in ../data/Mass/Test.
        for _, _, files in os.walk(top):
            for f in files:
                if f.endswith(extension) and "FULL" in f:
                    full_count += 1
                elif f.endswith(extension) and "CROP" in f:
                    crop_count += 1
                elif f.endswith(extension) and "MASK" in f:
                    mask_count += 1
                elif f.endswith(extension) and all(
                    s not in f for s in ["FULL", "MASK", "CROP"]
                ):
                    unknown_count += 1

    except Exception as e:
        # logger.error(f'Unable to countFileType!\n{e}')
        print((f"Unable to countFileType!\n{e}"))

    return full_count, crop_count, mask_count, unknown_count


# ----------------------------------


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for countDicom module.

    This function recursively counts the number of .dcm files in
    the given directory (i.e. includes all its subdirectories).

    Parameters
    ----------
    logger : {logging.Logger}
        The logger used for logging error information
    resultsDict: {dict}
        A dictionary containing information about the
        command line arguments. These can be used for
        overwriting command line arguments as needed.
    """

    print("=" * 30)
    print("Main function of countFileType module.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Get the path to the folder that contains all the nested .dcm files.
    top = Path(config_countFileType["top"])
    extension = config_countFileType["extension"]

    # Count.
    full_count, crop_count, mask_count, unknown_count = countFileType(
        top=top, extension=extension
    )

    print(f"Number of FULL {extension} files: {full_count}")
    print(f"Number of CROP {extension} files: {crop_count}")
    print(f"Number of MASK {extension} files: {mask_count}")
    print(f"Number of UNKNOWN {extension} files: {unknown_count}")
    print()
    print("Getting out of countFileType.")
    print("-" * 30)

    return

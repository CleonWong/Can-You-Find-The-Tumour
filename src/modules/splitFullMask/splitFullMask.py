import os
import pprint
import jsonref
from pathlib import Path
import shutil

from logs import logDecorator as lD

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = config["logging"]["logBase"] + ".modules.splitFullMask.splitFullMask"
config_splitFullMask = jsonref.load(open("../config/modules/splitFullMask.json"))


@lD.log(logBase + ".splitFullMask")
def splitFullMask(logger, top, FULL_or_MASK, extension, copy_to):

    """
    This function recursively walks through a given directory
    (`top`) using depth-first search (bottom up), finds file names
    containing the `FULL_or_MASK` substring and copies it to the
    target directory `copy_to`.

    Parameters
    ----------
    top : {str}
        The directory to look in.
    FULL_or_MASK : {str}
        The substring to look for, either "FULL" or "MASK".
    extension : {str}
        The extension of the file to look for. e.g. ".png".
    copy_to : {str}
        The directory to copy to.

    Returns
    -------
    files_moved : {int}
        The number of files moved.
    """

    try:
        files_moved = 0

        # Count number of .dcm files in ../data/Mass/Test.
        for curdir, _, files in os.walk(top):

            files.sort()

            for f in files:

                if f.endswith(extension) and FULL_or_MASK in f:

                    source = os.path.join(curdir, f)
                    dest = os.path.join(copy_to, f)
                    shutil.move(source, dest)

                    files_moved += 1
                # if files_moved == 1:
                #     break

    except Exception as e:
        # logger.error(f'Unable to splitFullMask!\n{e}')
        print((f"Unable to splitFullMaskk!\n{e}"))

    return files_moved


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
    print("Main function of splitFullMask module.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Get the path to the folder that contains all the nested .dcm files.
    top = Path(config_splitFullMask["top"])
    FULL_or_MASK = config_splitFullMask["FULL_or_MASK"]
    extension = config_splitFullMask["extension"]
    copy_to = Path(config_splitFullMask["copy_to"])

    # Count.
    files_moved = splitFullMask(
        top=top, FULL_or_MASK=FULL_or_MASK, extension=extension, copy_to=copy_to
    )

    print(f"Number of files with '{FULL_or_MASK}' moved: {files_moved}")
    print()
    print("Getting out of countFileType.")
    print("-" * 30)
    print()

    return

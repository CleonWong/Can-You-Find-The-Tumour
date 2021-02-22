import os
import pprint
import jsonref
from pathlib import Path
import shutil
import random

from logs import logDecorator as lD

# ----------------------------------

config = jsonref.load(open("../config/config.json"))
logBase = (
    config["logging"]["logBase"] + ".modules.redoTrainTestSplit.redoTrainTestSplit"
)
config_redoTTS = jsonref.load(open("../config/modules/redoTrainTestSplit.json"))


@lD.log(logBase + ".getMultiTumourIdentifiers")
def getMultiTumourIdentifiers(
    logger, extension, train_summed_masks_dir, test_summed_masks_dir
):

    """
    This function gets the names of all the files that are contains more than
    one tumour.

    Parameters
    ----------
    extension : {str}
        The extension of the file to look for. e.g. ".png".
    train_summed_masks_dir : {str}
        The directory of the train masks (original train split) with more than
        one tumour.
    test_summed_masks_dir : {str}
        The directory of the test masks (original test split) with more than
        one tumour.

    Returns
    -------
    multi_tumour_identifier : {list}
        List of "Mass-Training_P_XXXXX_LEFT_CC" or
        ""Mass-Test_P_XXXXX_LEFT_CC".
    """

    try:
        multi_tumour_identifier = []

        # Train MASKs.
        for f in os.listdir(train_summed_masks_dir):

            if f.endswith(extension):
                identifier = f.replace("_MASK___PRE.png", "")
                multi_tumour_identifier.append(identifier)

        # Test MASKs.
        for f in os.listdir(test_summed_masks_dir):

            if f.endswith(extension):
                identifier = f.replace("_MASK___PRE.png", "")
                multi_tumour_identifier.append(identifier)

        print()
        print(f"Number of images with > 1 tumours: {len(multi_tumour_identifier)}")

    except Exception as e:
        # logger.error(f'Unable to getMultiTumourIdentifiers!\n{e}')
        print((f"Unable to getMultiTumourIdentifiers!\n{e}"))

    return multi_tumour_identifier


@lD.log(logBase + ".getTrainTestSplit")
def getTrainTestSplit(
    logger,
    extension,
    multi_tumour_identifier,
    all_full_dir,
    all_mask_dir,
    test_split,
):
    """
    This function gets the names of all the files that are contains more than
    one tumour.

    Parameters
    ----------
    extension : {str}
        The extension of the file to look for. e.g. ".png".
    train_summed_masks_dir : {str}
        The directory of the train masks (original train split) with more than
        one tumour.
    test_summed_masks_dir : {str}
        The directory of the test masks (original test split) with more than
        one tumour.

    Returns
    -------
    multi_tumour_identifier : {list}
        List of "Mass-Training_P_XXXXX_LEFT_CC" or
        ""Mass-Test_P_XXXXX_LEFT_CC".
    """

    try:

        # Get lengths of train and test sets.
        total_len1 = len(multi_tumour_identifier)
        test_len1 = round(total_len1 * test_split)

        # Get a random sample of names without replacement.
        sampled_multi_tumour_test = random.sample(multi_tumour_identifier, test_len1)
        sampled_multi_tumour_train = [
            q for q in multi_tumour_identifier if q not in sampled_multi_tumour_test
        ]

        # Get list of all FULL and MASK images.
        all_full_names = []
        for f in os.listdir(all_full_dir):
            if f.endswith(extension):
                all_full_names.append(f)

        all_mask_names = []
        for f in os.listdir(all_mask_dir):
            if f.endswith(extension):
                all_mask_names.append(f)

        print(f"Total number of FULL images: {len(all_full_names)}")
        print(f"Total number of MASK images: {len(all_mask_names)}")

        # Get list of FULL and MASK images that only have 1 tumour.
        onetumour_full_names = []
        for full_name in all_full_names:
            if not any(
                multi_iden in full_name for multi_iden in multi_tumour_identifier
            ):
                onetumour_full_names.append(full_name)

        onetumour_mask_names = []
        for mask_name in all_mask_names:
            if not any(
                multi_iden in mask_name for multi_iden in multi_tumour_identifier
            ):
                onetumour_mask_names.append(mask_name)

        print(f"Number of FULL images of 1 tumour: {len(onetumour_full_names)}")
        print(f"Number of MASK images of 1 tumour: {len(onetumour_mask_names)}")

        total_len2 = len(onetumour_full_names)
        test_len2 = round(total_len2 * test_split)

        sampled_onetumour_full_test = random.sample(onetumour_full_names, test_len2)
        sampled_onetumour_test_identifier = [
            q.replace("_FULL___PRE.png", "") for q in sampled_onetumour_full_test
        ]

        sampled_onetumour_full_test = []
        for full_name in onetumour_full_names:
            if any(iden in full_name for iden in sampled_onetumour_test_identifier):
                sampled_onetumour_full_test.append(full_name)

        sampled_onetumour_mask_test = []
        for mask_name in onetumour_mask_names:
            if any(iden in mask_name for iden in sampled_onetumour_test_identifier):
                sampled_onetumour_mask_test.append(mask_name)

        sampled_onetumour_full_train = [
            q for q in onetumour_full_names if q not in sampled_onetumour_full_test
        ]
        sampled_onetumour_mask_train = [
            q for q in onetumour_mask_names if q not in sampled_onetumour_mask_test
        ]

        # Combine
        sampled_full_train = sampled_onetumour_full_train + sampled_multi_tumour_train
        sampled_mask_train = sampled_onetumour_mask_train + sampled_multi_tumour_train
        sampled_full_test = sampled_onetumour_full_test + sampled_multi_tumour_test
        sampled_mask_test = sampled_onetumour_mask_test + sampled_multi_tumour_test

        sampled_full_train.sort()
        sampled_mask_train.sort()
        sampled_full_test.sort()
        sampled_mask_test.sort()

        print()
        print("AFTER REDO SPLIT:")
        print(
            f"Number of (FULL, MASK) training images: {len(sampled_full_train)}, {len(sampled_mask_train)}"
        )
        print(
            f"Number of (FULL, MASK) test images: {len(sampled_full_test)}, {len(sampled_mask_test)}"
        )

    except Exception as e:
        # logger.error(f'Unable to getTrainTestSplit!\n{e}')
        print((f"Unable to getTrainTestSplit!\n{e}"))

    return sampled_full_train, sampled_mask_train, sampled_full_test, sampled_mask_test


@lD.log(logBase + ".copyFile")
def copyFile(
    logger,
    sampled_full_train,
    sampled_mask_train,
    sampled_full_test,
    sampled_mask_test,
    extension,
    all_full_dir,
    all_mask_dir,
    target_train_full_dir,
    target_train_mask_dir,
    target_test_full_dir,
    target_test_mask_dir,
):

    """
    [Summary]


    Parameters
    ----------


    Returns
    -------

    """

    try:

        train_full_dest = []
        train_mask_dest = []
        test_full_dest = []
        test_mask_dest = []

        # Copy FULL images first.
        # -----------------------
        for curdir, _, files in os.walk(all_full_dir):
            files.sort()

            for f in files:

                # Move train FULL.
                if f.endswith(extension) and any(
                    iden in f for iden in sampled_full_train
                ):

                    # Remove "Mass-Training_" from filename.
                    source = os.path.join(curdir, f)
                    dest = os.path.join(
                        target_train_full_dir,
                        f.replace("Mass-Training_", "").replace("Mass-Test_", ""),
                    )
                    train_full_dest.append(dest)
                    shutil.copyfile(source, dest)

                # Move test FULL.
                elif f.endswith(extension) and any(
                    iden in f for iden in sampled_full_test
                ):
                    # Remove "Mass-Test_" from filename.
                    source = os.path.join(curdir, f)
                    dest = os.path.join(
                        target_test_full_dir,
                        f.replace("Mass-Training_", "").replace("Mass-Test_", ""),
                    )
                    test_full_dest.append(dest)
                    shutil.copyfile(source, dest)

        # Then copy MASK images.
        # ----------------------
        for curdir, _, files in os.walk(all_mask_dir):
            files.sort()

            for f in files:

                # Move train MASK.
                if f.endswith(extension) and any(
                    iden in f for iden in sampled_mask_train
                ):
                    # Remove "Mass-Training_" from filename.
                    source = os.path.join(curdir, f)
                    dest = os.path.join(
                        target_train_mask_dir,
                        f.replace("Mass-Training_", "").replace("Mass-Test_", ""),
                    )
                    train_mask_dest.append(dest)
                    shutil.copyfile(source, dest)

                # Move test MASK.
                elif f.endswith(extension) and any(
                    iden in f for iden in sampled_mask_test
                ):
                    # Remove "Mass-Test_" from filename.
                    source = os.path.join(curdir, f)
                    dest = os.path.join(
                        target_test_mask_dir,
                        f.replace("Mass-Training_", "").replace("Mass-Test_", ""),
                    )
                    test_mask_dest.append(dest)
                    shutil.copyfile(source, dest)

    except Exception as e:
        # logger.error(f'Unable to copyFile!\n{e}')
        print((f"Unable to copyFile!\n{e}"))

    print()
    print("FILES COPIED:")
    print(
        f"Number of (FULL, MASK) train images copied: {len(train_full_dest)}, {len(train_mask_dest)}"
    )
    print(
        f"Number of (FULL, MASK) train images copied: {len(test_full_dest)}, {len(test_mask_dest)}"
    )

    return


# ----------------------------------


@lD.log(logBase + ".main")
def main(logger, resultsDict):
    """main function for redoTrainTestSplit module.

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
    print("Main function of redoTrainSplit module.")
    print("=" * 30)
    print("We get a copy of the result dictionary over here ...")
    pprint.pprint(resultsDict)

    # Get parameters.
    extension = config_redoTTS["extension"]
    train_summed_masks_dir = Path(config_redoTTS["train_summed_masks_dir"])
    test_summed_masks_dir = Path(config_redoTTS["test_summed_masks_dir"])
    all_full_dir = Path(config_redoTTS["all_full_dir"])
    all_mask_dir = Path(config_redoTTS["all_mask_dir"])
    test_split = config_redoTTS["test_split"]
    target_train_full_dir = Path(config_redoTTS["target_train_full_dir"])
    target_train_mask_dir = Path(config_redoTTS["target_train_mask_dir"])
    target_test_full_dir = Path(config_redoTTS["target_test_full_dir"])
    target_test_mask_dir = Path(config_redoTTS["target_test_mask_dir"])

    # Execute.
    multi_tumour_identifier = getMultiTumourIdentifiers(
        extension=extension,
        train_summed_masks_dir=train_summed_masks_dir,
        test_summed_masks_dir=test_summed_masks_dir,
    )

    (
        sampled_full_train,
        sampled_mask_train,
        sampled_full_test,
        sampled_mask_test,
    ) = getTrainTestSplit(
        extension=extension,
        multi_tumour_identifier=multi_tumour_identifier,
        all_full_dir=all_full_dir,
        all_mask_dir=all_mask_dir,
        test_split=test_split,
    )
    print(len(multi_tumour_identifier))
    print(len(sampled_full_train), len(sampled_mask_train))
    print(len(sampled_full_test), len(sampled_mask_test))
    copyFile(
        sampled_full_train=sampled_full_train,
        sampled_mask_train=sampled_mask_train,
        sampled_full_test=sampled_full_test,
        sampled_mask_test=sampled_mask_test,
        extension=extension,
        all_full_dir=all_full_dir,
        all_mask_dir=all_mask_dir,
        target_train_full_dir=target_train_full_dir,
        target_train_mask_dir=target_train_mask_dir,
        target_test_full_dir=target_test_full_dir,
        target_test_mask_dir=target_test_mask_dir,
    )

    print()
    print("Getting out of redoTrainTestSplit.")
    print("-" * 30)
    print()

    return

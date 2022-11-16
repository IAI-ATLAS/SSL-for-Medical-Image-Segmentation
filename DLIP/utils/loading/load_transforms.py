import logging

from DLIP.utils.loading.append_to_transforms import append_to_transforms
from DLIP.utils.loading.dict_to_config import dict_to_config
from DLIP.utils.loading.split_parameters import split_parameters
from DLIP.utils.dlip_globals import DATA_MODULE


def load_transforms(data_params: dict):
    img_processing_args = split_parameters(
                                dict_to_config(data_params),
                                ["img_processing"])["img_processing"]

    transformations = split_parameters(dict_to_config(img_processing_args))
    general_configuration = transformations['other']
    del transformations['other']

    # extract additional validation time transforms
    val_augs = None
    if 'validation_aug' in transformations:
        val_augs = transformations['validation_aug']
        del transformations['validation_aug']

    train_transforms = []
    val_transforms = []
    test_transforms = []
    if len(transformations.values()) > 0:
        for img_processing in transformations.values():
            append_to_transforms(
                                dict(general_configuration, **img_processing),
                                train_transforms,
                                val_transforms,
                                test_transforms,
                                val_augs)
    else:
        append_to_transforms(
                            general_configuration,
                            train_transforms,
                            val_transforms,
                            test_transforms,
                            val_augs)

    return train_transforms, val_transforms, test_transforms
DATASET_SIZES = {

    # MC
    ("crpe", "relation"): 7576,

    ("vsr", "zeroshot_test"): 1222,

    ("sugarcrepe", 2812) # 2812/2 = 1406 if you count based on images

    ("mmbench", "dev"): 4239,
    ("mmbench", "test"): 6666,

    ("seedbench", "test_image"): 14233,

    # VQA
    ("gqa", "testdev"): 12578,

    # Region based
    ("vcr", "val_qa"): 26534,
    ("vcr", "val_qar"): 26534,
}

def get_dataset_size(name, split):
    return DATASET_SIZES[(name, split)]

import sys
sys.path.append("../")

from Dataset.data_generation import *

def generate_multi_res_from_same_set():
    base_res       = 256
    base_data_root = "/data/tavonputl/data/MIT-BIH"

    create_raw_dataset(
        data_path   = f"{base_data_root}/mit-bih-arrhythmia-database-1.0.0",
        output_path = f"{base_data_root}/Datasets/Resolution-{base_res}/signal_full.hdf5",
        window_size = base_res,
        full        = True,
    )
    create_shuffled_dataset(
        dataset_path = f"{base_data_root}/Datasets/Resolution-{base_res}/signal_full.hdf5",
        output_path  = f"{base_data_root}/Datasets/Resolution-{base_res}/signal_full_shuffled.hdf5",
        data_key     = "segments",
        batch_size   = 5000,
    )

    for res in [base_res, 224, 192, 160, 128]:
        if res != base_res:
            create_preprocessed_dataset(
                dataset_path = f"{base_data_root}/Datasets/Resolution-{base_res}/signal_full_shuffled.hdf5",
                output_path  = f"{base_data_root}/Datasets/Resolution-{res}/signal_full.hdf5",
                data_key     = "segments",
                data_type    = "signal",
                method       = "crop",
                new_size     = res,
                batch_size   = 5000,
            )

        create_train_test_from_dataset(
            dataset_path = f"{base_data_root}/Datasets/Resolution-{res}/signal_full.hdf5",
            output_name  = "signal_full",
            ratio        = 0.8,
            data_key     = "segments",
            shuffle      = False,
            batch_size   = 5000,
        )
        for split in ["train", "test"]:
            create_image_dataset_contiguous(
                signal_dataset_path = f"{base_data_root}/Datasets/Resolution-{res}/signal_full_{split}.hdf5",
                output_path         = f"{base_data_root}/Datasets/Resolution-{res}/image_full_{split}.hdf5",
                batch_size          = 1000,
            )
        

#===========================================================================================================================
# Main 
#
def main():
    generate_multi_res_from_same_set()
    return


if __name__ == "__main__":
    main()
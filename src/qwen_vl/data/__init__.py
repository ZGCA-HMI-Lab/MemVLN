import re


TRAIN_R2R_RXR_RELATIVE_POSE = {
    "annotation_path": "train_r2r_rxr_relative_pose_shrunk.tsv",
    "data_path": "",
    "tag": "train_r2r_rxr_relative_pose",
}


data_dict = {
    "train_r2r_rxr_relative_pose": TRAIN_R2R_RXR_RELATIVE_POSE,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name not in data_dict:
            raise ValueError(f"do not find {dataset_name}")
        config = data_dict[dataset_name].copy()
        config["sampling_rate"] = sampling_rate
        config["dataset_name"] = dataset_name
        config_list.append(config)
    return config_list


if __name__ == "__main__":
    print(data_list(["train_r2r_rxr_relative_pose"]))

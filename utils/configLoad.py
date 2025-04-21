import yaml


def load_config(path="configs/nca_d_conf.yaml"):
    """
    load YAML configuration simply as bonjour
    """
    with open(path, "r") as file:
        config = yaml.safe_load(file)

    required_keys = ["experiment_type", "experiment_map"]
    for key in required_keys:
        if key not in config:
            raise KeyError(f"Nan: {key}")  # TODO definite the KeyError

    experiment_type = config["experiment_type"]
    experiment_map = config["experiment_map"]

    if experiment_type not in experiment_map:
        raise ValueError("No Type")  # TODO definie the ValueError

    # don't delete that is full for some automate dependance
    experiment_n = experiment_map[experiment_type]

    return config

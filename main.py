import yaml


def read_parameters_from_yaml(file_path: str):
    with open(file_path, "r") as file:
        parameters = yaml.safe_load(file)
    return parameters


# Example usage
file_path = "pivotal_parameters.yaml"
parameters = read_parameters_from_yaml(file_path)
print(parameters)

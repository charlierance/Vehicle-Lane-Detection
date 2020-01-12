import yaml


class GeneralUtils:
    def __init__(self):
        self.yaml_dict = None

    def read_yaml(self, yaml_file: str) -> dict:
        """
        :param yaml_file: This is the path to the yaml file that needs to be read
        :return: Return a dictionary of the yaml file
        """
        stream = open(yaml_file, "r")
        self.yaml_dict = yaml.safe_load(stream)
        return self.yaml_dict

    def write_yaml(self, yaml_file: str, values_dict: dict) -> None:
        with open(yaml_file, "w") as open_file:
            params = yaml.dump(values_dict, open_file)

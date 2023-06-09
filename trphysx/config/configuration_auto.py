"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import os
import json
import logging
from collections import OrderedDict
from typing import Dict
from .configuration_cylinder import CylinderConfig
from .configuration_phys import PhysConfig

CONFIG_MAPPING = OrderedDict(
    [
        ("cylinder", CylinderConfig),
    ]
)
CONFIG_NAME = "config_trphysx.json"

logger = logging.getLogger(__name__)

class AutoPhysConfig:
    """Helper class for creating configurations for different built in examples

    Raises:
        EnvironmentError: If direct initialization of this class is attempted.
    """
    def __init__(self):
        raise EnvironmentError(
            "AutoPhysConfig should not be initiated directly. The class methods should be used instead."
        )

    @classmethod
    def load_config(cls, model_name_or_path, **kwargs) -> PhysConfig:
        """Creates a configuration object for a transformer model.
        Predefined configs currently support: "cylinder"

        Args:
            model_name_or_path (str): Name of model or path to save config JSON file

        Returns:
            (PhysConfig): Configuration of transformer
        """
        # Check if file is a folder path
        # print("model_name_or_path ", model_name_or_path) # cylinder
        if os.path.isdir(model_name_or_path):
            # print("1 ")
            config_file = os.path.join(model_name_or_path, CONFIG_NAME)
            config_dict = cls.from_json_file(config_file)
        elif os.path.isfile(model_name_or_path):
            # print("2 ")
            config_file = model_name_or_path
            config_dict = cls.from_json_file(config_file)
        else:
            # print("2.5 ") # go here!
            config_dict = {}

        # First check if the model name is a pre-defined config
        if(model_name_or_path in CONFIG_MAPPING.keys()):
            # print("3 ")
            config_class = CONFIG_MAPPING[model_name_or_path]
            # print("config_class ", config_class) # config_class  <class 'trphysx.config.configuration_cylinder.CylinderConfig'>
            # Init config class
            config = config_class(**kwargs)
            config.update(config_dict)
        else:
            config = PhysConfig.from_dict(config_dict, **kwargs)

        # exit()

        return config

    @classmethod
    def from_json_file(cls, json_file: str) -> Dict:
        """
        Reads a json file and loads it into a dictionary.

        Args:
            json_file (string): Path to the JSON file containing the parameters.
        Returns:
            Dict: Dictionary of parsed JSON
        """
        config_dict = cls._dict_from_json_file(json_file)
        return config_dict

    @classmethod
    def _dict_from_json_file(cls, json_file: str):
        with open(json_file, "r", encoding="utf-8") as reader:
            text = reader.read()
        return json.loads(text)
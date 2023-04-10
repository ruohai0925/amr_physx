"""
=====
Distributed by: Notre Dame SCAI Lab (MIT Liscense)
- Associated publication:
url: https://arxiv.org/abs/2010.03957
doi: 
github: https://github.com/zabaras/transformer-physx
=====
"""
import numpy as np
import os, time
import h5py
import torch
import logging
from typing import Tuple, List
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from abc import abstractmethod
from collections import OrderedDict
from typing import Dict, List, Optional
from dataclasses import dataclass

logger = logging.getLogger(__name__)

class EmbeddingDataHandler(object):
    """Base class for embedding data handlers.
    Data handlers are used to create the training and
    testing datasets.
    """
    mu = None
    std = None

    @property
    def norm_params(self) -> Tuple:
        """Get normalization parameters

        Raises:
            ValueError: If normalization parameters have not been initialized

        Returns:
            (Tuple): mean and standard deviation
        """
        if self.mu is None or self.std is None:
            raise ValueError("Normalization constants set yet!")
        return self.mu, self.std

    @abstractmethod
    def createTrainingLoader(self, *args, **kwargs):
        pass

    @abstractmethod
    def createTestingLoader(self, *args, **kwargs):
        pass


class CylinderDataHandler(EmbeddingDataHandler):
    """Built in embedding data handler for flow around a cylinder system
    """
    class CylinderDataset(Dataset):
        """Dataset for training flow around a cylinder embedding model

        Args:
            examples (List): list of training/testing example flow fields
            visc (List): list of training/testing example viscosities
        """
        # It seems ok if both types of examples and visc are torch.tensor......
        def __init__(self, examples: List, visc: List) -> None:
            """Constructor
            """
            self.examples = examples
            self.visc = visc

        def __len__(self):
            return len(self.examples)

        def __getitem__(self, i) -> Dict[str, torch.Tensor]:
            return {"states": self.examples[i], "viscosity": self.visc[i]}

    # The @dataclass decorator is used to automatically generate special methods like __init__, __repr__, and __eq__ for the class.
    # In this case, the @dataclass decorator may seem unnecessary since there are no attributes to define. 
    # However, it can still be useful to use @dataclass to automatically generate some special methods for the class, such as __repr__, __eq__, and others. 
    # It also provides a clear way of documenting the purpose of the class, as shown in the docstring of the class.
    @dataclass
    class CylinderDataCollator:
        """Data collator for flow around a cylinder embedding problem
        """
        # Default collator
        def __call__(self, examples:List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
            # Stack examples in mini-batch
            x_data_tensor =  torch.stack([example["states"] for example in examples])
            visc_tensor =  torch.stack([example["viscosity"] for example in examples])

            return {"states": x_data_tensor, "viscosity": visc_tensor}

    def createTrainingLoader(self, 
        file_path: str,
        block_size: int, # 4
        stride: int = 1, # 16
        ndata: int = -1,
        batch_size: int = 32, # 64
        shuffle: bool = True,
    ) -> DataLoader:
        """Creating training data loader for the flow around a cylinder system.
        For a single training simulation, the total time-series is sub-chunked into
        smaller blocks for training.

        Args:
            file_path (str): Path to HDF5 file with training data
            block_size (int): The length of time-series blocks
            stride (int): Stride of each time-series block
            ndata (int, optional): Number of training time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Training batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to True.

        Returns:
            (DataLoader): Training loader
        """
        logging.info('Creating training loader')
        print("file_path ", file_path)
        assert os.path.isfile(file_path), "Training HDF5 file {} not found".format(file_path)

        examples = []
        visc = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0

            for key in f.keys():
                visc0 = (2.0/float(key))
                ux = torch.Tensor(f[key+'/ux'])
                uy = torch.Tensor(f[key + '/uy'])
                p = torch.Tensor(f[key + '/p'])
                data_series = torch.stack([ux, uy, p], dim=1)

                # # Print the variables
                # print("key ", key)
                # print("visc0: ", visc0)
                # print("Size of ux: ", ux.size())
                # print("Size of uy: ", uy.size())
                # print("Size of p: ", p.size())
                # print("Size of data_series: ", data_series.size())
                # print(" ")

                # key  100
                # visc0:  0.02
                # Size of ux:  torch.Size([401, 64, 128])
                # Size of uy:  torch.Size([401, 64, 128])
                # Size of p:  torch.Size([401, 64, 128])
                # Size of data_series:  torch.Size([401, 3, 64, 128])

                # # Stride over time-series
                # print("data_series.size(0) ", data_series.size(0))
                # print("block_size ", block_size)
                # print("stride ", stride)

                # data_series.size(0)  401
                # block_size  4
                # stride  16


                for i in range(0, data_series.size(0) - block_size + 1, stride):  # Truncate in block of block_size
                    examples.append(data_series[i: i + block_size])
                    visc.append(torch.tensor([visc0]))
                #     print("i ", i, " len of examples ", len(examples), " len of visc ", len(visc), 'size of data_series[i: i + block_size] ', data_series[i: i + block_size].size())
                # print(" ")

                # if (samples == 1):
                #     exit()

                samples = samples + 1
                # print("samples ", samples)
                # print(" ")
                if (ndata > 0 and samples > ndata):  # If we have enough time-series samples break loop
                    break

            # exit()

        # print("torch.tensor(visc) ", torch.tensor(visc))
        # print("torch.tensor(visc).size ", torch.tensor(visc).size()) # torch.Size([675])

        data = torch.stack(examples, dim=0) # data.size()  torch.Size([675, 4, 3, 64, 128])
        # calculate the mean and std of u, v, p, and mu
        self.mu = torch.tensor([torch.mean(data[:,:,0]), torch.mean(data[:,:,1]), torch.mean(data[:,:,2]), torch.mean(torch.tensor(visc))]) #  torch.Size([4])
        self.std = torch.tensor([torch.std(data[:,:,0]), torch.std(data[:,:,1]), torch.std(data[:,:,2]), torch.std(torch.tensor(visc))]) #  torch.Size([4])

        # print("data.size() ", data.size(), "self.mu ", self.mu.size(), "self.std.size() ", self.std.size())
        # exit()

        # Needs to min-max normalization due to the reservoir matrix, needing to have a spectral density below 1
        if (data.size(0) < batch_size):
            logging.warn('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        # print(type(data), type(torch.stack(visc, dim=0))) # <class 'torch.Tensor'> <class 'torch.Tensor'>
        dataset = self.CylinderDataset(data, torch.stack(visc, dim=0))
        # print("torch.stack(visc, dim=0).size ", torch.stack(visc, dim=0).size()) # torch.Size([675, 1])
        # print("torch.stack(visc, dim=1).size ", torch.stack(visc, dim=1).size()) # torch.Size([1, 675])
        # exit()

        data_collator = self.CylinderDataCollator()
        training_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=True)
        return training_loader

    def createTestingLoader(self, 
            file_path: str,
            block_size: int,
            ndata: int = -1,
            batch_size: int = 32,
            shuffle: bool =False,
        ) -> DataLoader:
        """Creating testing/validation data loader for the flow around a cylinder system.
        For a data case with time-steps [0,T], this method extract a smaller
        time-series to be used for testing [0, S], s.t. S < T.

        Args:
            file_path (str): Path to HDF5 file with testing data
            block_size (int): The length of testing time-series
            ndata (int, optional): Number of testing time-series. If negative, all of the provided 
            data will be used. Defaults to -1.
            batch_size (int, optional): Testing batch size. Defaults to 32.
            shuffle (bool, optional): Turn on mini-batch shuffling in dataloader. Defaults to False.

        Returns:
            (DataLoader): Testing/validation data loader
        """
        logging.info('Creating testing loader')
        assert os.path.isfile(file_path), "Eval HDF5 file {} not found".format(file_path)

        examples = []
        visc = []
        with h5py.File(file_path, "r") as f:
            # Iterate through stored time-series
            samples = 0
            for key in f.keys():
                visc0 = (2.0/float(key))
                ux = torch.Tensor(f[key + '/ux'])
                uy = torch.Tensor(f[key + '/uy'])
                p = torch.Tensor(f[key + '/p'])
                data_series = torch.stack([ux, uy, p], dim=1)
                # Stride over time-series data_series.size(0)
                for i in range(0, data_series.size(0) - block_size + 1, block_size):  # Truncate in block of block_size
                    examples.append(data_series[i: i + block_size])
                    visc.append(torch.tensor([visc0]))
                    break

                samples = samples + 1
                if (ndata > 0 and samples >= ndata):  # If we have enough time-series samples break loop
                    break

        # Combine data-series
        data = torch.stack(examples, dim=0)
        if (data.size(0) < batch_size):
            logging.warning('Lower batch-size to {:d}'.format(data.size(0)))
            batch_size = data.size(0)

        dataset = self.CylinderDataset(data, torch.stack(visc, dim=0))
        data_collator = self.CylinderDataCollator()
        testing_loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=data_collator, drop_last=False)

        return testing_loader

LOADER_MAPPING = OrderedDict(
    [
        ("cylinder", CylinderDataHandler),
    ]
)
class AutoDataHandler():
    """Helper class for intializing different built in data-handlers for embedding training
    """
    @classmethod
    def load_data_handler(cls, model_name: str, **kwargs) -> EmbeddingDataHandler:
        """Gets built-in data handler.
        Currently supports: "lorenz", "cylinder", "grayscott"

        Args:
            model_name (str): Model name

        Raises:
            KeyError: If model_name is not a supported model type

        Returns:
            (EmbeddingDataHandler): Embedding data handler
        """
        # First check if the model name is a pre-defined config
        if (model_name in LOADER_MAPPING.keys()):
            loader_class = LOADER_MAPPING[model_name]
            # Init config class
            loader = loader_class(**kwargs)
        else:
            err_str = "Provided model name: {}, not present in built in data handlers".format(model_name)
            raise KeyError(err_str)

        return loader
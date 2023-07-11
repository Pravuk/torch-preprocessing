import unittest

import numpy as np
import torch
from torch.nn import Conv2d, Module, ReLU, MaxPool2d, Linear, LogSoftmax

from torch_pp import StandardScaler
from torch_pp.minmaxscaler import MinMaxScaler


class Cnn(Module):
    def __int__(self, num_channels: int, classes: int):
        super(self).__init__()
        self.scaler = StandardScaler()
        self.conv1 = Conv2d(in_channels=num_channels, out_channels=20, kernel_size=(5, 5))
        self.relu1 = ReLU()
        self.maxpool1 = MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.fc2 = Linear(in_features=500, out_features=classes)
        self.logSoftmax = LogSoftmax(dim=1)

    def forward(self, x):
        x = self.scaler.fit_transform(x)
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.maxpool1(x)
        x = self.fc2(x)
        output = self.logSoftmax(x)
        return output


class TestScalers(unittest.TestCase):
    def test_standard_scaler_transform(self):
        input_x = torch.from_numpy(np.array([[20., 1.], [-3., 700.], [-11., 3.]])).to(dtype=torch.double)
        scaler = StandardScaler()
        transformed_x = scaler.fit_transform(input_x)
        transformed_back_x = scaler.inverse_transform(transformed_x)
        torch.testing.assert_close(input_x, transformed_back_x)

    def test_minmax_scaler_transform(self):
        input_x = torch.from_numpy(np.array([[20., 1.], [-3., 700.], [-11., 3.]])).to(dtype=torch.float64)
        scaler = MinMaxScaler()
        transformed_x = scaler.fit_transform(input_x)
        transformed_back_x = scaler.inverse_transform(transformed_x)
        torch.testing.assert_close(input_x, transformed_back_x)


if __name__ == '__main__':
    unittest.main()

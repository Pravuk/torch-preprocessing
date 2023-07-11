# torch-pp
## Data preprocessing for ML solutions 
Port of scikit-learn's scalers with pytorch tensor instead of numpy

### Installing
```sh
pip install torch-pp
```
or locally

```sh
git clone https://github.com/Pravuk/torch-preprocessing.git
cd torch-preprocessing
pip install -e .
```

### Scaler ported at this moment
- *StandardScaler*
- *MinMaxScaler*


### Examples

```python
import torch
from torch.nn import Conv2d, Module, ReLU, MaxPool2d, Linear, LogSoftmax
from torch_pp import StandardScaler

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
```
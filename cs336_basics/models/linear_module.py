import torch.nn as nn
import torch
import numpy as np

class LinearClass(nn.Module):

    def __init__(
        self, 
        in_features: int, 
        out_features: int, 
        device: torch.device | None = None, 
        dtype: torch.device | None = None
    ):
        super().__init__()

        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        std = np.sqrt( 2 / (in_features + out_features) )
        left_bound, right_bound = -3 * std, 3 * std

        # Fill with values strictly between -0.04 and 0.04
        nn.init.trunc_normal_(self.weight, mean=0.0, std=std, a=left_bound, b=right_bound)
    
    def forward(
        self, 
        x: torch.Tensor
    ) -> torch.Tensor:
        return x @ self.weight.t()
        

if __name__ == '__main__':
    layer1 = LinearClass(in_features=20, out_features=5)

    x = torch.randn(3, 20)

    output = layer1(x)
    print(output.shape) 
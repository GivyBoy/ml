import torch
from torch import nn
from funcs import FUNCS

from dataclasses import dataclass, field


def default_field(obj):
    """
    Helper used to allow a dictionary to be a variable in the dataclass instance
    """
    return field(default_factory=lambda: obj) if isinstance(obj, dict) else obj


@dataclass
class NetConfig:
    sequence: str = ""
    structure: dict[str:list] = default_field({})
    structure_depth: dict[str:int] = default_field({key: len(val) for key, val in structure.default_factory().items()})


"""
Potential Update:

1) allow for recursive declaration of layers, so we can generate more complicated models
    e.g., a network within a network or branches within a network

"""


class GenerateNet(nn.Module):
    def __init__(self, config: NetConfig) -> None:
        super().__init__()
        self.config = config

        if isinstance(self.config.structure, dict):
            self.structure = self.config.structure
            self.structure_depth = self.config.structure_depth
            self.sequence = self.config.sequence.split("-")
        else:
            raise ValueError(f"structure must be a dict, got {type(self.config.structure)} instead")

        self.conv_block, self.fc_layer = self._get_sequence()

    def forward(self, x: torch.tensor) -> torch.tensor:
        return self.fc_layer(self.conv_block(x).flatten())

    def _get_sequence(self) -> nn.Sequential:
        func_list = []
        fc_layer = []
        for val in self.sequence:
            if val in FUNCS:
                func_val = self.structure[val][0] if len(self.structure[val]) > 0 else None
                if func_val:
                    self.structure[val].pop(0)
                    if val == "fc":
                        fc_layer.append(FUNCS[val](*func_val))
                    else:
                        func_list.append(FUNCS[val](*func_val))
                else:
                    func_list.append(FUNCS[val]())
            else:
                raise ValueError(f"function {val} not recognized")
        return nn.Sequential(*func_list), nn.Sequential(*fc_layer)


if __name__ == "__main__":

    lenet_config = NetConfig(
        sequence="conv-relu-avg_pool-conv-relu-avg_pool-conv-relu-fc-fc",
        structure={
            "conv": [(1, 6, 5, 1, 0), (6, 16, 5, 1, 0), (16, 120, 5, 1, 0)],
            "relu": [],
            "avg_pool": [(2, 2), (2, 2)],
            "fc": [(120, 84), (84, 10)],
        },
    )
    mnist = torch.randn(1, 1, 32, 32)
    lenet_block = GenerateNet(lenet_config)
    lenet_block_out = lenet_block(mnist)
    print(lenet_block_out.shape)

    fc_config = NetConfig(
        sequence="fc-fc-fc",
        structure={
            "fc": [(120, 84), (84, 42), (42, 10)],
        },
    )
    fc_block = GenerateNet(fc_config)
    vals = torch.randn(1, 1, 120)
    fc_block_out = fc_block(vals)
    print(fc_block_out.shape)

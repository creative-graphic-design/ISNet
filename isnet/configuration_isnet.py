from transformers import PretrainedConfig


class ISNetConfig(PretrainedConfig):
    def __init__(self, in_channels: int = 3, out_channels: int = 1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels

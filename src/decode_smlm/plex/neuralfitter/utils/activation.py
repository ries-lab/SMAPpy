import torch


class ChanneledActivation(torch.nn.Module):
    def __init__(
        self, activations: torch.nn.ModuleList, dim: int = 4, validate: bool = True
    ):
        """
        Make channel-specific activations.

        Args:
            activations: list of activations, must have same length as input channels
            dim: required dim of input tensor, to make sure that batch / channel dim is not
             confused
            validate: validate on forward pass
        """
        super().__init__()
        self._activations = activations
        self._dim = dim
        self._validate = validate

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._validate_fn(x) if self._validate else None

        for i, activation in enumerate(self._activations):
            x[:, i] = activation.forward(x[:, i])

        return x

    def _validate_fn(self, x: torch.Tensor):
        if x.dim() != self._dim:
            raise ValueError(
                f"Input tensor must have {self._dim} dimensions, but has {x.dim()}."
            )
        if x.shape[1] != len(self._activations):
            raise ValueError(
                f"Input tensor must have {len(self._activations)} channels, "
                f"but has shape {x.shape}."
            )

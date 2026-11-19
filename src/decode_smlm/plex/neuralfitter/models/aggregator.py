import torch

from .. import spec


class AggregateModel(torch.nn.Module):
    def __init__(
        self,
        models_shared: torch.nn.ModuleList,
        model_union: torch.nn.Module,
        model_heads: torch.nn.ModuleList,
        model_output: torch.nn.Module,
        map_in: spec.ModelGMMMapInput,
        kaiming_normal: bool = False,
    ):
        super().__init__()

        self._models_shared = models_shared
        self._model_union = model_union
        self._model_heads = model_heads
        self._model_output = model_output
        self._map_in = map_in

        if kaiming_normal:
            self.apply(self.weight_init)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # forward through shared models
        out = [None] * (len(self._map_in.stage_shared))
        for i, (m, ix_ch) in enumerate(
            zip(self._models_shared, self._map_in.stage_shared, strict=True)
        ):
            out_ch = [None] * len(ix_ch)
            for j, ix_win in enumerate(ix_ch):
                if isinstance(ix_win, int):
                    ix_win = [ix_win]
                x_win = x[:, ix_win]
                out_ch[j] = m.forward(x_win)
            out[i] = torch.cat(out_ch, dim=1)

        # prepare intermediate output
        out = torch.cat(out, dim=1)
        x_mixin = x[:, self._map_in.stage_union]
        out = torch.cat((out, x_mixin), dim=1)

        # forward through union model
        out = self._model_union.forward(out)

        # forward through heads
        out = [m.forward(out) for m in self._model_heads]
        out = torch.cat(out, dim=1)

        # forward through final output layers
        out = self._model_output.forward(out)

        return out

    @staticmethod
    def weight_init(m):
        """
        Apply Kaiming normal init. Call this recursively by model.apply(model.weight_init)

        Args:
            m: model

        """
        if isinstance(m, torch.nn.Conv2d):
            torch.nn.init.kaiming_normal_(m.weight, mode="fan_in", nonlinearity="relu")

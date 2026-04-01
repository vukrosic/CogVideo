import torch
import torch.nn as nn
from packaging import version

OPENAIUNETWRAPPER = "sgm.modules.diffusionmodules.wrappers.OpenAIWrapper"

# OPTIMIZATION: Cache empty tensor to avoid recreation on every forward pass
_EMPTY_TENSOR = torch.Tensor()


class IdentityWrapper(nn.Module):
    def __init__(
        self, diffusion_model, compile_model: bool = False, dtype: torch.dtype = torch.float32
    ):
        super().__init__()
        compile = (
            torch.compile
            if (version.parse(torch.__version__) >= version.parse("2.0.0")) and compile_model
            else lambda x: x
        )
        self.diffusion_model = compile(diffusion_model)
        self.dtype = dtype

    def forward(self, *args, **kwargs):
        return self.diffusion_model(*args, **kwargs)


class OpenAIWrapper(IdentityWrapper):
    def forward(self, x: torch.Tensor, t: torch.Tensor, c: dict, **kwargs) -> torch.Tensor:
        # OPTIMIZATION: Only convert dtype if actually needed (avoid redundant tensor creation)
        for key in c:
            if c[key].dtype != self.dtype:
                c[key] = c[key].to(self.dtype)

        # OPTIMIZATION: Use cached empty tensor instead of creating new one each call
        concat_tensor = c.get("concat", _EMPTY_TENSOR.type_as(x))
        if x.dim() == 4:
            x = torch.cat((x, concat_tensor), dim=1)
        elif x.dim() == 5:
            x = torch.cat((x, concat_tensor), dim=2)
        else:
            raise ValueError("Input tensor must be 4D or 5D")

        return self.diffusion_model(
            x,
            timesteps=t,
            context=c.get("crossattn", None),
            y=c.get("vector", None),
            **kwargs,
        )

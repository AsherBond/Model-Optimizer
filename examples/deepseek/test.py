import pdb

import safetensors
import torch
from int4_kernel import int4_dequant

tensors = safetensors.safe_open("model-00001-of-00527.safetensors", framework="pt", device="cuda")

bf16 = tensors.get_tensor("model.layers.1.mlp.experts.0.down_proj.weight")
int32 = tensors.get_tensor("model.layers.1.mlp.experts.0.down_proj.weight_packed")
ws = tensors.get_tensor("model.layers.1.mlp.experts.0.down_proj.weight_scale")
torch.set_default_dtype(torch.bfloat16)
bf16_2 = int4_dequant(int32, ws, block_size=32)


if not torch.allclose(bf16_2, bf16, rtol=1e-4, atol=1e-4):
    pdb.set_trace()

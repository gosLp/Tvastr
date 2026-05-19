import sys
from pathlib import Path

import torch

generated_dir = Path("generated/matmul_bias_relu").resolve()
sys.path.insert(0, str(generated_dir))

from layout_guards import verify_layouts


def main():
    A_base = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    A = A_base.t()  # non-contiguous view

    B = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    bias = torch.randn((256,), device="cuda", dtype=torch.float16)

    print("A shape:", A.shape)
    print("A stride:", A.stride())

    verify_layouts(A, B, bias)


if __name__ == "__main__":
    main()
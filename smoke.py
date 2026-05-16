import sys
from pathlib import Path

import torch

generated_dir = Path("generated/matmul_bias_relu").resolve()
sys.path.insert(0, str(generated_dir))

from verifier import verify
from reference import reference


def main():
    A = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    B = torch.randn((256, 256), device="cuda", dtype=torch.float16)
    bias = torch.randn((256,), device="cuda", dtype=torch.float16)

    dims = verify(A, B, bias)
    C = reference(A, B, bias)

    print("[ok] verified:", dims)
    print("[ok] output shape:", tuple(C.shape))
    print("[ok] output dtype:", C.dtype)


if __name__ == "__main__":
    main()
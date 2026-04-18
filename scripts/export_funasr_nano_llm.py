import argparse
import os

import torch
from safetensors.torch import save_file


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export FunASR-nano Qwen LLM weights to safetensors.",
    )
    parser.add_argument(
        "--model-file",
        default=".download/Fun-ASR-Nano-2512/model.pt",
        help="Path to the FunASR-nano checkpoint file.",
    )
    parser.add_argument(
        "--out",
        default=".artifacts/local/funasr_nano_llm_fp16.safetensors",
        help="Output safetensors file.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    model_file = os.path.abspath(args.model_file)
    out_path = os.path.abspath(args.out)
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    obj = torch.load(model_file, map_location="cpu")
    state_dict = obj["state_dict"] if "state_dict" in obj else obj

    llm_tensors = {}
    for key, value in state_dict.items():
        if not key.startswith("llm.") or not isinstance(value, torch.Tensor):
            continue
        if key == "llm.lm_head.weight":
            continue
        llm_tensors[key[4:]] = value.to(torch.float16).contiguous()

    if not llm_tensors:
        raise RuntimeError("No llm.* tensors found in checkpoint")

    save_file(llm_tensors, out_path, metadata={"source": model_file, "format": "funasr-nano-llm"})
    total_bytes = sum(t.numel() * t.element_size() for t in llm_tensors.values())
    print(
        {
            "out": out_path,
            "tensor_count": len(llm_tensors),
            "total_bytes": total_bytes,
        }
    )


if __name__ == "__main__":
    main()

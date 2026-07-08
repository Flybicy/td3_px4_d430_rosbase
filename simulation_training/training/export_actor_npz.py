import argparse
import os

import numpy as np
import torch


def export_actor_npz(state_dict_path: str, output_path: str):
    state = torch.load(state_dict_path, map_location="cpu")
    if not isinstance(state, dict):
        raise ValueError("Expected a PyTorch state_dict.")

    required = [
        "net.0.weight",
        "net.0.bias",
        "net.2.weight",
        "net.2.bias",
        "net.4.weight",
        "net.4.bias",
    ]
    missing = [key for key in required if key not in state]
    if missing:
        raise ValueError("Missing state_dict keys: {}".format(", ".join(missing)))

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    np.savez(
        output_path,
        w0=state["net.0.weight"].detach().cpu().numpy(),
        b0=state["net.0.bias"].detach().cpu().numpy(),
        w1=state["net.2.weight"].detach().cpu().numpy(),
        b1=state["net.2.bias"].detach().cpu().numpy(),
        w2=state["net.4.weight"].detach().cpu().numpy(),
        b2=state["net.4.bias"].detach().cpu().numpy(),
        max_speed=np.asarray(1.5, dtype=np.float32),
    )


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dict", required=True)
    parser.add_argument("--output", required=True)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    export_actor_npz(args.state_dict, args.output)
    print("Saved NumPy actor to: {}".format(args.output))

from __future__ import annotations

import argparse


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Token extraction scaffold")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--image-folder", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--token-layer", type=int, default=-1)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    print("Token extraction is reserved for the next implementation step.")
    print(f"checkpoint={args.checkpoint}")
    print(f"image_folder={args.image_folder}")
    print(f"output={args.output}")
    print(f"token_layer={args.token_layer}")


if __name__ == "__main__":
    main()


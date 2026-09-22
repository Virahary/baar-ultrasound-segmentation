"""Command-line entry points for the two-stage BAAR workflow."""
import argparse

from .backbones import BACKBONES
from .engine import evaluate, predict, train


def main(argv=None):
    parser = argparse.ArgumentParser(
        prog="baar", description="Boundary-Aware Attention Refinement for ultrasound segmentation"
    )
    commands = parser.add_subparsers(dest="command", required=True)
    training = commands.add_parser("train", help="Train a baseline or BAAR refinement")
    training.add_argument("--stage", choices=("baseline", "baar"), required=True)
    training.add_argument("--data-root", required=True, help="Root containing train/ and val/")
    training.add_argument("--output-dir", required=True)
    training.add_argument("--backbone", choices=BACKBONES, default=None)
    training.add_argument("--baseline", help="Baseline checkpoint for refinement training")
    training.add_argument("--config", help="JSON with model and loss settings; defaults match the paper")
    training.add_argument("--image-size", type=int, default=None, help="Default: 256; inherited for BAAR")
    training.add_argument("--epochs", type=int, default=None, help="Default: 200 baseline, 20 BAAR")
    training.add_argument("--seed", type=int, default=3407)
    training.add_argument("--ramp-epochs", type=int, default=6, help="BAAR half-cosine ramp; 0 = immediate")
    training.set_defaults(func=train)

    evaluation = commands.add_parser("evaluate", help="Evaluate a selected checkpoint")
    evaluation.add_argument("--checkpoint", required=True)
    evaluation.add_argument("--data-root", required=True)
    evaluation.add_argument("--split", choices=("val", "test"), default="test")
    evaluation.add_argument("--output", required=True, help="Destination for aggregate metrics JSON")
    evaluation.set_defaults(func=evaluate)

    prediction = commands.add_parser("predict", help="Save binary segmentation masks")
    prediction.add_argument("--checkpoint", required=True)
    prediction.add_argument("--input-dir", required=True)
    prediction.add_argument("--output-dir", required=True)
    prediction.set_defaults(func=predict)

    for command in (training, evaluation, prediction):
        command.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    for command in (training, evaluation):
        command.add_argument("--batch-size", type=int, default=8)
        command.add_argument("--workers", type=int, default=0, help="DataLoader worker count")
    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()

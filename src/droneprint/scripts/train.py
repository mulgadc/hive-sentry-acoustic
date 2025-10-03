import argparse
import os

from droneprint.training import (
    train_binary_classifier,
    train_make_classifier,
    train_model_classifiers,
    train_openmax_closedset,
)


def main():
    parser = argparse.ArgumentParser(description="Train DronePrint cascaded classifiers")
    parser.add_argument("--datasets_root", type=str, required=True, help="Path to datasets root directory")
    parser.add_argument("--outputs", type=str, default="outputs/models", help="Directory to save trained models")
    parser.add_argument("--epochs", type=int, default=None, help="Override epochs")
    parser.add_argument("--batch_size", type=int, default=None, help="Override batch size")
    parser.add_argument("--lr", type=float, default=None, help="Override learning rate")
    parser.add_argument(
        "--augment_splits",
        type=str,
        nargs="*",
        default=["training"],
        choices=["training", "validation"],
        help="Which splits to augment with frequency warping",
    )
    parser.add_argument("--noise_mix_train", type=float, default=0.0, help="Probability to mix DS2 noise into DS1 training samples for Classifier X (0-1)")
    parser.add_argument("--noise_mix_val", type=float, default=0.0, help="Probability to mix DS2 noise into DS1 validation samples for Classifier X (0-1)")
    parser.add_argument("--train_openmax", action="store_true", help="Also train OpenMax-style closed-set detector over all drone models")
    args = parser.parse_args()

    os.makedirs(args.outputs, exist_ok=True)

    print("Training Classifier X (binary)...")
    train_binary_classifier(
        datasets_root=args.datasets_root,
        output_dir=args.outputs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        augment_splits=tuple(args.augment_splits),
        noise_mix_pct_train=args.noise_mix_train,
        noise_mix_pct_val=args.noise_mix_val,
    )

    print("Training Classifier Y (make)...")
    train_make_classifier(
        datasets_root=args.datasets_root,
        output_dir=args.outputs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        augment_splits=tuple(args.augment_splits),
    )

    print("Training Classifier Z (per-make models)...")
    train_model_classifiers(
        datasets_root=args.datasets_root,
        output_dir=args.outputs,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        augment_splits=tuple(args.augment_splits),
    )

    if args.train_openmax:
        print("Training OpenMax-style closed-set model (class-wise thresholds)...")
        train_openmax_closedset(
            datasets_root=args.datasets_root,
            output_dir=args.outputs,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            augment_splits=tuple(args.augment_splits),
        )

    print("Done. Models saved to:", args.outputs)


if __name__ == "__main__":
    main()

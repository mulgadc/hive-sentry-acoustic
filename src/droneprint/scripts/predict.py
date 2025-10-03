import argparse

from droneprint.system import DronePrintSystem


def main():
    parser = argparse.ArgumentParser(description="Predict with DronePrint cascaded classifiers")
    parser.add_argument("--audio", type=str, required=True, help="Path to audio file to classify")
    parser.add_argument("--models_dir", type=str, required=True, help="Directory containing saved models")
    parser.add_argument("--channel", type=int, default=None, help="Optional 0-based channel index to use from a multichannel WAV (default: mono mix)")
    parser.add_argument("--detector-mode", type=str, default="binary", choices=["binary", "openmax"], help="Detector mode: binary or openmax (default: binary)")
    args = parser.parse_args()

    system = DronePrintSystem()
    system.load_from_dir(args.models_dir)
    if args.channel is not None:
        result = system.predict_single_channel(args.audio, channel=args.channel, detector_mode=args.detector_mode)
    else:
        result = system.predict(args.audio, detector_mode=args.detector_mode)
    print(f"Prediction: {result}")


if __name__ == "__main__":
    main()

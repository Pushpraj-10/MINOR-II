"""
Depression Detection ML Service — Entry Point

Quick-start commands:
    python main.py train --arch mel_cnn          Train a single model
    python main.py train --arch bilstm --epochs 50
    python main.py train-all                     Train all architectures
    python main.py evaluate                      Evaluate all TFLite models
    python main.py list                          List available architectures
"""

import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    command = sys.argv[1]
    # Forward remaining args
    sys.argv = [sys.argv[0]] + sys.argv[2:]

    if command == "train":
        from scripts.train_model import main as train_main
        train_main()
    elif command == "train-all":
        from scripts.train_all import main as train_all_main
        train_all_main()
    elif command == "evaluate":
        from scripts.evaluate import main as eval_main
        eval_main()
    elif command == "list":
        from src.models.architectures import MODEL_REGISTRY
        print("Available architectures:")
        for name, info in MODEL_REGISTRY.items():
            print(f"  {name:20s}  {info['description']}")
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

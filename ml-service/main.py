"""
Depression Detection ML Service — Entry Point

Quick-start commands:
    python main.py train --arch mel_cnn          Train on RAVDESS (acted emotion)
    python main.py train-eatd --arch mel_cnn     Train on EATD-Corpus (real depression)    python main.py train-processed               Train on processed multi-feature data
    python main.py train-recurrent               Train GRU/BiLSTM on processed data
    python main.py train-recurrent --cell gru    Train GRU variant
    python main.py export-recurrent --cell gru   Export trained recurrent model to TFLite
    python main.py test-recurrent                Test GRU/BiLSTM TFLite models on EATD
    python main.py test-recurrent --sweep        Find best threshold then evaluate
    python main.py train-all                     Train all architectures (RAVDESS)
    python main.py evaluate                      Evaluate all TFLite models
    python main.py evaluate-subjects             Subject-level evaluation (aggregated)
    python main.py evaluate-subjects --sweep     Sweep thresholds to find best
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
    elif command == "train-eatd":
        from scripts.train_eatd import main as train_eatd_main
        train_eatd_main()
    elif command == "train-processed":
        from scripts.train_processed import main as train_processed_main
        train_processed_main()
    elif command == "train-recurrent":
        from scripts.train_recurrent import main as train_recurrent_main
        train_recurrent_main()
    elif command == "export-recurrent":
        from scripts.export_recurrent import main as export_recurrent_main
        export_recurrent_main()
    elif command == "test-recurrent":
        from scripts.test_recurrent_tflite import main as test_recurrent_main
        test_recurrent_main()
    elif command == "train-all":
        from scripts.train_all import main as train_all_main
        train_all_main()
    elif command == "evaluate":
        from scripts.evaluate import main as eval_main
        eval_main()
    elif command == "evaluate-subjects":
        from scripts.evaluate_subject_level import main as eval_subj_main
        eval_subj_main()
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

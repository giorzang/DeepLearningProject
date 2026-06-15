#!/usr/bin/env python3
"""
Master run script for the Avito Demand Prediction pipeline.

Usage:
    python run.py preprocess   # Step 1: Extract features (run once)
    python run.py train        # Step 2: Train the model
    python run.py predict      # Step 3: Generate submission
    python run.py all          # Run all steps sequentially
"""

import sys
import os

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1].lower()

    if command == "preprocess":
        from preprocess import run_preprocessing
        run_preprocessing()

    elif command == "train":
        from train import run_training
        run_training()

    elif command == "predict":
        from predict import generate_predictions
        generate_predictions()

    elif command == "all":
        from preprocess import run_preprocessing
        from train import run_training
        from predict import generate_predictions

        run_preprocessing()
        run_training()
        generate_predictions()

    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()

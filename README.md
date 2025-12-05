# ML Coursework

Short repository README for a machine learning coursework project.

## Summary
A small, reproducible ML project for coursework. Implements data preprocessing, model training, evaluation, and basic inference. Designed to be clear, modular, and easy to run locally.

## Requirements
- Python 3.8+
- Recommended: create a virtual environment
- Install dependencies:
```
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS / Linux
source .venv/bin/activate

pip install -r requirements.txt
```

## Project structure
- data/
    - raw/           ← raw input files (not tracked)
    - processed/     ← cleaned / split data
- notebooks/       ← exploratory analysis and demos
- src/             ← source code: preprocessing, models, utils
- scripts/         ← CLI scripts: train.py, evaluate.py, infer.py
- experiments/     ← checkpoints and logs
- requirements.txt
- README.md

## Data
Place raw dataset files in data/raw/. Expected format: CSV with a header; include README in data/raw describing columns. A preprocessing step (src/preprocess.py) converts raw → data/processed.

## Usage
Train:
```
python scripts/train.py --config configs/train.yaml --output experiments/run1
```
Evaluate:
```
python scripts/evaluate.py --checkpoint experiments/run1/checkpoint.pt --data data/processed/test.csv
```
Infer:
```
python scripts/infer.py --checkpoint experiments/run1/checkpoint.pt --input path/to/input.csv --output predictions.csv
```

## Configuration & reproducibility
- Use config files in configs/ to control hyperparameters.
- Set random seeds in configs for reproducible runs.
- Save a copy of the config beside each experiment output.

## Contributing
- Follow simple, documented changes.
- Add tests for new functionality (tests/).
- Open an issue or PR with a clear description.

## License
Add a LICENSE file to specify terms (e.g., MIT).

For questions, run the relevant notebook in notebooks/ or inspect scripts/ for usage examples.
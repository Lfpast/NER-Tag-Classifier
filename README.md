# NER Tag Classifier

This repository contains a Named Entity Recognition (NER) Tag Classification project implemented with PyTorch and Hugging Face Transformers.

## Structure

- `code/` - Python scripts for training and prediction.
  - `ner_for_report.py` - Training script and model definition (train & evaluation).
  - `predict_for_report.py` - Prediction script (supports CLI args; writes `submission.csv`).
  - `model_v3.py` - `EnhancedModel` and configuration used by the project.
  - `exp_configs.py` - Experiment configuration dataclass.
- `data/` - dataset (train/test CSVs), default test sample at `data/test.csv`.
- `requirements.txt` - project dependencies.

## Requirements

Please create a Python environment and install dependencies. You must also install a suitable `torch` build for your platform and (optional) GPU support.

Alternatively, you can use the provided `setup.sh` script to create a conda environment with recommended dependencies.

Example (via conda script):

```bash
# From repo root
./setup.sh ner_env 3.9
conda activate ner_env
```

Example (via conda script with GPU/CUDA 12.6):

```bash
# From repo root (GPU + CUDA 12.6)
./setup.sh ner_env 3.9 gpu 12.6
conda activate ner_env
```

### Dependencies (summary)

- Conda core dependencies (installed by `setup.sh`):
  - python (>=3.8)
  - pandas
  - numpy
  - scikit-learn
  - nltk
  - tqdm
  - pip
  - pytorch (CPU or CUDA variant via conda)

- Pip packages (installed from `requirements.txt`):
  - transformers==4.49.0
  - datasets
  - evaluate
  - safetensors
  - tqdm
  - scikit-learn

Note: `torch` is installed using conda/pytorch channel to ensure the correct CUDA build if needed.

GPU notes:
- If you select GPU mode in `setup.sh`, the script will try to install the matching `pytorch-cuda` package for the specified CUDA version, e.g. `pytorch-cuda=12.6` for CUDA 12.6.
- Ensure that your system has the appropriate NVIDIA drivers compatible with CUDA 12.6 installed (for an RTX 4060, using recent NVIDIA drivers should be fine).
- The conda command used is: `conda install -c pytorch -c nvidia pytorch pytorch-cuda=12.6 torchvision torchaudio`.
- If the GPU install fails, the script falls back to a CPU-only PyTorch install.
```

Example (CPU):

```bash
python -m venv .venv
source .venv/Scripts/activate   # Windows PowerShell: .venv\Scripts\Activate
pip install -U pip
pip install -r requirements.txt
# Install torch separately (choose correct wheel for system):
# CPU-only (example):
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

If you have a GPU, follow the official PyTorch install instructions to get an appropriate build.

## Prediction / Inference

A simple bash script `run_predict.sh` is provided to run the prediction and generate a submission CSV.

Usage:

```bash
# From repo root
# ./run_predict.sh [MODEL_DIR] [TEST_CSV] [BASE_MODEL] [OUTPUT_CSV]
# Example with checkpoint in `checkpoint-2414`:
./run_predict.sh ./checkpoint-2414 ./data/test.csv bert-large-cased-whole-word-masking ./submission.csv
```

Notes:
- The script runs `code/predict_for_report.py`.
- `MODEL_DIR` should contain `model.safetensors` and `config.json`.
- The `config.json` must include `label2id` and `id2label` mappings that the model was trained with, as well as feature flags (POS/char/capital) for correct feature generation.

Fixes / Notes
- `predict_for_report.py` now passes `tokenizer`, `exp_config`, `pos_tag2id` and `label2id` into the tokenization function via `datasets.map(..., fn_kwargs=...)` so tokenization + feature generation works correctly inside worker processes.

## How the prediction script works

- `predict_for_report.py` accepts CLI arguments:
  - `--model-dir` (default: `./checkpoint-2414`)
  - `--test-csv` (default: `./data/test.csv`)
  - `--base-model` (default: `bert-large-cased-whole-word-masking`)
  - `--output-csv` (default: `./submission.csv`)

- The script will load the model safetensor (`model.safetensors`) and `config.json` from `--model-dir`.
- It tokenizes `test.csv` sentences, generates additional features (POS, capital, char) if configured, performs inference, aligns subword predictions back to whole words, and writes `submission.csv` in the required format.

## Output format

The `submission.csv` includes these columns:

- `id`: original id from `test.csv`.
- `NER Tag`: list-serialized string representing predicted tags for each token in the sentence. (Matches the input CSV expectations.)

## Troubleshooting

- If `predict_for_report.py` errors due to missing NLTK resources, install the NLTK tagger:

```bash
python -c "import nltk; nltk.download('averaged_perceptron_tagger')"
```
Alternatively, the `setup.sh` will download the tagger for you when creating the conda environment.

- If you see `FileNotFoundError` for `model.safetensors` or `config.json` make sure you supply the correct `--model-dir`.

- If your base model (e.g., `bert-large-cased-whole-word-masking`) is not available locally, the script will download it from Hugging Face; ensure you have internet access or local cache.

Quick GPU check (inside activated env):

```bash
python - <<'PY'
import torch
print('torch version:', torch.__version__)
print('CUDA available:', torch.cuda.is_available())
if torch.cuda.is_available():
  print('CUDA version (torch):', torch.version.cuda)
  print('GPU name:', torch.cuda.get_device_name(0))
PY
```

## Notes / Next steps

- For reproducible inference, ensure your Python environment contains exact versions of dependencies as used during development.

```bash
# for RTX 4060 8GB with CUDA 12.6
./setup.sh ner_env 3.9 gpu 12.6
conda activate ner_env
```
- If you have a different checkpoint filename, adjust `model.safetensors` name or update `predict_for_report.py` to read the correct file name.

## License & Credits

Check the license of Hugging Face models on https://huggingface.co.

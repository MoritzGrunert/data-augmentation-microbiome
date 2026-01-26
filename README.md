# Project Structure

- `paper/`: LaTeX source of the term paper
- `code/`: Code for data augmentation experiments
- `notebooks/`: Exploratory analysis

## Environment Setup

This project was developed and tested with Python 3.10.

Using Conda (recommended)

```bash
conda create -n microbiome-da python=3.10
conda activate microbiome-da
pip install -r requirements.txt
```

Alternative: without Conda

```bash
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

```bash
python evaluation.py --experiment e1 --model rf --reps 20
```

```bash
python evaluation.py --experiment e1 --model lr --reps 20
```

```bash
python evaluation.py --experiment e2 --model lr --reps 20 --reduce-r 194
```

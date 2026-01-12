## Project Structure

- `paper/`: LaTeX source of the term paper
- `code/`: Code for data augmentation experiments
- `notebooks/`: Exploratory analysis

### Environment Setup

This project was developed and tested with Python 3.10.

Using Conda (recommended)
```bash
conda create -n microbiome-da python=3.10
conda activate microbiome-da
pip install -r requirements.txt
```

Alternative: without Conda
```
python -m venv .venv
source .venv/bin/activate      # macOS / Linux
# .venv\Scripts\activate       # Windows
pip install -r requirements.txt
```

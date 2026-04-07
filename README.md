# functiongemma-250m-it finetune

Repository for finetuning `functiongemma`. Code based on [google docs](https://ai.google.dev/gemma/docs/functiongemma/finetuning-with-functiongemma)

Running:

```bash
git clone https://github.com/M1chol/gemma-finetune.git
cd gemma-finetune
python -m venv .venv
pip install -r requierments.txt
python download_base.py
python test_dataset.py
python train.py
```

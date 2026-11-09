# Teaching a tiny GPT arithmetic with DPO

We fine-tuned a small character-level GPT (nanoGPT) with **Direct Preference Optimisation (DPO)** so it solves arithmetic and one-step algebra. The base model had only seen conversational QA data and refused every math question. After fine-tuning, it answers **96.7% of held-out questions correctly** (232/240).

```
17*121=?         → The answer is 2057 because 17*121 equals 2057.
x/24=51,x=?      → The answer is 1224 because 24*51 equals 1224.
2639/x=91,x=?    → The answer is 29 because 2639/91 equals 29.
```

## Results

The test set has 240 questions (20 for each of 12 question types). None of them appear in the training or validation data.

| Question type | First model | Intermediate | **Final** |
|---|---|---|---|
| `a+b`, `a-b` | 17, 20 | 20, 20 | **20, 20** |
| `a*b`, `a/b` | 3, 4 | 7, 13 | **19, 18** |
| `a+x=b`, `x+a=b` | 18, 18 | 19, 19 | **20, 20** |
| `a-x=b`, `x-a=b` | 18, 18 | 19, 20 | **20, 20** |
| `a*x=b`, `x*a=b` | 5, 3 | 15, 18 | **18, 20** |
| `a/x=b`, `x/a=b` | 7, 3 | 19, 6 | **19, 18** |
| **Overall** | 55.8% | 81.2% | **96.7%** |

Each of the 8 remaining errors is a single wrong digit (for example, `9*63 → 587`).

## Approach

1. **Data:** 101k positive/negative pairs across 12 question types, generated in Python so every answer is exactly right. A positive is a worked answer and a negative is a refusal. We checked the base model first: it gives the same refusal to 96% of math questions, so we wrote the negatives directly instead of sampling them from the model.
2. **Loss:** the DPO preference loss plus an SFT term on the positive answer:

   $$\mathcal{L} = -\log\sigma\Big(\tfrac{\log\pi(y^+) - \log\pi(y^-)}{\beta}\Big) - 0.1\cdot\log\pi(y^+), \quad \beta = 0.5$$

   The preference term teaches the model to answer instead of refusing. It saturates early, and after that the SFT term does most of the work of teaching the digits.
3. **Training:** AdamW with cosine decay from `3e-4` over 15 epochs. After every epoch, a 120-question validation set picks the checkpoint to keep.

### What made the difference

| Change | Held-out accuracy |
|---|---|
| Template settings (5 epochs, `lr=1e-4`) | 7.1% |
| 15 epochs, `lr=3e-4`, cosine schedule, best-val checkpoint | 81.2% |
| + dropout off, + oversampling of `a*b` and `x/a=b` | **96.7%** |

The biggest lesson: short runs learn the *answer format* but stop at about 7% before they learn the arithmetic. Many ideas we tried looked equally bad simply because they had not been trained long enough. The [notebook](dpo/dpo.ipynb) covers every experiment, the loss curves and the error analysis.

## Repository layout

```
dpo/dpo.ipynb            full pipeline: data → training → evaluation, with analysis
dpo/pos_neg_pairs.json   generated preference pairs
dpo/dpo.pt               fine-tuned checkpoint (Git LFS)
sft/gpt.pt               pretrained base model (Git LFS)
sft/meta.pkl             character-level tokenizer
model.py                 GPT model definition
```

## Running it

```bash
git lfs install && git clone <this repo>
pip install torch numpy matplotlib tqdm jupyter
cd dpo && jupyter notebook dpo.ipynb
```

Run the cells top to bottom. The notebook uses CUDA, Apple MPS or CPU, whichever is available in that order. A full training run takes about 3 hours on an Apple-silicon laptop. To evaluate the included checkpoint without retraining, jump to Step 8.

## Team

| | |
|---|---|
| Natasha Lye | data generation, training |
| Shiqi | training |
| Zayd Shabaz Ali | evaluation, write-up |

Built for SC3000 (Artificial Intelligence) at NTU. `model.py` is adapted from Andrej Karpathy's [nanoGPT](https://github.com/karpathy/nanoGPT) (MIT). The pretrained base model came with the course.

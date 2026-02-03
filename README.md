# VRQA: Context-Adaptive View Routing for Long-document Question-answer Generation

VRQA provides a full pipeline for data preparation, QA generation, training, inference, and evaluation.

**Repository Layout**
```
VRQA/
  README.md
  environment.yml
  requirements.txt
  .env.example

  vrqa/
    data/
    prompts/
    router/
    selector/
    pipeline/
    eval/
    utils/

  scripts/
    01_prepare_data.py
    02_train_stage0.py
    03_train_stage1.py
    04_train_stage2.py
    05_infer.py
    06_eval_quality.py
    07_eval_diversity.py
```

**Step 1. Environment Setup**
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install -e .
```

Copy `.env.example` to `.env` and set required variables:
- `LLM_API_KEY=""`
- `LLM_API_BASE_URL=""`
- `LLM_MODEL=""`
- `EMBEDDING_MODEL_PATH=""`

Optional:
- `QA_SCORER_MODEL_PATH=""` for local scoring model

**Step 2. Data Preparation**
Prepare chunks with doc-level and chunk-level anchors.
```bash
python scripts/01_prepare_data.py \
  --input "" \
  --output "" \
  --backend llm \
  --min_chars 800 \
  --max_chars 2200 \
  --overlap 150
```

**Step 3. Training (3 Stages)**
Stage 0: LinUCB only.
```bash
python scripts/02_train_stage0.py \
  --input "" \
  --output "" \
  --num_rounds 10 \
  --sample_ratio_per_round 0.08
```

Stage 1: FiLM only (uniform/mixed view sampling).
```bash
python scripts/03_train_stage1.py \
  --input "" \
  --output "" \
  --num_rounds 20 \
  --sample_ratio_per_round 0.08
```

Stage 2: LinUCB only.
```bash
python scripts/04_train_stage2.py \
  --input "" \
  --output "" \
  --num_rounds 20 \
  --sample_ratio_per_round 0.08
```

**Step 4. Inference**
```bash
python scripts/05_infer.py \
  --input "" \
  --output ""
```

**Step 5. Evaluation**
Quality evaluation:
```bash
python scripts/06_eval_quality.py --method alignscore \
  --results_json "" \
  --anchors_json "" \
  --ckpt_path "" \
  --model "" \
  --out_dir ""
```

Diversity evaluation:
```bash
python scripts/07_eval_diversity.py \
  --input "" \
  --out_dir "" \
  --simcse_model "" \
  --bertscore_model ""
```

**Evaluation Notes**
- AlignScore Evaluation  
  Note: AlignScore must be installed from its GitHub repository.
- UniEval Evaluation  
  Note: UniEval must be installed from its GitHub repository, and both `unieval-dialog` and `unieval-sum` models should be downloaded beforehand.
- Diversity Evaluation (`vrqa/eval/diversity_semantic.py`)  
  Note: this script requires a SimCSE-style encoder and a BERTScore encoder. Download both model checkpoints in advance.
  Example SimCSE model: `princeton-nlp/sup-simcse-bert-base-uncased`  
  Example BERTScore model: `roberta-large`

**Notes**
- All scripts support `--help` for detailed options.
- The generator uses `LLM_API_*` for LLM calls and `EMBEDDING_MODEL_PATH` for semantic features.

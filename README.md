# Project 3 – Context-aware Emotion Recognition
## CAP 4628 / CAI 5615 Affective Computing

---

## Project Structure

```
emotic-emotion-recognition/
├── emotic/                        ← unzip emotic.zip here
│   ├── images/                    ← image subsets (ade20k, mscoco, …)
│   └── annotations.csv
│
├── src/
│   ├── config.py                  ← all paths, model registry, 26 EMOTIC classes
│   ├── dataset.py                 ← load & validate annotations.csv
│   ├── face_crop.py               ← GT bbox face cropping (PIL)
│   ├── prompt.py                  ← forced-choice prompt builder + output parser
│   ├── inference.py               ← inference loop with resume support
│   ├── evaluate.py                ← accuracy, precision, recall, F1, confusion matrix
│   ├── analyze.py                 ← comparison charts, failure cases, sample images
│   └── models/
│       ├── base_model.py          ← abstract VLM base class
│       ├── qwen_model.py          ← Qwen2.5-VL-7B-Instruct
│       ├── llava_model.py         ← LLaVA-NeXT (Mistral-7B)
│       └── internvl_model.py      ← InternVL2.5-8B
│
├── outputs/
│   ├── task1/                     ← Task 1 results CSVs, metrics, confusion matrices
│   ├── task2/                     ← Task 2 results CSVs, crop failure report
│   └── analysis/                  ← comparison charts, failure mosaics, sample images
│
├── cropped_faces/                 ← auto-generated face crops (Task 2)
│
├── run_task1.py                   ← Task 1 runner
├── run_task2.py                   ← Task 2 runner
├── run_analysis.py                ← analysis & plots runner
├── requirements.txt
└── README.md
```

---

## Models Used

| Key | Model | HuggingFace ID |
|-----|-------|----------------|
| `qwen` | Qwen2.5-VL-7B-Instruct | Qwen/Qwen2.5-VL-7B-Instruct |
| `llava` | LLaVA-NeXT (Mistral-7B) | llava-hf/llava-v1.6-mistral-7b-hf |
| `internvl` | InternVL2.5-8B | OpenGVLab/InternVL2_5-8B |

---

## Setup & Run Steps (VS Code)

### Step 1 — Unzip the dataset
Unzip `emotic.zip` so the folder structure is:
```
emotic-emotion-recognition/emotic/images/ade20k/...
emotic-emotion-recognition/emotic/images/mscoco/...
emotic-emotion-recognition/emotic/annotations.csv
```

### Step 2 — Open in VS Code
```
File → Open Folder → select emotic-emotion-recognition/
```

### Step 3 — Create and activate a virtual environment
Open the VS Code terminal  (`Ctrl + `` ` ``)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# Mac / Linux
source venv/bin/activate
```

### Step 4 — Install dependencies
```bash
pip install -r requirements.txt
```
> **Note:** First run will download model weights (~14–16 GB per model).
> Make sure you have a GPU with ≥16 GB VRAM and a stable internet connection.
> To run on CPU instead, open `src/config.py` and set `DEVICE = "cpu"`.

### Step 5 — Quick sanity test (50 images, one model)
```bash
python run_task1.py --model qwen --sample 50
python run_task2.py --model qwen --sample 50
python run_analysis.py
```

### Step 6 — Full Task 1 (Full Image Classification)
```bash
# One model at a time:
python run_task1.py --model qwen
python run_task1.py --model llava
python run_task1.py --model internvl

# Or all three sequentially:
python run_task1.py --model all
```

### Step 7 — Full Task 2 (Face-Only Classification)
```bash
python run_task2.py --model qwen
python run_task2.py --model llava
python run_task2.py --model internvl

# Or all at once:
python run_task2.py --model all
```

### Step 8 — Generate analysis plots
```bash
python run_analysis.py
```

---

## Output Files

| File | Description |
|------|-------------|
| `outputs/task1/{model}_task1_results.csv` | Per-image predictions (Task 1) |
| `outputs/task1/{model}_task1_metrics.json` | Accuracy / precision / recall / F1 |
| `outputs/task1/{model}_task1_confusion_matrix.png` | Heatmap |
| `outputs/task2/{model}_task2_results.csv` | Per-image predictions (Task 2) |
| `outputs/task2/crop_failures.csv` | Images where GT bbox cropping failed |
| `outputs/analysis/comparison_table.csv` | Cross-model results table |
| `outputs/analysis/f1_comparison_bar.png` | F1 bar chart full vs face |
| `outputs/analysis/{model}_per_class_recall.png` | Per-class recall bar chart |
| `outputs/analysis/failure_cases/` | Qualitative mosaic images |
| `outputs/analysis/sample_images_per_label/` | One example per EMOTIC class |

---

## Key Config Options (`src/config.py`)

| Variable | Default | Description |
|----------|---------|-------------|
| `DEVICE` | `"cuda"` | Change to `"cpu"` if no GPU |
| `SAMPLE_SIZE` | `None` | Set to an int (e.g. `200`) to always run on a subset |
| `MAX_NEW_TOKENS` | `50` | Max tokens the model generates per prediction |
| `RANDOM_SEED` | `42` | Reproducibility seed |

---

## Resume Support
If a run is interrupted, simply re-run the same command.
`inference.py` reads the existing results CSV and skips already-processed images.

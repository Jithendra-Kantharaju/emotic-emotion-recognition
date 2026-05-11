import os

BASE_DIR        = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EMOTIC_ROOT     = os.path.join(BASE_DIR, "emotic")
ANNOTATIONS_CSV = os.path.join(EMOTIC_ROOT, "annotations.csv")
IMAGES_ROOT     = os.path.join(EMOTIC_ROOT, "images")

OUTPUT_DIR   = os.path.join(BASE_DIR, "outputs")
TASK1_OUT    = os.path.join(OUTPUT_DIR, "task1")
TASK2_OUT    = os.path.join(OUTPUT_DIR, "task2")
ANALYSIS_OUT = os.path.join(OUTPUT_DIR, "analysis")
CROPPED_DIR  = os.path.join(BASE_DIR, "cropped_faces")

EMOTIC_CLASSES = [
    "Affection", "Anger", "Annoyance", "Anticipation", "Aversion",
    "Confidence", "Disapproval", "Disconnection", "Disquietment",
    "Doubt/Confusion", "Embarrassment", "Engagement", "Esteem",
    "Excitement", "Fatigue", "Fear", "Happiness", "Pain", "Peace",
    "Pleasure", "Sadness", "Sensitivity", "Suffering", "Surprise",
    "Sympathy", "Yearning",
]

# 3 different Gemini models — all FREE, no download needed
MODELS = {
    "qwen": {
        "name":   "LLaVA-7B (Full Image)",
        "hf_id":  "llava:7b",
        "loader": "qwen",
    },
    "llava": {
        "name":   "LLaVA-7B (Face Only)",
        "hf_id":  "llava:7b",
        "loader": "llava",
    },
    "internvl": {
        "name":   "LLaVA-7B (Fusion)",
        "hf_id":  "llava:7b",
        "loader": "internvl",
    },
}

MAX_NEW_TOKENS = 50
DEVICE         = "cpu"
TORCH_DTYPE    = "float32"
SAMPLE_SIZE    = None
RANDOM_SEED    = 42
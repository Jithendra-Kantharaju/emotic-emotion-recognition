from src.config import EMOTIC_CLASSES

# Short comma-separated list instead of numbered list (saves ~300 tokens per request)
_CLASS_LIST = ", ".join(EMOTIC_CLASSES)

def build_prompt(context: str = "full") -> str:
    if context == "full":
        img_desc = "Full scene image with person and background."
    else:
        img_desc = "Cropped face image only, no background."

    return (
        f"{img_desc}\n"
        f"Classify the person's emotion into exactly ONE of these 26 categories:\n"
        f"{_CLASS_LIST}\n"
        "Reply with ONLY the category name, nothing else."
    )

def parse_prediction(raw: str) -> str:
    if not raw:
        return "Unknown"
    cleaned = raw.strip().strip(".")
    for cls in EMOTIC_CLASSES:
        if cleaned.lower() == cls.lower():
            return cls
    for cls in EMOTIC_CLASSES:
        if cls.lower() in cleaned.lower():
            return cls
    return "Unknown"
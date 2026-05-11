import requests
import base64
from io import BytesIO
from PIL import Image
from src.models.base_model import BaseVLM

OLLAMA_URL = "http://localhost:11434/api/generate"
OLLAMA_MODEL = "llava-phi3"

def _to_b64(image: Image.Image, max_size: int = 512) -> str:
    img = image.convert("RGB")
    w, h = img.size
    if max(w, h) > max_size:
        scale = max_size / max(w, h)
        img = img.resize((int(w * scale), int(h * scale)), Image.LANCZOS)
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=80)
    return base64.b64encode(buf.getvalue()).decode("utf-8")

class QwenVLModel(BaseVLM):
    def load(self):
        try:
            resp = requests.get("http://localhost:11434/api/tags", timeout=5)
            if resp.ok:
                print(f"  {self.model_name} ready via Ollama (local, no quota).\n")
            else:
                raise ConnectionError("Ollama not running")
        except Exception:
            raise EnvironmentError(
                "\n[ERROR] Ollama is not running.\n"
                "1. Install from https://ollama.com\n"
                "2. Run: ollama pull llava:7b\n"
                "3. Ollama starts automatically after install.\n"
            )

    def predict(self, image: Image.Image, prompt: str) -> str:
        try:
            payload = {
                "model": OLLAMA_MODEL,
                "prompt": prompt,
                "images": [_to_b64(image)],
                "stream": False,
                "options": {"temperature": 0, "num_predict": 20}
            }
            resp = requests.post(OLLAMA_URL, json=payload, timeout=300)
            if resp.ok:
                return resp.json().get("response", "Unknown").strip()
            print(f"  [ERROR] {resp.status_code}: {resp.text[:100]}")
            return "Unknown"
        except Exception as e:
            print(f"  [ERROR] {e}")
            return "Unknown"
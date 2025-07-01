# ~/Zoidberg/Web/chat_agent.py
"""
Agent local alimenté par Mistral-7B-Instruct v0.2 (format GGUF + llama-cpp-python)
"""

from pathlib import Path
from llama_cpp import Llama

# 🔧 Chemin vers le fichier .gguf fraîchement téléchargé
MODEL_PATH = Path("/home/zoidberg/Zoidberg/models/mistral/mistral-7b-instruct-v0.2.Q4_K_M.gguf")

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Modèle introuvable : {MODEL_PATH}")

print("🚀  Chargement de Mistral-7B-Instruct… (quelques secondes)")
llm = Llama(
    model_path=str(MODEL_PATH),
    # ——— Réglages que tu peux ajuster ———
    n_ctx=2048,      # longueur de contexte (2048 = valeur sûre)
    n_threads=8,     # nb. de threads CPU (mets le nb. de cœurs logiques)
    n_gpu_layers=0,  # 0 = tout CPU | >0 si tu veux déporter des couches sur GPU
    verbose=False,
)

def generate_response(message: str, max_tokens: int = 256) -> str:
    """
    Renvoie la réponse brute du modèle (sans tags spéciaux).
    """
    prompt = f"[INST] {message.strip()} [/INST]"
    out = llm(
        prompt,
        max_tokens=max_tokens,
        temperature=0.7,
        top_p=0.95,
        stop=["</s>"],
    )

    # Le texte renvoyé est dans out["choices"][0]["text"]
    return out["choices"][0]["text"].strip()

# -*- coding: utf-8 -*-
import os

# Mode CI pour éviter de charger le modèle
SKIP_MODEL = os.environ.get(
    "SKIP_MODEL_DOWNLOAD",
    "false"
).lower() in ("1", "true", "yes")

# Charger un modèle léger compatible Render
if not SKIP_MODEL:
    from transformers import (
        AutoTokenizer,
        AutoModelForSeq2SeqLM,
        pipeline
    )

    MODEL_NAME = "google/flan-t5-small"

    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)

    gen_pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=150,
        temperature=0.4,
    )


def generer_reponse(texte, sentiment="negative"):
    """
    Génère une réponse polie au client.
    Version compatible Render (CPU friendly).
    """

    # MODE TEST CI
    if SKIP_MODEL:
        return (
            f"[MOCK] Réponse simulée. "
            f"Texte='{texte[:40]}...' "
            f"(sentiment={sentiment})"
        )

    prompt = (
        "Tu es un assistant de service client professionnel.\n"
        f"Message du client : {texte}\n"
        f"Sentiment détecté : {sentiment}\n"
        "Rédige une réponse polie, courte et empathique.\n\n"
        "Réponse :"
    )

    result = gen_pipe(prompt)[0]["generated_text"]
    return result

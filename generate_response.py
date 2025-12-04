# -*- coding: utf-8 -*-

import os
import requests

HF_MODEL = "Qwen/Qwen2.5-3B-Instruct"
HF_TOKEN = os.environ.get("HF_TOKEN")  # clé HuggingFace dans Render

API_URL = f"https://api-inference.huggingface.co/models/{HF_MODEL}"
headers = {"Authorization": f"Bearer {HF_TOKEN}"}


def generer_reponse(texte, sentiment="negative"):
    prompt = (
        "Tu es un assistant service client professionnel.\n"
        f"Message du client : {texte}\n"
        f"Sentiment détecté : {sentiment}\n"
        "Rédige une réponse polie, courte et empathique.\n\n"
        "Réponse :"
    )

    payload = {"inputs": prompt}

    response = requests.post(API_URL, headers=headers, json=payload)
    data = response.json()

    try:
        return data[0]["generated_text"]
    except:
        return "[HF ERROR] " + str(data)

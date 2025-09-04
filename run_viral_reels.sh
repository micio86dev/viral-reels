#!/bin/bash

# Script di avvio per Viral Reels
# Assicura che l'ambiente virtuale sia attivato correttamente

echo "🚀 Avvio Viral Reels..."

# Vai alla directory corretta
cd "$(dirname "$0")"

# Controlla se l'ambiente virtuale esiste
if [ ! -d "venv" ]; then
    echo "❌ Ambiente virtuale non trovato!"
    echo "💡 Esegui prima: python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt"
    exit 1
fi

# Attiva l'ambiente virtuale ed esegui il programma
echo "🔧 Attivando ambiente virtuale..."
source venv/bin/activate

echo "🎬 Avviando Viral Reels..."
python viral_reels.py
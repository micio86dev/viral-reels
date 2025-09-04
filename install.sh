#!/bin/bash
echo "🎬 Installazione Viral Reels Generator"
APP_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$APP_DIR"

echo "📦 Ricreo venv..."
rm -rf venv
python3 -m venv venv
source venv/bin/activate

echo "⬆️ Aggiorno pip e strumenti..."
pip install --upgrade pip wheel setuptools

echo "📥 Installo dipendenze..."
pip install moviepy==1.0.3 "imageio[ffmpeg]" yt-dlp pysubs2 faster-whisper transformers torch torchvision torchaudio soundfile sentencepiece tqdm

deactivate

echo "⚡ Creo lanciatore sul Desktop..."
cat > ~/Desktop/ViralReels.command << EOF
#!/bin/bash
cd "$APP_DIR"
source venv/bin/activate
python viral_reels.py
deactivate
EOF
chmod +x ~/Desktop/ViralReels.command

echo "✅ Installazione completata!"
echo "Trovi il lanciatore 'ViralReels.command' sul Desktop."

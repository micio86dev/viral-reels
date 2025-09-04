# 🎬 Viral Reels Generator

Turn any YouTube video into viral-ready reels with **AI-powered analysis, synced subtitles, and vertical format output**.  
This version improves subtitle accuracy, removes distracting overlays, and produces natural, social‑media‑friendly clips.

---

## ✨ Features

- **Automatic YouTube download** via `yt-dlp`
- **AI analysis** with Faster‑Whisper for speech-to-text + HuggingFace sentiment scoring
- **Smart segment selection** — picks the most engaging moments based on viral score
- **Synced subtitles** generated from Whisper timestamps, styled for readability
- **No score overlay** — clean output with only subtitles
- **9:16 vertical format** (1080x1920) for TikTok, Instagram Reels, YouTube Shorts
- **Customizable** duration, number of reels, and output folder
- **Simple desktop app GUI** (Tkinter)

---

## 📦 Installation

1. Place these files in your project folder:
   - `viral_reels.py`
   - `install.sh`

2. Run installation:
   ```bash
   cd ~/Desktop/"Reels Generator"
   chmod +x install.sh
   ./install.sh
   ```

3. After installation, you will find a **ViralReels.command** launcher on your Desktop.

---

## 🚀 Usage

1. Double‑click **ViralReels.command**
2. Paste a YouTube video URL
3. Configure:
   - **Max clip duration (seconds)** → e.g., 45
   - **Number of reels to generate**
   - **Output folder** (default: `~/Desktop/ViralReels`)
4. Click **🚀 GENERATE REELS**
5. Wait for processing — reels will appear in the output folder

---

## 📂 Output

- Video format: `.mp4`, resolution **1080x1920**
- Subtitles: clean, centered at the bottom:
  - White bold text
  - Black outline for readability
  - Synced with spoken audio
- Multiple reels created automatically (based on top viral segments)

---

## 🔧 Requirements

- macOS 10.14 or newer (Apple Silicon or Intel)
- Python 3.8+
- FFmpeg installed (handled by Homebrew or already present on macOS)
- Internet connection for downloading videos and models

---

## 📝 Notes

- The first run may be slower as models are downloaded (Whisper + Transformers).
- Video/audio quality is downscaled to **720p** for faster processing.
- Subtitles are generated from actual timestamps — no manual timing guesswork.
- This version **removes the yellow score overlay** to keep videos clean.

---

## 🎉 Enjoy!
Quickly transform your long YouTube videos into short, engaging reels ready for TikTok, Instagram Reels, and YouTube Shorts.

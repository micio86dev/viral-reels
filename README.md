# 🎬 Viral Reels

Turn any YouTube video into viral-ready reels with **AI-powered analysis, synced subtitles, and vertical format output**.  
This version is specifically optimized for **Mac M1/M2/M3 chips** with improved memory management and performance.

---

## ✨ Features

- **Automatic YouTube download** via `yt-dlp`
- **AI analysis** with Faster-Whisper for speech-to-text + HuggingFace sentiment scoring
- **Smart segment selection** — picks the most engaging moments based on viral score
- **High-quality synced subtitles** with Courier New Bold font in green color, positioned to avoid faces
- **No score overlay** — clean output with only subtitles
- **9:16 vertical format** (1080x1920) for TikTok, Instagram Reels, YouTube Shorts
- **Customizable** duration, number of reels, and output folder
- **Simple desktop app GUI** (Tkinter)
- **Optimized for Mac M1** — reduced memory usage and improved performance

---

## 📦 Installation

1. Place these files in your project folder:
   - `viral_reels.py`
   - `requirements.txt`
   - `install.sh`

2. Run installation:
   ```bash
   cd /path/to/viral-reels
   chmod +x install.sh
   ./install.sh
   ```

3. After installation, you will find a **ViralReels.command** launcher on your Desktop.

---

## 🚀 Usage

1. Double-click **ViralReels.command**
2. Paste a YouTube video URL
3. Configure:
   - **Max clip duration (seconds)** → e.g., 45
   - **Number of reels to generate**
   - **Output folder** (default: `~/Desktop/ViralReels`)
4. Click **🚀 GENERATE REELS**
5. Wait for processing — reels will appear in the output folder

---

## 📂 Output

Generated reels will be saved in your specified output folder with the following structure:
- `reel_1.mp4`, `reel_2.mp4`, etc. (in 9:16 vertical format)
- Each reel contains **high-quality subtitles** with these features:
  - **Courier New Bold** font for excellent readability
  - **Green color** text that stands out against most backgrounds
  - **Lower positioning** to avoid covering faces
  - **Perfect synchronization** with the audio
- Clean output with no score overlays or watermarks

---

## 🔧 Mac M1 Optimizations

This version includes several optimizations specifically for Apple Silicon Macs:

- **Lazy model loading** — AI models are only loaded when needed
- **Memory management** — Automatic cleanup to prevent RAM saturation
- **Thread limiting** — Optimized CPU usage to avoid overheating
- **Streaming processing** — Large videos are processed in chunks
- **Context managers** — Proper resource cleanup

---

## 🛠️ Troubleshooting

**If you get permission errors:**
```bash
chmod +x install.sh
chmod +x ViralReels.command
```

**If models fail to download:**
- Check your internet connection
- Ensure you have at least 2GB free disk space
- Try running the application again

**If processing is slow:**
- Close other applications
- Ensure you have at least 4GB free RAM
- First-time runs are slower as models download

---

## 📄 License

This project is for educational purposes. Please respect YouTube's Terms of Service and copyright laws when downloading videos.
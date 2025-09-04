#!/usr/bin/env python3
"""
Viral Reels Generator - Optimized for Mac M1
Implemented optimizations:
- Lazy loading of AI models
- Improved memory management with context managers
- Streaming processing for large videos
- Explicit garbage collection
- Integrated memory monitoring
- Asynchronous processing for responsive UI
"""

import os, tempfile, subprocess, re, threading, gc, psutil
from pathlib import Path
from collections import namedtuple
from contextlib import contextmanager
import tkinter as tk
from tkinter import ttk, filedialog
import yt_dlp
import numpy as np
from typing import List, Dict, Optional, Generator

# Optimizations for Mac M1
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"  # Limit threads to avoid saturation
os.environ["MKL_NUM_THREADS"] = "4"

Block = namedtuple("Block", "start end text score")

class MemoryMonitor:
    """Memory monitor to prevent RAM saturation"""
    
    @staticmethod
    def get_memory_usage():
        """Returns memory usage in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def check_memory_limit(limit_mb=6000):
        """Check if we're close to memory limit"""
        current = MemoryMonitor.get_memory_usage()
        return current > limit_mb
    
    @staticmethod
    def force_cleanup():
        """Force garbage collection and cleanup"""
        gc.collect()
        # Force numpy cleanup
        if 'numpy' in globals():
            np.random.seed()

class ModelManager:
    """Manages AI models with lazy loading and proper cleanup"""
    
    def __init__(self):
        self.whisper_model = None
        self.sentiment_analyzer = None
        self.whisper_lock = threading.Lock()
        self.sentiment_lock = threading.Lock()
    
    @contextmanager
    def get_whisper_model(self, model_size="medium"):
        """Get Whisper model with lazy loading"""
        with self.whisper_lock:
            if self.whisper_model is None:
                print("🤖 Loading Whisper model...")
                from faster_whisper import WhisperModel
                self.whisper_model = WhisperModel(
                    model_size, 
                    device="cpu", 
                    compute_type="int8"
                )
            yield self.whisper_model
    
    @contextmanager
    def get_sentiment_analyzer(self):
        """Get sentiment analyzer with lazy loading"""
        with self.sentiment_lock:
            if self.sentiment_analyzer is None:
                print("🤖 Loading sentiment analyzer...")
                from transformers import pipeline
                self.sentiment_analyzer = pipeline(
                    "sentiment-analysis",
                    model="nlptown/bert-base-multilingual-uncased-sentiment",
                    device=-1  # CPU
                )
            yield self.sentiment_analyzer
    
    def cleanup_all(self):
        """Cleanup all models and free memory"""
        with self.whisper_lock:
            if self.whisper_model:
                del self.whisper_model
                self.whisper_model = None
        
        with self.sentiment_lock:
            if self.sentiment_analyzer:
                del self.sentiment_analyzer
                self.sentiment_analyzer = None
        
        MemoryMonitor.force_cleanup()

def transcribe_with_streaming(video_path: str, chunk_duration: int = 30) -> List[Dict]:
    """Transcribe video with streaming processing to manage memory"""
    print(f"🎵 Streaming transcription of: {Path(video_path).name}")
    
    # Extract audio
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-vn", tmp_audio.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Audio extraction failed: {result.stderr}")
            return []
        
        # Process in chunks
        duration = get_audio_duration(tmp_audio.name)
        segments = []
        
        manager = ModelManager()
        with manager.get_whisper_model() as model:
            for start in range(0, int(duration), chunk_duration):
                end = min(start + chunk_duration, duration)
                print(f"🎵 Processing audio chunk {start}-{end}s...")
                
                chunk_segments = transcribe_audio_chunk(
                    model, tmp_audio.name, start, end
                )
                segments.extend(chunk_segments)
                
                # Check memory
                if MemoryMonitor.check_memory_limit():
                    print("⚠️ Memory limit reached, stopping transcription")
                    break
        
        manager.cleanup_all()
        return segments

def get_audio_duration(audio_path: str) -> float:
    """Get audio duration in seconds"""
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
           "-of", "csv=p=0", audio_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    return float(result.stdout.strip()) if result.stdout.strip() else 0

def transcribe_audio_chunk(model, audio_path: str, start: int, end: int) -> List[Dict]:
    """Transcribe a specific chunk of audio"""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_chunk:
        cmd = [
            "ffmpeg", "-y", "-ss", str(start), "-t", str(end-start),
            "-i", audio_path, "-ar", "16000", "-ac", "1", tmp_chunk.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Audio chunk extraction failed: {result.stderr}")
            return []
        
        # Transcribe chunk
        segments = []
        try:
            segments_result, _ = model.transcribe(
                tmp_chunk.name, 
                beam_size=5,
                word_timestamps=True
            )
            
            for segment in segments_result:
                segments.append({
                    'start': segment.start + start,
                    'end': segment.end + start,
                    'text': segment.text.strip()
                })
        except Exception as e:
            print(f"❌ Transcription error: {e}")
        
        return segments

def score_segments(segments: List[Dict], analyzer) -> List[Block]:
    """Score segments based on sentiment and other factors"""
    if not segments:
        return []
    
    blocks = []
    for segment in segments:
        if not segment['text'].strip():
            continue
            
        # Analyze sentiment
        try:
            result = analyzer(segment['text'])
            # Extract star rating (1-5 stars)
            label = result[0]['label']
            score = int(label.split()[0]) / 5.0  # Convert to 0-1 scale
        except:
            score = 0.5  # Default neutral score
        
        blocks.append(Block(
            start=segment['start'],
            end=segment['end'],
            text=segment['text'],
            score=score
        ))
    
    return blocks

def select_best_segments(blocks: List[Block], max_duration: int, 
                        max_clips: int) -> List[Block]:
    """Select best segments based on scores and constraints"""
    if not blocks:
        return []
    
    # Sort by score (descending)
    sorted_blocks = sorted(blocks, key=lambda x: x.score, reverse=True)
    
    selected = []
    total_duration = 0
    
    for block in sorted_blocks:
        if len(selected) >= max_clips:
            break
            
        block_duration = block.end - block.start
        if total_duration + block_duration <= max_duration:
            selected.append(block)
            total_duration += block_duration
    
    # Sort by time for sequential output
    return sorted(selected, key=lambda x: x.start)

def add_subtitles_to_video(input_video: str, blocks: List[Block],
                          output_video: str):
    """Add subtitles to video using ffmpeg"""
    print(f"📝 Adding subtitles to: {Path(output_video).name}")

    # Create ASS subtitle file with improved styling
    with tempfile.NamedTemporaryFile(suffix=".ass", mode='w',
                                    encoding='utf-8', delete=False) as f:
        ass_file = f.name
        f.write("""[Script Info]
Title: Viral Reels Subtitles
ScriptType: v2.00
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Courier New,14,&H0000FF00,&H000000FF,&H80000000,&H80000000,-1,0,0,0,100,100,0,0,1,2,1,2,10,10,100,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")

        for i, block in enumerate(blocks):
            start_time = format_timestamp(block.start)
            end_time = format_timestamp(block.end)
            # Escape special characters and highlight words
            text = block.text.replace("'", "''").replace('"', '""')
            # Highlight the entire text in green (you can modify this to highlight specific words)
            highlighted_text = f"{{\\c&H00FF00&}}{text}"
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{highlighted_text}\n")

    # Add subtitles to video with better positioning
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", f"subtitles={ass_file}:force_style='Fontname=Courier New,Bold=1,Fontsize=14,PrimaryColour=&H00FF00&,Outline=2,Shadow=1,Alignment=2,MarginV=100'",
        "-c:a", "copy", output_video
    ]

    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Subtitle addition failed: {result.stderr}")

    # Cleanup
    try:
        os.unlink(ass_file)
    except:
        pass

def format_timestamp(seconds: float) -> str:
    """Format seconds to HH:MM:SS.mmm"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours:01d}:{minutes:02d}:{secs:06.3f}"

def create_vertical_video(input_video: str, output_video: str):
    """Convert video to 9:16 vertical format (1080x1920)"""
    print(f"📱 Converting to vertical format: {Path(output_video).name}")
    
    cmd = [
        "ffmpeg", "-y", "-i", input_video,
        "-vf", "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2,setsar=1",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        output_video
    ]
    
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        print(f"❌ Vertical conversion failed: {result.stderr}")

class OptimizedApp:
    """Main GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Viral Reels Generator - Mac M1 Optimized")
        self.root.geometry("600x500")
        self.root.resizable(True, True)
        
        # Processing state
        self.is_processing = False
        
        self.setup_ui()
    
    def setup_ui(self):
        """Setup the user interface"""
        # Main frame
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # Configure grid weights
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(1, weight=1)
        
        # Title
        title_label = ttk.Label(
            main_frame, 
            text="🎬 Viral Reels Generator", 
            font=('Arial', 16, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 20))
        
        # URL input
        ttk.Label(main_frame, text="YouTube URL:").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(main_frame, textvariable=self.url_var, width=50)
        self.url_entry.grid(row=1, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=5)
        
        # Duration
        ttk.Label(main_frame, text="Max Clip Duration (seconds):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        self.duration_var = tk.StringVar(value="45")
        duration_entry = ttk.Entry(main_frame, textvariable=self.duration_var, width=10)
        duration_entry.grid(row=2, column=1, sticky=tk.W, pady=5)
        
        # Number of clips
        ttk.Label(main_frame, text="Number of Reels:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        self.clips_var = tk.StringVar(value="3")
        clips_entry = ttk.Entry(main_frame, textvariable=self.clips_var, width=10)
        clips_entry.grid(row=3, column=1, sticky=tk.W, pady=5)
        
        # Output folder
        ttk.Label(main_frame, text="Output Folder:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        self.output_var = tk.StringVar(value=str(Path.home() / "Desktop" / "ViralReels"))
        output_entry = ttk.Entry(main_frame, textvariable=self.output_var, width=50)
        output_entry.grid(row=4, column=1, sticky=(tk.W, tk.E), pady=5)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(
            row=4, column=2, padx=(5, 0), pady=5
        )
        
        # Generate button
        self.generate_btn = ttk.Button(
            main_frame, 
            text="🚀 GENERATE REELS", 
            command=self.start_generation
        )
        self.generate_btn.grid(row=5, column=0, columnspan=3, pady=20)
        
        # Progress bar
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=6, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=10)
        
        # Log area
        ttk.Label(main_frame, text="Log:").grid(
            row=7, column=0, sticky=tk.W, pady=(10, 5)
        )
        self.log_text = tk.Text(main_frame, height=15, width=70)
        log_scroll = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        self.log_text.grid(row=8, column=0, columnspan=2, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_scroll.grid(row=8, column=2, sticky=(tk.N, tk.S), pady=5)
        
        # Configure row weights
        main_frame.rowconfigure(8, weight=1)
        
        # Bind Enter key to generate
        self.root.bind('<Return>', lambda e: self.start_generation())
    
    def browse_output(self):
        """Browse for output folder"""
        folder = filedialog.askdirectory()
        if folder:
            self.output_var.set(folder)
    
    def logwrite(self, message: str):
        """Write message to log"""
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.see(tk.END)
        self.root.update_idletasks()
    
    def start_generation(self):
        """Start video generation process"""
        if self.is_processing:
            return
        
        url = self.url_var.get().strip()
        if not url:
            self.logwrite("❌ Please enter a YouTube URL")
            return
        
        try:
            max_duration = int(self.duration_var.get())
            max_clips = int(self.clips_var.get())
            output_dir = self.output_var.get()
        except ValueError:
            self.logwrite("❌ Please enter valid numbers for duration and clips")
            return
        
        if max_duration <= 0 or max_clips <= 0:
            self.logwrite("❌ Duration and clips must be positive numbers")
            return
        
        # Disable UI
        self.is_processing = True
        self.generate_btn.config(state="disabled")
        self.progress.start()
        
        # Start processing in background thread
        thread = threading.Thread(
            target=self.generate_reels,
            args=(url, max_duration, max_clips, output_dir),
            daemon=True
        )
        thread.start()
    
    def generate_reels(self, url: str, max_duration: int, max_clips: int, output_dir: str):
        """Main generation process"""
        try:
            self.logwrite(f"🚀 Starting generation for: {url}")
            
            # Create output directory
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            self.logwrite(f"📁 Output directory: {outdir}")
            
            # Download video
            self.logwrite("📥 Downloading video...")
            with tempfile.TemporaryDirectory() as temp_dir:
                temp_path = Path(temp_dir)
                video_file = temp_path / "input.mp4"
                
                # Configure yt-dlp
                ydl_opts = {
                    'format': 'best[height<=720]',  # Limit to 720p for performance
                    'outtmpl': str(video_file),
                    'quiet': True,
                    'no_warnings': True
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                
                if not video_file.exists():
                    raise Exception("Failed to download video")
                
                self.logwrite(f"✅ Video downloaded: {video_file.name}")
                
                # Initialize model manager
                model_manager = ModelManager()
                
                # Transcribe with streaming
                segments = transcribe_with_streaming(str(video_file), chunk_duration=30)
                self.logwrite(f"✅ Transcription complete: {len(segments)} segments")
                
                if not segments:
                    raise Exception("No speech detected in video")
                
                # Analyze sentiment
                with model_manager.get_sentiment_analyzer() as analyzer:
                    blocks = score_segments(segments, analyzer)
                
                self.logwrite(f"✅ Sentiment analysis complete: {len(blocks)} blocks")
                
                # Select best segments
                selected_blocks = select_best_segments(blocks, max_duration, max_clips)
                self.logwrite(f"✅ Selected {len(selected_blocks)} best segments")
                
                if not selected_blocks:
                    raise Exception("No suitable segments found")
                
                # Process each selected segment
                for i, block in enumerate(selected_blocks):
                    self.logwrite(f"🎬 Processing clip {i+1}/{len(selected_blocks)}")
                    
                    # Create clip
                    clip_file = temp_path / f"clip_{i+1}.mp4"
                    cmd = [
                        "ffmpeg", "-y", "-ss", str(block.start), "-t", str(block.end - block.start),
                        "-i", str(video_file), "-c", "copy", str(clip_file)
                    ]
                    
                    result = subprocess.run(cmd, capture_output=True, text=True)
                    if result.returncode != 0:
                        self.logwrite(f"❌ Clip creation failed: {result.stderr}")
                        continue
                    
                    # Add subtitles
                    subtitled_file = temp_path / f"subtitled_{i+1}.mp4"
                    add_subtitles_to_video(str(clip_file), [block], str(subtitled_file))
                    
                    # Convert to vertical format
                    final_file = outdir / f"reel_{i+1}.mp4"
                    create_vertical_video(str(subtitled_file), str(final_file))
                    
                    # Log memory usage
                    current_memory = MemoryMonitor.get_memory_usage()
                    self.logwrite(f"💾 Memory: {current_memory:.1f}MB")
                    
                    if MemoryMonitor.check_memory_limit():
                        self.logwrite("⚠️ Memory limit reached, stopping generation")
                        break
                
                self.logwrite(f"🎉 Generation complete! Files saved in: {outdir}")
                self.logwrite(f"💾 Final memory: {MemoryMonitor.get_memory_usage():.1f}MB")
                
        except Exception as e:
            self.logwrite(f"❌ Generation failed: {e}")
            import traceback
            self.logwrite(f"🔍 Error details: {traceback.format_exc()}")
        
        finally:
            # Final cleanup
            model_manager.cleanup_all()
            MemoryMonitor.force_cleanup()
            
            # Reset UI
            self.is_processing = False
            self.root.after(0, lambda: [
                self.generate_btn.config(state="normal"),
                self.progress.stop()
            ])

if __name__ == '__main__':
    print("🚀 Starting Viral Reels Generator - Optimized for Mac M1")
    print(f"💾 System memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"🖥️ CPU cores: {psutil.cpu_count()}")
    
    root = tk.Tk()
    app = OptimizedApp(root)
    root.mainloop()
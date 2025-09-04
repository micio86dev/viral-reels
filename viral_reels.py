#!/usr/bin/env python3
"""
Viral Reels
Advanced features:
- Intelligent sentence-based video cuts
- Dynamic word-by-word subtitle highlighting
- OpusClip-style reel generation
- Perfect sync between audio and subtitles
- TikTok-optimized 9:16 format
"""

import os, tempfile, subprocess, re, threading, gc, psutil, json, hashlib
from pathlib import Path
from collections import namedtuple
from contextlib import contextmanager
import tkinter as tk
from tkinter import ttk, filedialog
import yt_dlp
import numpy as np
from typing import List, Dict, Optional, Generator
import nltk
from nltk.tokenize import sent_tokenize
import string

# Download required NLTK data
try:
    nltk.data.find('tokenizers/punkt')
except LookupError:
    nltk.download('punkt')

# Optimizations for Mac M1
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"

# Data structures
Block = namedtuple("Block", "start end text score words")
Word = namedtuple("Word", "text start end")
Sentence = namedtuple("Sentence", "text start end words score")

CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)

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
                    device=-1
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

def get_video_id(url: str) -> str:
    """Extract video ID from YouTube URL"""
    patterns = [
        r"v=([a-zA-Z0-9_-]+)",
        r"youtu\.be/([a-zA-Z0-9_-]+)",
        r"embed/([a-zA-Z0-9_-]+)"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    
    # Fallback to hash of URL
    return hashlib.md5(url.encode()).hexdigest()[:12]

def transcribe_with_word_timestamps(video_path: str) -> List[Dict]:
    """Transcribe video with word-level timestamps"""
    print(f"🎵 Transcribing with word timestamps: {Path(video_path).name}")
    
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp_audio:
        # Extract audio with better quality for word-level accuracy
        cmd = [
            "ffmpeg", "-y", "-i", video_path,
            "-ar", "16000", "-ac", "1", "-vn", 
            "-af", "highpass=f=200,lowpass=f=3000",  # Audio filtering for better speech clarity
            tmp_audio.name
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        if result.returncode != 0:
            print(f"❌ Audio extraction failed: {result.stderr}")
            return []
        
        segments = []
        manager = ModelManager()
        
        with manager.get_whisper_model() as model:
            try:
                segments_result, _ = model.transcribe(
                    tmp_audio.name,
                    beam_size=5,
                    word_timestamps=True,
                    condition_on_previous_text=False,
                    temperature=0.0
                )
                
                for segment in segments_result:
                    words = []
                    if hasattr(segment, 'words') and segment.words:
                        for word in segment.words:
                            words.append({
                                'text': word.word.strip(),
                                'start': word.start,
                                'end': word.end
                            })
                    else:
                        # Fallback: estimate word timings
                        text_words = segment.text.strip().split()
                        word_duration = (segment.end - segment.start) / len(text_words) if text_words else 0
                        
                        for i, word_text in enumerate(text_words):
                            word_start = segment.start + i * word_duration
                            word_end = word_start + word_duration
                            words.append({
                                'text': word_text,
                                'start': word_start,
                                'end': word_end
                            })
                    
                    segments.append({
                        'start': segment.start,
                        'end': segment.end,
                        'text': segment.text.strip(),
                        'words': words
                    })
                    
            except Exception as e:
                print(f"❌ Transcription error: {e}")
        
        manager.cleanup_all()
        return segments

def segment_into_sentences(transcription_segments: List[Dict]) -> List[Sentence]:
    """Segment transcription into meaningful sentences"""
    print("📝 Segmenting into sentences...")
    
    # Combine all text and create a mapping
    full_text = ""
    word_mapping = []  # Maps character position to word info
    
    for segment in transcription_segments:
        for word_info in segment['words']:
            word_text = word_info['text']
            start_pos = len(full_text)
            full_text += word_text + " "
            end_pos = len(full_text) - 1
            
            word_mapping.append({
                'text': word_text,
                'start': word_info['start'],
                'end': word_info['end'],
                'char_start': start_pos,
                'char_end': end_pos
            })
    
    # Split into sentences
    sentences = sent_tokenize(full_text.strip())
    sentence_objects = []
    
    char_pos = 0
    for sentence_text in sentences:
        sentence_text = sentence_text.strip()
        if not sentence_text:
            continue
            
        # Find sentence position in full text
        sentence_start_char = full_text.find(sentence_text, char_pos)
        if sentence_start_char == -1:
            continue
            
        sentence_end_char = sentence_start_char + len(sentence_text)
        char_pos = sentence_end_char
        
        # Find words that belong to this sentence
        sentence_words = []
        sentence_start_time = None
        sentence_end_time = None
        
        for word_info in word_mapping:
            word_start_char = word_info['char_start']
            word_end_char = word_info['char_end']
            
            # Check if word overlaps with sentence
            if (word_start_char >= sentence_start_char and word_start_char <= sentence_end_char) or \
               (word_end_char >= sentence_start_char and word_end_char <= sentence_end_char):
                
                sentence_words.append(Word(
                    text=word_info['text'],
                    start=word_info['start'],
                    end=word_info['end']
                ))
                
                if sentence_start_time is None:
                    sentence_start_time = word_info['start']
                sentence_end_time = word_info['end']
        
        if sentence_words and sentence_start_time is not None:
            # Calculate engagement score
            score = calculate_sentence_score(sentence_text, sentence_words)
            
            sentence_objects.append(Sentence(
                text=sentence_text,
                start=sentence_start_time,
                end=sentence_end_time,
                words=sentence_words,
                score=score
            ))
    
    print(f"✅ Created {len(sentence_objects)} sentences")
    return sentence_objects

def calculate_sentence_score(text: str, words: List[Word]) -> float:
    """Calculate engagement score for a sentence"""
    score = 0.5  # Base score
    
    # Length factor (prefer medium-length sentences)
    word_count = len(words)
    if 5 <= word_count <= 15:
        score += 0.2
    elif word_count > 20:
        score -= 0.1
    
    # Excitement indicators
    excitement_words = [
        'amazing', 'incredible', 'wow', 'unbelievable', 'fantastic', 
        'awesome', 'shocking', 'surprising', 'crazy', 'insane'
    ]
    
    text_lower = text.lower()
    for word in excitement_words:
        if word in text_lower:
            score += 0.15
    
    # Question or exclamation
    if text.endswith('?') or text.endswith('!'):
        score += 0.1
    
    # Numbers and facts
    if re.search(r'\d+', text):
        score += 0.1
    
    # Action words
    action_words = ['show', 'see', 'look', 'watch', 'check', 'discover', 'learn']
    for word in action_words:
        if word in text_lower:
            score += 0.05
    
    return min(score, 1.0)  # Cap at 1.0

def select_best_sentences(sentences: List[Sentence], max_duration: int, max_clips: int) -> List[Sentence]:
    """Select best sentences for reel creation"""
    if not sentences:
        return []
    
    # Sort by score (descending)
    sorted_sentences = sorted(sentences, key=lambda x: x.score, reverse=True)
    
    selected = []
    
    for sentence in sorted_sentences:
        if len(selected) >= max_clips:
            break
        
        sentence_duration = sentence.end - sentence.start
        
        # Filter sentences that are too short or too long
        if sentence_duration < 2 or sentence_duration > max_duration:
            continue
        
        # Avoid sentences that are too close to already selected ones
        too_close = False
        for selected_sentence in selected:
            time_gap = min(
                abs(sentence.start - selected_sentence.end),
                abs(selected_sentence.start - sentence.end)
            )
            if time_gap < 5:  # At least 5 seconds apart
                too_close = True
                break
        
        if not too_close:
            selected.append(sentence)
    
    # Sort by time for sequential processing
    return sorted(selected, key=lambda x: x.start)

def create_word_level_subtitles(sentence: Sentence, output_path: str):
    """Create ASS subtitle file with word-level highlighting"""
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("""[Script Info]
Title: OpusClip Style Subtitles
ScriptType: v4.00+
WrapStyle: 0
ScaledBorderAndShadow: yes
YCbCr Matrix: TV.601

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Default,Courier New,28,&H00FFFFFF,&H000000FF,&H00000000,&H80000000,-1,0,0,0,100,100,0,0,1,3,0,2,30,30,150,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
""")
        
        words = sentence.words
        if not words:
            return
        
        # Adjust word timings relative to sentence start
        sentence_start = sentence.start
        
        for i, word in enumerate(words):
            word_start = word.start - sentence_start
            word_end = word.end - sentence_start
            
            # Create the subtitle line with current word highlighted
            subtitle_parts = []
            
            for j, w in enumerate(words):
                if j == i:
                    # Highlight current word in green
                    subtitle_parts.append(f"{{\\c&H00FF00&}}{w.text}{{\\c&HFFFFFF&}}")
                else:
                    # Regular white text
                    subtitle_parts.append(w.text)
            
            subtitle_text = " ".join(subtitle_parts)
            
            # Escape special characters
            subtitle_text = subtitle_text.replace("\\", "\\\\").replace("{", "\\{").replace("}", "\\}")
            
            start_time = format_ass_time(max(0, word_start))
            end_time = format_ass_time(word_end)
            
            f.write(f"Dialogue: 0,{start_time},{end_time},Default,,0,0,0,,{subtitle_text}\n")

def format_ass_time(seconds: float) -> str:
    """Format seconds to ASS time format (H:MM:SS.cc)"""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hours}:{minutes:02d}:{secs:05.2f}"

def create_reel_from_sentence(video_path: str, sentence: Sentence, output_path: str, temp_dir: Path, logger=None):
    """Create a single reel from a sentence"""

    def log_message(msg):
        if logger:
            logger(msg)
        else:
            print(msg)

    # Step 1: Extract video segment
    segment_file = temp_dir / "segment.mp4"

    # Add small padding to ensure we don't cut mid-word
    padding = 0.2
    start_time = max(0, sentence.start - padding)
    duration = sentence.end - sentence.start + (2 * padding)

    cmd_extract = [
        "ffmpeg", "-y", "-ss", str(start_time), "-t", str(duration),
        "-i", str(video_path), "-c:v", "libx264", "-c:a", "aac",
        "-avoid_negative_ts", "make_zero", str(segment_file)
    ]

    log_message(f"🎬 Extracting video segment: {start_time:.2f}s to {start_time + duration:.2f}s")
    result = subprocess.run(cmd_extract, capture_output=True, text=True)
    if result.returncode != 0:
        log_message(f"❌ Segment extraction failed: {result.stderr}")
        return False

    # Step 2: Create word-level subtitles
    subtitle_file = temp_dir / "subtitles.ass"
    create_word_level_subtitles(sentence, str(subtitle_file))
    log_message(f"📝 Created subtitles with {len(sentence.words)} words")

    # Step 3: Apply subtitles to video
    subtitled_file = temp_dir / "subtitled.mp4"

    cmd_subtitle = [
        "ffmpeg", "-y", "-i", str(segment_file), "-vf",
        f"ass={subtitle_file}",
        "-c:a", "copy", str(subtitled_file)
    ]

    log_message(f"📝 Applying subtitles to video")
    result = subprocess.run(cmd_subtitle, capture_output=True, text=True)
    if result.returncode != 0:
        log_message(f"❌ Subtitle application failed: {result.stderr}")
        return False

    # Step 4: Convert to TikTok format (9:16)
    cmd_format = [
        "ffmpeg", "-y", "-i", str(subtitled_file),
        "-vf",
        "scale=1080:1920:force_original_aspect_ratio=decrease,pad=1080:1920:(ow-iw)/2:(oh-ih)/2:black,setsar=1",
        "-c:v", "libx264", "-preset", "medium", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k", "-ar", "44100",
        str(output_path)
    ]

    log_message(f"🎥 Converting to TikTok format (9:16)")
    result = subprocess.run(cmd_format, capture_output=True, text=True)
    if result.returncode != 0:
        log_message(f"❌ Format conversion failed: {result.stderr}")
        return False

    log_message(f"✅ Successfully created reel: {Path(output_path).name}")
    return True

class OpusClipApp:
    """Main GUI application"""
    
    def __init__(self, root):
        self.root = root
        self.root.title("🎬 Viral Reels")
        self.root.geometry("650x600")
        self.root.resizable(True, True)

        # Processing state
        self.is_processing = False
        self.generation_success = False
        self.generated_count = 0

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
            text="🎬 OpusClip Style Reels Generator", 
            font=('Arial', 18, 'bold')
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=(0, 25))
        
        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Generate viral reels with intelligent cuts and dynamic subtitles",
            font=('Arial', 10),
            foreground='gray'
        )
        desc_label.grid(row=1, column=0, columnspan=3, pady=(0, 20))
        
        # URL input
        ttk.Label(main_frame, text="YouTube URL:", font=('Arial', 11, 'bold')).grid(
            row=2, column=0, sticky=tk.W, pady=8
        )
        self.url_var = tk.StringVar()
        self.url_entry = ttk.Entry(main_frame, textvariable=self.url_var, width=55, font=('Arial', 10))
        self.url_entry.grid(row=2, column=1, columnspan=2, sticky=(tk.W, tk.E), pady=8)
        
        # Duration
        ttk.Label(main_frame, text="Max Reel Duration (seconds):", font=('Arial', 11, 'bold')).grid(
            row=3, column=0, sticky=tk.W, pady=8
        )
        self.duration_var = tk.StringVar(value="60")
        duration_entry = ttk.Entry(main_frame, textvariable=self.duration_var, width=15, font=('Arial', 10))
        duration_entry.grid(row=3, column=1, sticky=tk.W, pady=8)
        
        # Number of clips
        ttk.Label(main_frame, text="Number of Reels:", font=('Arial', 11, 'bold')).grid(
            row=4, column=0, sticky=tk.W, pady=8
        )
        self.clips_var = tk.StringVar(value="5")
        clips_entry = ttk.Entry(main_frame, textvariable=self.clips_var, width=15, font=('Arial', 10))
        clips_entry.grid(row=4, column=1, sticky=tk.W, pady=8)
        
        # Output folder
        ttk.Label(main_frame, text="Output Folder:", font=('Arial', 11, 'bold')).grid(
            row=5, column=0, sticky=tk.W, pady=8
        )
        self.output_var = tk.StringVar(value=str(Path.home() / "Desktop" / "OpusClip_Reels"))
        output_entry = ttk.Entry(main_frame, textvariable=self.output_var, width=55, font=('Arial', 10))
        output_entry.grid(row=5, column=1, sticky=(tk.W, tk.E), pady=8)
        ttk.Button(main_frame, text="Browse", command=self.browse_output).grid(
            row=5, column=2, padx=(8, 0), pady=8
        )
        
        # Generate button
        self.generate_btn = ttk.Button(
            main_frame, 
            text="🚀 GENERATE OPUSCLIP REELS", 
            command=self.start_generation,
            style="Accent.TButton"
        )
        self.generate_btn.grid(row=6, column=0, columnspan=3, pady=25)
        
        # Progress bar for overall process
        ttk.Label(main_frame, text="Overall Progress:", font=('Arial', 10, 'bold')).grid(
            row=7, column=0, sticky=tk.W, pady=(15, 5)
        )
        self.progress = ttk.Progressbar(main_frame, mode='indeterminate')
        self.progress.grid(row=8, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        
        # Progress bar for individual reel creation
        ttk.Label(main_frame, text="Reel Generation Progress:", font=('Arial', 10, 'bold')).grid(
            row=9, column=0, sticky=tk.W, pady=(15, 5)
        )
        self.reel_progress = ttk.Progressbar(main_frame, mode='determinate')
        self.reel_progress.grid(row=10, column=0, columnspan=3, sticky=(tk.W, tk.E), pady=5)
        self.reel_progress_label = ttk.Label(main_frame, text="", font=('Arial', 9))
        self.reel_progress_label.grid(row=11, column=0, columnspan=3, pady=(5, 15))
        
        # Log area
        ttk.Label(main_frame, text="Processing Log:", font=('Arial', 10, 'bold')).grid(
            row=12, column=0, sticky=tk.W, pady=(10, 5)
        )
        
        # Log frame with scrollbar
        log_frame = ttk.Frame(main_frame)
        log_frame.grid(row=13, column=0, columnspan=3, sticky=(tk.W, tk.E, tk.N, tk.S), pady=5)
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        
        self.log_text = tk.Text(log_frame, height=15, width=75, font=('Consolas', 9))
        log_scroll = ttk.Scrollbar(log_frame, orient=tk.VERTICAL, command=self.log_text.yview)
        self.log_text.configure(yscrollcommand=log_scroll.set)
        
        self.log_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        log_scroll.grid(row=0, column=1, sticky=(tk.N, tk.S))
        
        # Configure row weights
        main_frame.rowconfigure(13, weight=1)
        
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
        
        # Clear log
        self.log_text.delete(1.0, tk.END)

        # Reset generation state
        self.generation_success = False
        self.generated_count = 0

        # Disable UI
        self.is_processing = True
        self.generate_btn.config(state="disabled")
        self.progress.start()

        # Reset reel progress
        self.reel_progress['value'] = 0
        self.reel_progress_label.config(text="Preparing to generate reels...")
        
        # Start processing in background thread
        thread = threading.Thread(
            target=self.generate_reels,
            args=(url, max_duration, max_clips, output_dir),
            daemon=True
        )
        thread.start()
    
    def generate_reels(self, url: str, max_duration: int, max_clips: int, output_dir: str):
        """Main generation process - OpusClip style"""
        # Reset state
        self.generation_success = False
        self.generated_count = 0

        try:
            self.logwrite(f"🚀 Starting OpusClip-style generation for: {url}")

            # Step 1: Extract video ID
            self.logwrite("🔍 Extracting video ID...")
            video_id = get_video_id(url)
            self.logwrite(f"✅ Video ID: {video_id}")

            video_cache_dir = CACHE_DIR / video_id
            video_cache_dir.mkdir(exist_ok=True)

            video_file = video_cache_dir / "video.mp4"
            transcription_file = video_cache_dir / "transcription_words.json"

            # Create output directory
            outdir = Path(output_dir)
            outdir.mkdir(parents=True, exist_ok=True)
            self.logwrite(f"📁 Output directory: {outdir}")

            # Download video
            if not video_file.exists():
                self.logwrite("📥 Downloading video...")
                ydl_opts = {
                    'format': 'best[height<=720]',
                    'outtmpl': str(video_file),
                    'quiet': True,
                    'no_warnings': True
                }
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    ydl.download([url])
                self.logwrite("✅ Video downloaded")
            else:
                self.logwrite("✅ Video found in cache")

            # Verify video file exists and has size
            if not video_file.exists():
                raise Exception("Video file was not created")
            if video_file.stat().st_size == 0:
                raise Exception("Video file is empty")
            self.logwrite(f"📊 Video file size: {video_file.stat().st_size} bytes")

            # Transcribe with word-level timestamps
            if not transcription_file.exists():
                self.logwrite("🎵 Transcribing with word-level timestamps...")
                segments = transcribe_with_word_timestamps(str(video_file))
                with open(transcription_file, 'w') as f:
                    json.dump(segments, f, indent=2)
                self.logwrite("✅ Transcription complete")
            else:
                self.logwrite("✅ Transcription found in cache")
                with open(transcription_file, 'r') as f:
                    segments = json.load(f)

            if not segments:
                raise Exception("No speech detected in video")

            self.logwrite(f"📊 Transcription segments: {len(segments)}")

            # Segment into sentences
            self.logwrite("📝 Segmenting into sentences...")
            sentences = segment_into_sentences(segments)
            self.logwrite(f"✅ Created {len(sentences)} meaningful sentences")

            if not sentences:
                raise Exception("No meaningful sentences found")

            # Select best sentences
            self.logwrite("🎯 Selecting best sentences...")
            selected_sentences = select_best_sentences(sentences, max_duration, max_clips)
            self.logwrite(f"✅ Selected {len(selected_sentences)} best sentences for reels")

            if not selected_sentences:
                raise Exception("No suitable sentences found for reel creation")

            # Generate reels
            total_clips = len(selected_sentences)
            self.root.after(0, self._update_progress_init, total_clips)

            successful_reels = 0
            for i, sentence in enumerate(selected_sentences):
                self.root.after(0, self._update_reel_progress, i + 1, total_clips)
                self.logwrite(f"🎬 Creating reel {i+1}/{total_clips}: '{sentence.text[:50]}...'")

                # Create temporary directory for this reel
                with tempfile.TemporaryDirectory() as temp_dir:
                    temp_path = Path(temp_dir)
                    final_file = outdir / f"reel_{i+1:02d}.mp4"

                    success = create_reel_from_sentence(
                        str(video_file), sentence, str(final_file), temp_path, self.logwrite
                    )

                    if success:
                        successful_reels += 1
                        self.logwrite(f"✅ Reel {i+1} created successfully")
                        # Verify file was actually created
                        if final_file.exists():
                            self.logwrite(f"📁 File saved: {final_file} ({final_file.stat().st_size} bytes)")
                        else:
                            self.logwrite(f"❌ File not found: {final_file}")
                    else:
                        self.logwrite(f"❌ Failed to create reel {i+1}")

                # Memory check
                current_memory = MemoryMonitor.get_memory_usage()
                self.logwrite(f"💾 Memory: {current_memory:.1f}MB")

                if MemoryMonitor.check_memory_limit():
                    self.logwrite("⚠️ Memory limit reached, stopping generation")
                    break

            self.generated_count = successful_reels

            if successful_reels > 0:
                self.generation_success = True
                self.logwrite(f"🎉 Generation complete! Created {successful_reels} reels")
                self.logwrite(f"📁 Files saved in: {outdir}")
            else:
                self.logwrite(f"❌ No reels were successfully created!")

        except Exception as e:
            self.logwrite(f"❌ Generation failed: {e}")
            import traceback
            self.logwrite(f"🔍 Error details: {traceback.format_exc()}")

        finally:
            # Cleanup
            MemoryMonitor.force_cleanup()
            self.is_processing = False
            self.root.after(0, self._reset_ui)
    
    def _update_progress_init(self, total_clips):
        """Initialize progress bar"""
        self.reel_progress.config(maximum=total_clips)
        self.reel_progress['value'] = 0
    
    def _update_reel_progress(self, current, total):
        """Update reel progress bar"""
        self.reel_progress.config(value=current)
        self.reel_progress_label.config(text=f"Creating reel {current} of {total}")
    
    def _reset_ui(self):
        """Reset UI after processing"""
        self.generate_btn.config(state="normal")
        self.progress.stop()
        self.reel_progress['value'] = self.reel_progress['maximum']

        # Show appropriate message based on results
        if self.generation_success and self.generated_count > 0:
            self.reel_progress_label.config(text=f"✅ {self.generated_count} reels generated successfully! 🎉")
        elif self.generated_count == 0:
            self.reel_progress_label.config(text="❌ No reels were created")
        else:
            self.reel_progress_label.config(text="⚠️ Generation completed with errors")

if __name__ == '__main__':
    print("🚀 Starting Viral Reels")
    print(f"💾 System memory: {psutil.virtual_memory().total / 1024**3:.1f}GB")
    print(f"🖥️ CPU cores: {psutil.cpu_count()}")

    # Check if ffmpeg is available
    try:
        result = subprocess.run(['ffmpeg', '-version'], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✅ FFmpeg is available")
        else:
            print("❌ FFmpeg check failed")
            print("Please install FFmpeg: brew install ffmpeg")
            exit(1)
    except FileNotFoundError:
        print("❌ FFmpeg not found")
        print("Please install FFmpeg: brew install ffmpeg")
        exit(1)
    except subprocess.TimeoutExpired:
        print("❌ FFmpeg check timed out")
        exit(1)

    root = tk.Tk()
    app = OpusClipApp(root)
    root.mainloop()
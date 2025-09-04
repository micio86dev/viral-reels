#!/usr/bin/env python3
"""
Viral Reels Generator - Versione Ottimizzata per Mac M1
Ottimizzazioni implementate:
- Lazy loading dei modelli AI
- Gestione memoria migliorata con context managers
- Streaming processing per video grandi
- Garbage collection esplicito
- Memory monitoring integrato
- Processing asincrono per UI responsiva
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

# Ottimizzazioni per Mac M1
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["OMP_NUM_THREADS"] = "4"  # Limita thread per evitare saturazione
os.environ["MKL_NUM_THREADS"] = "4"

Block = namedtuple("Block", "start end text score")

class MemoryMonitor:
    """Monitor memoria per prevenire saturazione RAM"""
    
    @staticmethod
    def get_memory_usage():
        """Ritorna uso memoria in MB"""
        process = psutil.Process()
        return process.memory_info().rss / 1024 / 1024
    
    @staticmethod
    def check_memory_limit(limit_mb=6000):
        """Verifica se siamo vicini al limite memoria"""
        current = MemoryMonitor.get_memory_usage()
        return current > limit_mb
    
    @staticmethod
    def force_cleanup():
        """Forza garbage collection e cleanup"""
        gc.collect()
        # Forza cleanup numpy
        if 'numpy' in globals():
            np.random.seed()

class ModelManager:
    """Gestione lazy loading dei modelli AI per ottimizzare memoria"""
    
    def __init__(self):
        self._whisper_model = None
        self._sentiment_analyzer = None
    
    @contextmanager
    def get_whisper_model(self):
        """Context manager per Whisper - carica solo quando serve"""
        if self._whisper_model is None:
            print("🤖 Loading Whisper model...")
            from faster_whisper import WhisperModel
            # Usa int8 per ridurre memoria su Mac M1
            self._whisper_model = WhisperModel(
                "small", 
                device="cpu", 
                compute_type="int8",
                num_workers=2  # Limita worker per Mac M1
            )
        
        try:
            yield self._whisper_model
        finally:
            # Non rilasciamo subito per riuso, ma monitoriamo memoria
            if MemoryMonitor.check_memory_limit():
                print("⚠️ Memory limit reached, releasing Whisper model")
                self._whisper_model = None
                MemoryMonitor.force_cleanup()
    
    @contextmanager
    def get_sentiment_analyzer(self):
        """Context manager per sentiment analyzer"""
        if self._sentiment_analyzer is None:
            print("🤖 Loading sentiment analyzer...")
            from transformers import pipeline
            self._sentiment_analyzer = pipeline(
                "sentiment-analysis",
                model="nlptown/bert-base-multilingual-uncased-sentiment",
                device=-1,  # Forza CPU per stabilità su Mac M1
                model_kwargs={"torch_dtype": "float16"}  # Riduce memoria
            )
        
        try:
            yield self._sentiment_analyzer
        finally:
            if MemoryMonitor.check_memory_limit():
                print("⚠️ Memory limit reached, releasing sentiment analyzer")
                self._sentiment_analyzer = None
                MemoryMonitor.force_cleanup()
    
    def cleanup_all(self):
        """Rilascia tutti i modelli"""
        self._whisper_model = None
        self._sentiment_analyzer = None
        MemoryMonitor.force_cleanup()

# Istanza globale del manager
model_manager = ModelManager()

def transcribe_with_streaming(video_path: str, chunk_duration: int = 300) -> List[Dict]:
    """Trascrizione con streaming per video lunghi"""
    
    # Ottieni durata video
    cmd = ["ffprobe", "-v", "quiet", "-show_entries", "format=duration", 
           "-of", "csv=p=0", str(video_path)]
    result = subprocess.run(cmd, capture_output=True, text=True)
    
    try:
        total_duration = float(result.stdout.strip())
    except:
        total_duration = chunk_duration  # Fallback
    
    all_segments = []
    
    # Processa in chunk per evitare saturazione memoria
    for start_time in range(0, int(total_duration), chunk_duration):
        end_time = min(start_time + chunk_duration, total_duration)
        
        print(f"🎵 Processing audio chunk {start_time}-{end_time}s...")
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            # Estrai chunk audio
            cmd = [
                "ffmpeg", "-y", "-ss", str(start_time), "-t", str(end_time - start_time),
                "-i", str(video_path), "-ar", "16000", "-ac", "1", "-vn", tmp.name
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
            
            # Trascrivi chunk
            with model_manager.get_whisper_model() as whisper_model:
                segments, _ = whisper_model.transcribe(tmp.name, language="it", vad_filter=True)
                
                for seg in segments:
                    all_segments.append({
                        "start": seg.start + start_time,  # Offset temporale
                        "end": seg.end + start_time,
                        "text": seg.text.strip()
                    })
        
        # Cleanup memoria dopo ogni chunk
        MemoryMonitor.force_cleanup()
        
        # Verifica memoria
        if MemoryMonitor.check_memory_limit():
            print("⚠️ Memory limit reached during transcription")
            break
    
    return all_segments

def audio_energy_optimized(video_path: str, start: float, dur: float) -> float:
    """Calcolo energia audio ottimizzato"""
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=True) as tmp:
            cmd = [
                "ffmpeg", "-y", "-ss", str(start), "-t", str(dur),
                "-i", str(video_path), "-ar", "8000", "-ac", "1", "-vn",  # Ridotta qualità per velocità
                tmp.name
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            import soundfile as sf
            data, _ = sf.read(tmp.name)
            
            if data is None or len(data) == 0:
                return 0.0
            
            # Calcolo RMS ottimizzato
            rms = float(np.sqrt(np.mean(data**2)))
            return min(rms * 10, 1.0)
            
    except Exception as e:
        print(f"⚠️ Audio energy calculation failed: {e}")
        return 0.0

def score_block_optimized(text: str, video_path: str, start: float, end: float) -> float:
    """Scoring ottimizzato con gestione memoria"""
    
    # Sentiment analysis con context manager
    try:
        with model_manager.get_sentiment_analyzer() as analyzer:
            sent = analyzer(text[:512])[0]  # Limita lunghezza testo
            pos = sent['score'] if sent['label'] == 'POSITIVE' else (1 - sent['score'])
    except Exception as e:
        print(f"⚠️ Sentiment analysis failed: {e}")
        pos = 0.5
    
    # Keywords scoring
    keywords = sum(1 for k in [
        "segreto", "incredibile", "pazzesco", "soldi", "viral", 
        "scopri", "trucco", "gratis", "shock", "mai visto"
    ] if k in text.lower())
    
    # Audio energy (solo per clip brevi per performance)
    duration = end - start
    if duration <= 60:
        energy = audio_energy_optimized(video_path, start, duration)
    else:
        energy = 0.5  # Valore neutro per clip lunghe
    
    # Length scoring ottimizzato
    length_score = 1.0 if 20 <= duration <= 60 else 0.5
    
    # Weighted scoring
    final_score = 0.4 * pos + 0.3 * min(keywords / 3, 1.0) + 0.2 * energy + 0.1 * length_score
    
    return final_score

def build_blocks_optimized(segments: List[Dict], target_dur: int = 45, overlap: float = 0.3) -> List[tuple]:
    """Costruzione blocchi ottimizzata con memory-aware processing"""
    
    if not segments:
        return []
    
    blocks = []
    i = 0
    
    while i < len(segments):
        start = segments[i]['start']
        text = segments[i]['text']
        j = i + 1
        end = segments[i]['end']
        
        # Costruisci blocco rispettando target duration
        while j < len(segments) and (end - start) < target_dur:
            end = segments[j]['end']
            text += ' ' + segments[j]['text']
            j += 1
        
        # Evita blocchi troppo corti
        if (end - start) >= 10:  # Minimo 10 secondi
            blocks.append((start, end, text))
        
        # Calcola prossimo indice con overlap
        step = max(1, int(j - i - overlap * (j - i)))
        i += step
        
        # Memory check periodico
        if len(blocks) % 10 == 0:
            MemoryMonitor.force_cleanup()
    
    return blocks

def analyse_video_optimized(input_file: str, target_dur: int = 45) -> List[Block]:
    """Analisi video ottimizzata per Mac M1"""
    
    print(f"🎬 Starting optimized analysis (Memory: {MemoryMonitor.get_memory_usage():.1f}MB)")
    
    try:
        # Trascrizione con streaming
        segments = transcribe_with_streaming(input_file)
        print(f"📝 Transcribed {len(segments)} segments")
        
        if not segments:
            print("⚠️ No segments found in transcription")
            return []
        
        # Costruzione blocchi
        blocks = build_blocks_optimized(segments, target_dur)
        print(f"🧩 Created {len(blocks)} blocks")
        
        if not blocks:
            print("⚠️ No valid blocks created")
            return []
        
        # Scoring con progress tracking
        scored_blocks = []
        for i, (start, end, text) in enumerate(blocks):
            if i % 5 == 0:
                print(f"📊 Scoring progress: {i+1}/{len(blocks)} (Memory: {MemoryMonitor.get_memory_usage():.1f}MB)")
            
            try:
                score = score_block_optimized(text, input_file, start, end)
                scored_blocks.append(Block(start, end, text, score))
            except Exception as e:
                print(f"⚠️ Scoring failed for block {i}: {e}")
                continue
            
            # Memory check
            if MemoryMonitor.check_memory_limit():
                print("⚠️ Memory limit reached during scoring, stopping early")
                break
        
        # Ordina per score
        result = sorted(scored_blocks, key=lambda b: b.score, reverse=True)
        print(f"✅ Analysis complete. Top score: {result[0].score:.3f}" if result else "⚠️ No scored blocks")
        
        return result
        
    except Exception as e:
        print(f"❌ Analysis failed: {e}")
        return []
    finally:
        # Cleanup finale
        model_manager.cleanup_all()
        print(f"🧹 Cleanup complete (Memory: {MemoryMonitor.get_memory_usage():.1f}MB)")

def create_srt_optimized(block: Block, out_path: str):
    """Creazione SRT ottimizzata"""
    lines = []
    words = block.text.split()
    
    if not words:
        return
    
    chunk_size = 6
    duration = block.end - block.start
    
    for i in range(0, len(words), chunk_size):
        chunk_words = words[i:i + chunk_size]
        
        # Calcolo timing proporzionale
        start_time = block.start + (i / len(words)) * duration
        end_time = block.start + ((i + chunk_size) / len(words)) * duration
        
        # Formato SRT
        start_srt = f"{int(start_time//3600):02d}:{int((start_time%3600)//60):02d}:{start_time%60:06.3f}".replace('.', ',')
        end_srt = f"{int(end_time//3600):02d}:{int((end_time%3600)//60):02d}:{end_time%60:06.3f}".replace('.', ',')
        
        lines.append(f"{len(lines)+1}\n{start_srt} --> {end_srt}\n{' '.join(chunk_words)}\n")
    
    # Scrivi file
    with open(out_path, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

def build_reel_optimized(input_file: str, block: Block, out_file: Path):
    """Costruzione reel ottimizzata con ffmpeg puro"""
    
    print(f"🎬 Building reel: {block.start:.1f}-{block.end:.1f}s")
    
    # File temporanei
    tmp_clip = out_file.parent / f"tmp_clip_{os.getpid()}.mp4"
    srt_file = out_file.parent / f"subs_{os.getpid()}.srt"
    
    try:
        # Step 1: Estrai e ridimensiona clip
        vf = "scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
        cmd1 = [
            "ffmpeg", "-y", "-ss", str(block.start), "-to", str(block.end),
            "-i", str(input_file), "-vf", vf, "-preset", "ultrafast",  # Più veloce
            "-crf", "28", "-c:a", "aac", "-b:a", "96k",  # Qualità ridotta per velocità
            str(tmp_clip)
        ]
        
        result = subprocess.run(cmd1, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg clip extraction failed: {result.stderr}")
        
        # Step 2: Crea sottotitoli
        shifted_block = Block(0, block.end - block.start, block.text, block.score)
        create_srt_optimized(shifted_block, srt_file)
        
        # Step 3: Aggiungi sottotitoli
        style = "force_style='FontName=Arial Bold,FontSize=26,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Shadow=1,Alignment=2,MarginV=80'"
        cmd2 = [
            "ffmpeg", "-y", "-i", str(tmp_clip), "-vf", f"subtitles={srt_file}:{style}",
            "-c:v", "libx264", "-preset", "ultrafast", "-crf", "28",
            "-c:a", "copy", str(out_file)
        ]
        
        result = subprocess.run(cmd2, capture_output=True, text=True)
        if result.returncode != 0:
            raise Exception(f"FFmpeg subtitle overlay failed: {result.stderr}")
        
        print(f"✅ Reel created: {out_file.name}")
        
    except Exception as e:
        print(f"❌ Reel creation failed: {e}")
        raise
    finally:
        # Cleanup file temporanei
        for temp_file in [tmp_clip, srt_file]:
            if temp_file.exists():
                temp_file.unlink()

class OptimizedApp:
    """Applicazione ottimizzata con UI asincrona"""
    
    def __init__(self, root):
        self.root = root
        self.is_processing = False
        self.setup_ui()
        
    def setup_ui(self):
        self.root.title("Viral Reels Generator - Optimized for Mac M1")
        self.root.geometry("800x600")
        
        # URL input
        ttk.Label(self.root, text="YouTube URL").pack(pady=5)
        self.url = tk.Entry(self.root, width=80)
        self.url.pack(pady=5)
        
        # Parameters frame
        params = ttk.LabelFrame(self.root, text="Parameters")
        params.pack(fill="x", padx=10, pady=10)
        
        # Duration
        ttk.Label(params, text="Max clip duration (sec):").grid(row=0, column=0, sticky="w")
        self.dur = tk.IntVar(value=45)  # Ridotto default per performance
        ttk.Spinbox(params, from_=15, to=90, textvariable=self.dur, width=8).grid(row=0, column=1)
        
        # Number of reels
        ttk.Label(params, text="Number of reels:").grid(row=1, column=0, sticky="w")
        self.n = tk.IntVar(value=3)
        ttk.Spinbox(params, from_=1, to=5, textvariable=self.n, width=8).grid(row=1, column=1)  # Ridotto max
        
        # Output directory
        ttk.Label(params, text="Output dir:").grid(row=2, column=0, sticky="w")
        self.outvar = tk.StringVar(value=str(Path.home() / "Desktop" / "ViralReels"))
        ttk.Entry(params, textvariable=self.outvar, width=50).grid(row=2, column=1)
        ttk.Button(params, text="Browse", command=self.choose_dir).grid(row=2, column=2)
        
        # Memory monitor
        self.memory_label = ttk.Label(params, text="Memory: 0 MB")
        self.memory_label.grid(row=3, column=0, columnspan=3, pady=5)
        
        # Generate button
        self.generate_btn = ttk.Button(self.root, text="🚀 Generate Optimized", command=self.threaded_run)
        self.generate_btn.pack(pady=15)
        
        # Progress bar
        self.progress = ttk.Progressbar(self.root, mode='indeterminate')
        self.progress.pack(fill="x", padx=10, pady=5)
        
        # Log
        self.log = tk.Text(self.root, height=15)
        self.log.pack(fill="both", expand=True, padx=10, pady=5)
        
        # Start memory monitoring
        self.update_memory_display()
    
    def update_memory_display(self):
        """Aggiorna display memoria ogni secondo"""
        memory_mb = MemoryMonitor.get_memory_usage()
        self.memory_label.config(text=f"Memory: {memory_mb:.1f} MB")
        
        # Colore warning se memoria alta
        if memory_mb > 4000:  # 4GB warning
            self.memory_label.config(foreground="red")
        elif memory_mb > 2000:  # 2GB caution
            self.memory_label.config(foreground="orange")
        else:
            self.memory_label.config(foreground="green")
        
        self.root.after(1000, self.update_memory_display)
    
    def choose_dir(self):
        d = filedialog.askdirectory()
        if d:
            self.outvar.set(d)
    
    def logwrite(self, txt):
        """Thread-safe logging"""
        def _write():
            self.log.insert("end", txt + "\n")
            self.log.see("end")
            self.root.update()
        
        if threading.current_thread() == threading.main_thread():
            _write()
        else:
            self.root.after(0, _write)
    
    def threaded_run(self):
        """Avvia processing in thread separato"""
        if self.is_processing:
            self.logwrite("⚠️ Processing already in progress...")
            return
        
        self.is_processing = True
        self.generate_btn.config(state="disabled")
        self.progress.start()
        
        thread = threading.Thread(target=self.generate, daemon=True)
        thread.start()
    
    def generate(self):
        """Generazione ottimizzata"""
        try:
            url = self.url.get().strip()
            if not url:
                self.logwrite("❌ Please enter a YouTube URL")
                return
            
            self.logwrite("🚀 Starting optimized generation...")
            self.logwrite(f"💾 Initial memory: {MemoryMonitor.get_memory_usage():.1f}MB")
            
            with tempfile.TemporaryDirectory() as td:
                # Download con qualità ridotta per Mac M1
                self.logwrite("📥 Downloading video (optimized quality)...")
                ydl_opts = {
                    "outtmpl": f"{td}/%(title)s.%(ext)s",
                    "format": "best[height<=480]",  # Qualità ridotta per performance
                    "writesubtitles": False,
                    "writeautomaticsub": False
                }
                
                with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                    info = ydl.extract_info(url, download=True)
                    title = info.get('title', 'video')
                
                video_file = next(Path(td).glob('*'))
                self.logwrite(f"📁 Downloaded: {video_file.name}")
                self.logwrite(f"💾 Memory after download: {MemoryMonitor.get_memory_usage():.1f}MB")
                
                # Analisi ottimizzata
                self.logwrite("🤖 Starting optimized analysis...")
                blocks = analyse_video_optimized(str(video_file), self.dur.get())
                
                if not blocks:
                    self.logwrite("❌ No suitable blocks found")
                    return
                
                self.logwrite(f"✅ Found {len(blocks)} blocks")
                self.logwrite(f"💾 Memory after analysis: {MemoryMonitor.get_memory_usage():.1f}MB")
                
                # Creazione output
                outdir = Path(self.outvar.get())
                outdir.mkdir(exist_ok=True, parents=True)
                
                # Genera reel
                num_reels = min(self.n.get(), len(blocks))
                for i, block in enumerate(blocks[:num_reels], 1):
                    safe_title = re.sub(r'[^\w\s-]', '', title)[:30]  # Titolo più corto
                    out_file = outdir / f"{safe_title}_reel{i}.mp4"
                    
                    self.logwrite(f"✂️ Creating reel {i}/{num_reels}: {block.start:.1f}-{block.end:.1f}s (Score: {block.score:.3f})")
                    
                    try:
                        build_reel_optimized(str(video_file), block, out_file)
                        self.logwrite(f"✅ Reel {i} completed: {out_file.name}")
                    except Exception as e:
                        self.logwrite(f"❌ Reel {i} failed: {e}")
                        continue
                    
                    # Memory check tra reel
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
            # Cleanup finale
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

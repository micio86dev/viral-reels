#!/usr/bin/env python3
import os, tempfile, subprocess, re, threading
from pathlib import Path
from collections import namedtuple
import tkinter as tk
from tkinter import ttk, filedialog
import yt_dlp
import moviepy.editor as mp
from faster_whisper import WhisperModel
from transformers import pipeline
import numpy as np

# Disabilita parallelismo tokenizers per evitare warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

Block = namedtuple("Block", "start end text score")

def transcribe_full(model, video_path):
    """Trascrivi l'intero video con segmenti timestamp."""
    with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
        cmd = ["ffmpeg","-y","-i",video_path,"-ar","16000","-ac","1","-vn",tmp.name]
        subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,check=True)
        segments, _ = model.transcribe(tmp.name, language="it", vad_filter=True)
        result = []
        for seg in segments:
            result.append({
                "start": seg.start,
                "end": seg.end,
                "text": seg.text.strip()
            })
        return result

def audio_energy_safe(video_path, start, dur):
    try:
        with tempfile.NamedTemporaryFile(suffix=".wav") as tmp:
            cmd = ["ffmpeg","-y","-ss",str(start),"-t",str(dur),
                   "-i",video_path,"-ar","16000","-ac","1","-vn",tmp.name]
            subprocess.run(cmd,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
            import soundfile as sf
            data, _ = sf.read(tmp.name)
            if data is None or len(data)==0:
                return 0.0
            rms = float(np.sqrt((data**2).mean()))
            return min(rms*10,1.0)
    except:
        return 0.0

def score_block(analyser, text, video_path, start, end):
    try:
        sent = analyser(text)[0]
        pos = sent['score'] if sent['label']=='POSITIVE' else (1-sent['score'])
    except:
        pos=0.5
    keywords = sum(1 for k in ["segreto","incredibile","pazzesco","soldi","viral","scopri","trucco","gratis"] if k in text.lower())
    energy= audio_energy_safe(video_path,start,end-start)
    length_score = 1.0 if 20<= (end-start) <=60 else 0.5
    return 0.4*pos + 0.3*min(keywords/3,1.0) + 0.2*energy + 0.1*length_score

def build_blocks(segments,target_dur=45,overlap=0.3):
    """Raggruppa frasi consecutive in blocchi coerenti verso target_duration"""
    blocks=[]
    i=0
    while i < len(segments):
        start=segments[i]['start']
        text=segments[i]['text']
        j=i+1
        end=segments[i]['end']
        while j<len(segments) and (end-start)<target_dur:
            end=segments[j]['end']
            text += ' '+segments[j]['text']
            j+=1
        blocks.append((start,end,text))
        i=int(j - overlap*(j-i))  # sovrapponi un po' per diversità
    return blocks

def analyse_video(input_file, target_dur=45):
    print("🤖 Init models...")
    whisper=WhisperModel("small",device="cpu",compute_type="int8")
    analyser=pipeline("sentiment-analysis",model="nlptown/bert-base-multilingual-uncased-sentiment")
    segs=transcribe_full(whisper,input_file)
    blocks=build_blocks(segs,target_dur)
    scored=[]
    for s,e,t in blocks:
        sc=score_block(analyser,t,input_file,s,e)
        scored.append(Block(s,e,t,sc))
    return sorted(scored,key=lambda b:b.score,reverse=True)

def create_srt(block,out_path):
    lines=[]
    words=block.text.split()
    chunk=6
    dur=block.end-block.start
    for i in range(0,len(words),chunk):
        chunk_words=words[i:i+chunk]
        st=block.start+(i/len(words))*dur
        et=block.start+((i+chunk)/len(words))*dur
        start_srt=f"{int(st//3600):02d}:{int((st%3600)//60):02d}:{st%60:06.3f}".replace('.',',')
        end_srt=f"{int(et//3600):02d}:{int((et%3600)//60):02d}:{et%60:06.3f}".replace('.',',')
        lines.append(f"{len(lines)+1}\n{start_srt} --> {end_srt}\n{' '.join(chunk_words)}\n")
    with open(out_path,'w',encoding='utf-8') as f: f.write('\n'.join(lines))

def build_reel(input_file, block, out_file):
    vf="scale=1080:1920:force_original_aspect_ratio=increase,crop=1080:1920"
    tmp=out_file.parent/"tmp_clip.mp4"
    cmd=["ffmpeg","-y","-ss",str(block.start),"-to",str(block.end),"-i",str(input_file),"-vf",vf,"-preset","fast","-crf","23","-c:a","aac","-b:a","128k",str(tmp)]
    subprocess.run(cmd,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    srt=out_file.parent/"subs.srt"
    shifted=Block(0,block.end-block.start,block.text,block.score)
    create_srt(shifted,srt)
    style="force_style='FontName=Arial Bold,FontSize=28,PrimaryColour=&Hffffff,OutlineColour=&H000000,Outline=2,Shadow=1,Alignment=2,MarginV=80'"
    cmd2=["ffmpeg","-y","-i",str(tmp),"-vf",f"subtitles={srt}:{style}","-c:v","libx264","-preset","fast","-crf","23","-c:a","copy",str(out_file)]
    subprocess.run(cmd2,check=True,stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
    tmp.unlink(); srt.unlink()

class App:
    def __init__(self,root):
        self.root=root
        root.title("Viral Reels Generator - OpusClip style")
        root.geometry("750x550")
        ttk.Label(root,text="YouTube URL").pack(pady=5)
        self.url=tk.Entry(root,width=80); self.url.pack(pady=5)
        params=ttk.LabelFrame(root,text="Parameters"); params.pack(fill="x",padx=10,pady=10)
        ttk.Label(params,text="Max clip duration (sec):").grid(row=0,column=0,sticky="w")
        self.dur=tk.IntVar(value=60)
        ttk.Spinbox(params,from_=15,to=90,textvariable=self.dur,width=8).grid(row=0,column=1)
        ttk.Label(params,text="Number of reels:").grid(row=1,column=0,sticky="w")
        self.n=tk.IntVar(value=3)
        ttk.Spinbox(params,from_=1,to=8,textvariable=self.n,width=8).grid(row=1,column=1)
        ttk.Label(params,text="Output dir:").grid(row=2,column=0,sticky="w")
        self.outvar=tk.StringVar(value=str(Path.home()/"Desktop"/"ViralReels"))
        ttk.Entry(params,textvariable=self.outvar,width=50).grid(row=2,column=1)
        ttk.Button(params,text="Browse",command=self.choose_dir).grid(row=2,column=2)
        ttk.Button(root,text="🚀 Generate",command=self.threaded_run).pack(pady=15)
        self.log=tk.Text(root,height=15); self.log.pack(fill="both",expand=True,padx=10,pady=5)

    def choose_dir(self):
        d=filedialog.askdirectory()
        if d: self.outvar.set(d)
    def logwrite(self,txt):
        self.log.insert("end",txt+"\n"); self.log.see("end"); self.root.update()
    def threaded_run(self): threading.Thread(target=self.generate,daemon=True).start()

    def generate(self):
        url=self.url.get().strip()
        if not url: return
        with tempfile.TemporaryDirectory() as td:
            self.logwrite("📥 Download video...")
            ydl_opts={"outtmpl":f"{td}/%(title)s.%(ext)s","format":"best[height<=720]"}
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info=ydl.extract_info(url,download=True)
                title=info.get('title','video')
            video_file=next(Path(td).glob('*'))
            self.logwrite("🤖 Analysing video...")
            blocks=analyse_video(video_file,self.dur.get())
            outdir=Path(self.outvar.get()); outdir.mkdir(exist_ok=True,parents=True)
            for i,block in enumerate(blocks[:self.n.get()],1):
                safe_title = re.sub(r'[^\w\s-]', '', title)[:40]
                out = outdir / f"{safe_title}_reel{i}.mp4"
                self.logwrite(f"✂️ Reel {i}: {block.start:.1f}-{block.end:.1f}s Score {block.score:.2f}")
                build_reel(video_file,block,out)
            self.logwrite(f"✅ Done! Saved in {outdir}")

if __name__=='__main__':
    root=tk.Tk(); App(root); root.mainloop()

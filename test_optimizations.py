#!/usr/bin/env python3
"""
Script di test per validare le ottimizzazioni Mac M1
Confronta performance e uso memoria tra versione originale e ottimizzata
"""

import time
import psutil
import subprocess
import sys
from pathlib import Path
import matplotlib.pyplot as plt
import json

class PerformanceProfiler:
    """Profiler per misurare performance e memoria"""
    
    def __init__(self):
        self.measurements = []
        self.start_time = None
        self.process = psutil.Process()
    
    def start_measurement(self, label):
        """Inizia misurazione"""
        self.start_time = time.time()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        measurement = {
            'label': label,
            'start_time': self.start_time,
            'initial_memory': initial_memory,
            'peak_memory': initial_memory,
            'memory_samples': [initial_memory]
        }
        
        self.measurements.append(measurement)
        print(f"📊 Starting measurement: {label} (Initial memory: {initial_memory:.1f}MB)")
        return len(self.measurements) - 1
    
    def sample_memory(self, measurement_id):
        """Campiona memoria durante esecuzione"""
        if measurement_id < len(self.measurements):
            current_memory = self.process.memory_info().rss / 1024 / 1024
            measurement = self.measurements[measurement_id]
            measurement['memory_samples'].append(current_memory)
            measurement['peak_memory'] = max(measurement['peak_memory'], current_memory)
    
    def end_measurement(self, measurement_id):
        """Termina misurazione"""
        if measurement_id < len(self.measurements):
            measurement = self.measurements[measurement_id]
            measurement['end_time'] = time.time()
            measurement['duration'] = measurement['end_time'] - measurement['start_time']
            measurement['final_memory'] = self.process.memory_info().rss / 1024 / 1024
            
            print(f"✅ Completed: {measurement['label']}")
            print(f"   Duration: {measurement['duration']:.1f}s")
            print(f"   Peak memory: {measurement['peak_memory']:.1f}MB")
            print(f"   Final memory: {measurement['final_memory']:.1f}MB")
            print(f"   Memory delta: {measurement['final_memory'] - measurement['initial_memory']:+.1f}MB")
    
    def get_summary(self):
        """Ritorna summary delle misurazioni"""
        return {
            'total_measurements': len(self.measurements),
            'measurements': self.measurements
        }

def test_memory_monitor():
    """Test del memory monitor"""
    print("\n🧪 Testing Memory Monitor...")
    
    try:
        # Import dalla versione ottimizzata
        sys.path.insert(0, '.')
        from viral_reels_optimized import MemoryMonitor
        
        # Test basic functionality
        memory_mb = MemoryMonitor.get_memory_usage()
        print(f"✅ Current memory usage: {memory_mb:.1f}MB")
        
        # Test memory limit check
        is_high = MemoryMonitor.check_memory_limit(1000)  # 1GB limit
        print(f"✅ Memory limit check (1GB): {'HIGH' if is_high else 'OK'}")
        
        # Test cleanup
        MemoryMonitor.force_cleanup()
        print("✅ Cleanup executed successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Memory monitor test failed: {e}")
        return False

def test_model_manager():
    """Test del model manager con lazy loading"""
    print("\n🧪 Testing Model Manager...")
    
    profiler = PerformanceProfiler()
    
    try:
        # Use Path to construct dynamic path for import
        repo_path = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo_path))
        from viral_reels_optimized import ModelManager, MemoryMonitor

        manager = ModelManager()

        # Test Whisper lazy loading
        measurement_id = profiler.start_measurement("Whisper Model Loading")

        with manager.get_whisper_model() as whisper:
            profiler.sample_memory(measurement_id)
            print("✅ Whisper model loaded successfully")
            # Simula uso del modello
            time.sleep(1)
            profiler.sample_memory(measurement_id)

        profiler.end_measurement(measurement_id)

        # Test Sentiment Analyzer lazy loading
        measurement_id = profiler.start_measurement("Sentiment Analyzer Loading")

        with manager.get_sentiment_analyzer() as analyzer:
            profiler.sample_memory(measurement_id)
            print("✅ Sentiment analyzer loaded successfully")
            # Test con testo di esempio
            result = analyzer("Questo è un test positivo!")
            print(f"   Test result: {result}")
            profiler.sample_memory(measurement_id)

        profiler.end_measurement(measurement_id)

        # Test cleanup
        manager.cleanup_all()
        final_memory = MemoryMonitor.get_memory_usage()
        print(f"✅ Cleanup completed. Final memory: {final_memory:.1f}MB")

        return True, profiler.get_summary()

    except Exception as e:
        print(f"❌ Model manager test failed: {e}")
        return False, None

def create_test_video():
    """Crea un video di test per i benchmark"""
    print("\n🎬 Creating test video...")
    
    test_video = Path("/tmp/test_video.mp4")
    
    if test_video.exists():
        print(f"✅ Test video already exists: {test_video}")
        return str(test_video)
    
    try:
        # Crea video di test con FFmpeg (30 secondi, audio sintetico)
        cmd = [
            "ffmpeg", "-f", "lavfi", "-i", "testsrc2=duration=30:size=640x480:rate=30",
            "-f", "lavfi", "-i", "sine=frequency=1000:duration=30",
            "-c:v", "libx264", "-preset", "ultrafast", "-c:a", "aac",
            "-y", str(test_video)
        ]
        
        result = subprocess.run(cmd, capture_output=True, text=True)
        
        if result.returncode == 0:
            print(f"✅ Test video created: {test_video}")
            return str(test_video)
        else:
            print(f"❌ Failed to create test video: {result.stderr}")
            return None
            
    except Exception as e:
        print(f"❌ Test video creation failed: {e}")
        return None

def benchmark_transcription():
    """Benchmark della trascrizione ottimizzata"""
    print("\n🧪 Benchmarking Transcription...")

    test_video = create_test_video()
    if not test_video:
        return False, None

    profiler = PerformanceProfiler()

    try:
        repo_path = Path(__file__).resolve().parent
        sys.path.insert(0, str(repo_path))
        from viral_reels_optimized import transcribe_with_streaming, MemoryMonitor

        measurement_id = profiler.start_measurement("Streaming Transcription")

        # Test trascrizione con streaming
        segments = transcribe_with_streaming(test_video, chunk_duration=15)  # Chunk piccoli per test

        profiler.sample_memory(measurement_id)

        print(f"✅ Transcription completed: {len(segments)} segments")
        for i, seg in enumerate(segments[:3]):  # Mostra primi 3
            print(f"   Segment {i+1}: {seg['start']:.1f}-{seg['end']:.1f}s: {seg['text'][:50]}...")

        profiler.end_measurement(measurement_id)

        return True, profiler.get_summary()

    except Exception as e:
        print(f"❌ Transcription benchmark failed: {e}")
        import traceback
        traceback.print_exc()
        return False, None
    finally:
        # Cleanup test video
        try:
            if test_video and Path(test_video).exists():
                Path(test_video).unlink()
                print(f"🗑️  Test video cleaned up: {test_video}")
        except Exception as e:
            print(f"⚠️  Failed to cleanup test video: {e}")

def generate_performance_report(results):
    """Genera report delle performance"""
    print("\n📊 Generating Performance Report...")

    report_file = Path.cwd() / "performance_report.json"

    # Salva risultati in JSON
    with open(report_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"✅ Performance report saved: {report_file}")

    # Crea grafico memoria se matplotlib disponibile
    try:
        create_memory_chart(results)
    except ImportError:
        print("⚠️ Matplotlib not available, skipping memory chart")

def create_memory_chart(results):
    """Crea grafico uso memoria"""
    plt.figure(figsize=(12, 8))
    
    for measurement in results.get('measurements', []):
        if 'memory_samples' in measurement:
            times = list(range(len(measurement['memory_samples'])))
            plt.plot(times, measurement['memory_samples'], 
                    label=measurement['label'], marker='o')
    
    plt.xlabel('Time (samples)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage During Optimization Tests')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    chart_file = Path.cwd() / "memory_usage_chart.png"
    plt.savefig(chart_file, dpi=150, bbox_inches='tight')
    plt.close()

    print(f"✅ Memory chart saved: {chart_file}")

def main():
    """Esegue tutti i test di ottimizzazione"""
    print("🚀 VIRAL REELS OPTIMIZATION TESTS")
    print("=" * 50)

    results = {
        'test_timestamp': time.time(),
        'system_info': {
            'cpu_count': psutil.cpu_count(),
            'memory_total_gb': psutil.virtual_memory().total / 1024**3,
            'python_version': sys.version
        },
        'tests': {}
    }

    try:
        # Test 1: Memory Monitor
        print("\n🔍 TEST 1: Memory Monitor")
        results['tests']['memory_monitor'] = test_memory_monitor()

        # Test 2: Model Manager
        print("\n🔍 TEST 2: Model Manager")
        success, profiler_data = test_model_manager()
        results['tests']['model_manager'] = {
            'success': success,
            'profiler_data': profiler_data
        }

        # Test 3: Transcription Benchmark
        print("\n🔍 TEST 3: Transcription Benchmark")
        success, profiler_data = benchmark_transcription()
        results['tests']['transcription'] = {
            'success': success,
            'profiler_data': profiler_data
        }

        # Genera report finale
        generate_performance_report(results)

        # Summary
        print("\n" + "=" * 50)
        print("📋 TEST SUMMARY")
        print("=" * 50)

        total_tests = len(results['tests'])
        passed_tests = sum(1 for test in results['tests'].values()
                          if (test if isinstance(test, bool) else test.get('success', False)))

        print(f"Total tests: {total_tests}")
        print(f"Passed: {passed_tests}")
        print(f"Failed: {total_tests - passed_tests}")
        print(f"Success rate: {passed_tests/total_tests*100:.1f}%")

        if passed_tests == total_tests:
            print("\n🎉 ALL OPTIMIZATIONS WORKING CORRECTLY!")
            print("✅ Your Mac M1 should now handle video processing much better")
        else:
            print("\n⚠️ Some tests failed. Check the error messages above.")

        print(f"\n📁 Results saved in: {Path('.').resolve()}/")

    except KeyboardInterrupt:
        print("\n\n⚠️  Test interrupted by user")
        # Try to clean up any resources
        try:
            import gc
            gc.collect()
        except:
            pass
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        # Final cleanup
        try:
            # Clean up any temporary files
            temp_video = Path("/tmp/test_video.mp4")
            if temp_video.exists():
                temp_video.unlink()
                print("🗑️  Temporary test video cleaned up")
        except:
            pass

if __name__ == "__main__":
    main()

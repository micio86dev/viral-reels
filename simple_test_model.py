#!/usr/bin/env python3
"""
Simple test script for ModelManager
"""

import sys
import time
from pathlib import Path

class SimpleProfiler:
    """Simple profiler for testing"""
    
    def __init__(self):
        self.measurements = []
        
    def start_measurement(self, name):
        import psutil
        measurement_id = len(self.measurements)
        self.measurements.append({
            'name': name,
            'start_time': time.time(),
            'memory_samples': []
        })
        return measurement_id
        
    def sample_memory(self, measurement_id):
        import psutil
        process = psutil.Process()
        memory_mb = process.memory_info().rss / 1024 / 1024
        self.measurements[measurement_id]['memory_samples'].append(memory_mb)
        
    def end_measurement(self, measurement_id):
        self.measurements[measurement_id]['end_time'] = time.time()
        
    def get_summary(self):
        return self.measurements

def test_model_manager():
    """Test just the model manager functionality"""
    print("\n🧪 Testing Model Manager...")
    
    profiler = SimpleProfiler()
    
    try:
        sys.path.insert(0, '.')
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
        import traceback
        traceback.print_exc()
        return False, None

if __name__ == "__main__":
    success, results = test_model_manager()
    if success:
        print("\n🎉 Model manager test passed!")
        print(f"Results: {len(results) if results else 0} measurements")
    else:
        print("\n⚠️ Model manager test failed!")
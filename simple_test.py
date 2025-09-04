#!/usr/bin/env python3
"""
Simple test script to isolate multiprocessing issues
"""

import sys
from pathlib import Path

def test_memory_monitor():
    """Test just the memory monitor functionality"""
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

if __name__ == "__main__":
    success = test_memory_monitor()
    if success:
        print("\n🎉 Memory monitor test passed!")
    else:
        print("\n⚠️ Memory monitor test failed!")
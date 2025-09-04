#!/usr/bin/env python3
"""
Test script to verify that all functions work correctly
"""

import sys
import os
sys.path.append('.')

from viral_reels import get_video_id, transcribe_with_word_timestamps, segment_into_sentences, select_best_sentences

def test_get_video_id():
    """Test video ID extraction"""
    print("🧪 Testing get_video_id...")

    test_urls = [
        "https://www.youtube.com/watch?v=Yv0ekdimoXA",
        "https://youtu.be/Yv0ekdimoXA",
        "https://youtube.com/watch?v=Yv0ekdimoXA&t=30"
    ]

    for url in test_urls:
        try:
            video_id = get_video_id(url)
            print(f"✅ {url} -> {video_id}")
        except Exception as e:
            print(f"❌ {url} -> Error: {e}")

def test_transcribe_function():
    """Test if transcribe function can be called (without actually transcribing)"""
    print("\n🧪 Testing transcribe_with_word_timestamps function availability...")

    try:
        # Just check if function exists and can be called with wrong parameters
        # This will fail but should give us import errors if any
        transcribe_with_word_timestamps("nonexistent.mp4")
    except FileNotFoundError:
        print("✅ transcribe_with_word_timestamps function is available")
    except Exception as e:
        print(f"❌ transcribe_with_word_timestamps error: {e}")

def test_segment_function():
    """Test if segment function can be called"""
    print("\n🧪 Testing segment_into_sentences function availability...")

    try:
        # Test with empty list
        result = segment_into_sentences([])
        print(f"✅ segment_into_sentences([]) -> {len(result)} sentences")
    except Exception as e:
        print(f"❌ segment_into_sentences error: {e}")

def test_select_function():
    """Test if select function can be called"""
    print("\n🧪 Testing select_best_sentences function availability...")

    try:
        # Test with empty list
        result = select_best_sentences([], 30, 5)
        print(f"✅ select_best_sentences([], 30, 5) -> {len(result)} sentences")
    except Exception as e:
        print(f"❌ select_best_sentences error: {e}")

if __name__ == "__main__":
    print("🚀 Starting function tests...\n")

    test_get_video_id()
    test_transcribe_function()
    test_segment_function()
    test_select_function()

    print("\n✅ All tests completed!")
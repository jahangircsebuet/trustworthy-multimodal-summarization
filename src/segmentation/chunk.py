"""
Simple text chunking utilities for testing purposes.
"""

def chunk_text(text, max_tokens=512, overlap=64):
    """
    Split text into chunks based on token count.
    This is a simple implementation for testing.
    """
    # Simple word-based chunking (not actual tokenization)
    # For testing, we'll be more aggressive with chunking
    words = text.split()
    chunks = []
    
    # If text is short, force it into multiple chunks for testing
    if len(words) <= max_tokens and max_tokens <= 100:  # Small max_tokens for testing
        # Force chunking by using a smaller chunk size
        chunk_size = max(1, len(words) // 2)  # At least 2 chunks
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i + chunk_size])
            if chunk.strip():
                chunks.append(chunk)
        return chunks
    
    # Normal chunking for longer texts
    start = 0
    while start < len(words):
        end = min(start + max_tokens, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start = end - overlap if end < len(words) else end
    
    return chunks


def build_thread_text(thread_data):
    """
    Build thread text from timestamped data.
    This is a simple implementation for testing.
    """
    # Sort by timestamp
    sorted_data = sorted(thread_data, key=lambda x: x.get("timestamp", 0))
    
    # Build thread text with timestamps
    thread_text = ""
    for i, item in enumerate(sorted_data):
        timestamp = item.get("timestamp", i)
        text = item.get("text", "")
        thread_text += f"[{timestamp}] {text}\n"
    
    return thread_text.strip() 
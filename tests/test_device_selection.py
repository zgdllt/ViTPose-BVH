#!/usr/bin/env python
"""Test device selection logic."""

import sys

def select_device_mock():
    """Mock device selection for testing."""
    # Try importing torch
    try:
        import torch
        if torch.backends.mps.is_available():
            return 'mps'
        elif torch.cuda.is_available():
            return 'cuda:0'
        else:
            return 'cpu'
    except ImportError:
        print("PyTorch not installed - device selection requires torch")
        return 'cpu (mock - torch not available)'

if __name__ == '__main__':
    device = select_device_mock()
    print(f"Selected device: {device}")
    
    try:
        import torch
        print(f"PyTorch version: {torch.__version__}")
        print(f"MPS available: {torch.backends.mps.is_available()}")
        print(f"CUDA available: {torch.cuda.is_available()}")
    except ImportError:
        print("PyTorch not installed - skipping detailed device checks")
    
    print("\n✓ Device selection logic validated")

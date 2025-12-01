"""
MusicGen Model Cache with CPU/GPU Swapping
Keeps models in CPU RAM and swaps to GPU only when needed
"""

import torch
import gc
from typing import Optional, Dict
from threading import Lock
from audiocraft.models import MusicGen
import time

class MusicGenModelCache:
    """
    Cache for MusicGen models with CPU/GPU swapping.
    
    Instead of deleting models, we:
    1. Keep them in CPU RAM when not in use
    2. Swap to GPU only when generating
    3. Swap back to CPU when done
    
    This trades GPU memory for CPU memory and makes model loading instant!
    """
    
    def __init__(self, max_models: int = 3):
        """
        Args:
            max_models: Maximum number of models to keep in CPU cache
        """
        self._cache: Dict[str, MusicGen] = {}
        self._lock = Lock()
        self._max_models = max_models
        self._access_order = []  # Track LRU
        
    def _move_model_to_cpu(self, model: MusicGen):
        """Move all model components to CPU."""
        print("  📤 Moving model to CPU...")
        start = time.time()
        
        # Move language model (the big transformer)
        if hasattr(model, 'lm') and model.lm is not None:
            model.lm = model.lm.cpu()
        
        # Move compression model (EnCodec)
        if hasattr(model, 'compression_model') and model.compression_model is not None:
            model.compression_model = model.compression_model.cpu()
        
        # Clear GPU cache
        torch.cuda.empty_cache()
        
        elapsed = time.time() - start
        print(f"  ✅ Model on CPU (took {elapsed:.2f}s)")
        
    def _move_model_to_gpu(self, model: MusicGen, device: str = "cuda"):
        """Move all model components to GPU."""
        print("  📥 Moving model to GPU...")
        start = time.time()
        
        # Move language model
        if hasattr(model, 'lm') and model.lm is not None:
            model.lm = model.lm.to(device)
        
        # Move compression model
        if hasattr(model, 'compression_model') and model.compression_model is not None:
            model.compression_model = model.compression_model.to(device)
        
        elapsed = time.time() - start
        print(f"  ✅ Model on GPU (took {elapsed:.2f}s)")
        
    def _evict_lru_if_needed(self):
        """Evict least recently used model if cache is full."""
        if len(self._cache) >= self._max_models and self._access_order:
            lru_model_name = self._access_order[0]
            print(f"  🗑️  Cache full, evicting LRU model: {lru_model_name}")
            
            # Actually delete the LRU model
            if lru_model_name in self._cache:
                del self._cache[lru_model_name]
                self._access_order.remove(lru_model_name)
                
            # Cleanup
            gc.collect()
            torch.cuda.empty_cache()
    
    def get_model(self, model_name: str, device: str = "cuda") -> MusicGen:
        """
        Get a model, loading from cache or HuggingFace.
        Returns model on GPU, ready to use.
        """
        with self._lock:
            # Check if model is in cache
            if model_name in self._cache:
                print(f"🎯 Model '{model_name}' found in cache!")
                model = self._cache[model_name]
                
                # Update LRU
                if model_name in self._access_order:
                    self._access_order.remove(model_name)
                self._access_order.append(model_name)
                
                # Move to GPU
                self._move_model_to_gpu(model, device)
                return model
            
            # Model not in cache - need to load
            print(f"📦 Loading model '{model_name}' from HuggingFace...")
            start = time.time()
            
            # Evict LRU if cache is full
            self._evict_lru_if_needed()
            
            # Load model on GPU initially
            model = MusicGen.get_pretrained(model_name, device=device)
            
            elapsed = time.time() - start
            print(f"✅ Model loaded (took {elapsed:.2f}s)")
            
            # Add to cache
            self._cache[model_name] = model
            self._access_order.append(model_name)
            
            return model
    
    def return_model(self, model_name: str):
        """
        Return model to cache after use.
        Moves it to CPU to free GPU memory.
        """
        with self._lock:
            if model_name in self._cache:
                print(f"💾 Returning model '{model_name}' to cache...")
                model = self._cache[model_name]
                self._move_model_to_cpu(model)
    
    def clear_cache(self):
        """Clear all cached models."""
        with self._lock:
            print("🗑️  Clearing model cache...")
            self._cache.clear()
            self._access_order.clear()
            gc.collect()
            torch.cuda.empty_cache()
    
    def get_cache_info(self) -> dict:
        """Get information about cached models."""
        with self._lock:
            return {
                'cached_models': list(self._cache.keys()),
                'cache_size': len(self._cache),
                'max_size': self._max_models,
                'access_order': self._access_order.copy()
            }


# Global model cache instance
_model_cache = MusicGenModelCache(max_models=3)


def get_cached_model(model_name: str, device: str = "cuda") -> MusicGen:
    """
    Get a MusicGen model from cache or load it.
    Model is returned on GPU, ready to use.
    
    Usage:
        model = get_cached_model("thepatch/vanya_ai_dnb_0.1")
        # ... use model ...
        return_cached_model("thepatch/vanya_ai_dnb_0.1")
    """
    return _model_cache.get_model(model_name, device)


def return_cached_model(model_name: str):
    """
    Return a model to cache after use.
    Moves it to CPU to free GPU memory.
    """
    _model_cache.return_model(model_name)


def clear_model_cache():
    """Clear all cached models."""
    _model_cache.clear_cache()


def get_model_cache_info() -> dict:
    """Get information about the model cache."""
    return _model_cache.get_cache_info()


# Context manager for automatic model return
from contextlib import contextmanager

@contextmanager
def cached_model(model_name: str, device: str = "cuda"):
    """
    Context manager for using a cached model.
    Automatically returns model to cache (CPU) when done.
    
    Usage:
        with cached_model("thepatch/vanya_ai_dnb_0.1") as model:
            output = model.generate(...)
        # Model automatically moved to CPU here
    """
    model = get_cached_model(model_name, device)
    try:
        yield model
    finally:
        return_cached_model(model_name)
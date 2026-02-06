import numpy as np

def rms_level(samples: np.ndarray) -> float:
    """samples: int16 numpy array"""
    if len(samples) == 0:
        return 0.0
    samples = samples.astype(np.float32)
    rms = np.sqrt(np.mean(samples ** 2))
    return min(rms / 32768.0, 1.0)  # normalize 0–1

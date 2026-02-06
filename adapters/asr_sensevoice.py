# AImy/adapters/asr_sensevoice.py
import os, sys, time
import numpy as np
from pathlib import Path
import config


class SenseVoiceAdapter:
    """
    Thin adapter around SenseVoiceAx + tokenizer.
    Exposes .infer_audio(audio_f32) -> str
    """
    def __init__(self):
        self._tok = None
        self._asr = None
        self._post = None

    def init_asr(self):
        """
        Initialize the SenseVoice ASR model and tokenizer.
        """
        # Ensure the SENSEVOICE ROOT (that contains `utils/`) is on sys.path
        sv_root = Path(config.SENSEVOICE_DIR).resolve()
        sv_root_str = str(sv_root)
        if sv_root_str not in sys.path:
            sys.path.insert(0, sv_root_str)

        from utils.SenseVoiceAx import SenseVoiceAx
        from utils.tokenizer import SentencepiecesTokenizer
        from utils.print_utils import rich_transcription_postprocess

        # Convert Paths -> str for third-party libs
        model_path = str(Path(config.SENSEVOICE_MODEL_PATH))
        bpe_path   = str(Path(config.SENSEVOICE_BPE_PATH))

        # Initialize tokenizer and model
        self._tok = SentencepiecesTokenizer(bpemodel=bpe_path)
        self._asr = SenseVoiceAx(
            model_path,
            max_len=256,
            language="auto",
            use_itn=True,
            tokenizer=self._tok
        )

        # Postprocessing function
        self._post = rich_transcription_postprocess
        print("[ASR] SenseVoice initialized successfully.")

    def infer_audio(self, audio_f32: np.ndarray) -> str:
        """
        audio_f32: mono float32 @ 16kHz
        Returns a single concatenated text string (post-processed).
        """
        t0 = time.time()
        res = self._asr.infer(audio_f32, print_rtf=False)
        text = " ".join(self._post(s) for s in res).strip()
        # latency = time.time() - t0
        return text

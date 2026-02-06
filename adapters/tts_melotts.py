# AImy/adapters/tts_melotts.py
import sys, gc, uuid
from pathlib import Path
import numpy as np
import onnxruntime as ort
import axengine as axe
import soundfile
from loguru import logger

import config

def intersperse(lst, item):
    result = [item] * (len(lst) * 2 + 1)
    result[1::2] = lst
    return result

def audio_numpy_concat(segment_data_list, sr, speed=1.):
    audio_segments = []
    for segment_data in segment_data_list:
        audio_segments += segment_data.reshape(-1).tolist()
        audio_segments += [0] * int((sr * 0.05) / speed)
    audio_segments = np.array(audio_segments).astype(np.float32)
    return audio_segments

def merge_sub_audio(sub_audio_list, pad_size, audio_len):
    if pad_size > 0:
        for i in range(len(sub_audio_list) - 1):
            sub_audio_list[i][-pad_size:] += sub_audio_list[i+1][:pad_size]
            sub_audio_list[i][-pad_size:] /= 2
            if i > 0:
                sub_audio_list[i] = sub_audio_list[i][pad_size:]
    sub_audio = np.concatenate(sub_audio_list, axis=-1)
    return sub_audio[:audio_len]

def calc_word2pronoun(word2ph, pronoun_lens):
    indice = [0]
    for ph in word2ph[:-1]:
        indice.append(indice[-1] + ph)
    word2pronoun = []
    for i, ph in zip(indice, word2ph):
        word2pronoun.append(np.sum(pronoun_lens[i : i + ph]))
    return word2pronoun

def generate_slices(word2pronoun, dec_len):
    pn_start, pn_end = 0, 0
    zp_start, zp_end = 0, 0
    zp_len = 0
    pn_slices, zp_slices = [], []
    while pn_end < len(word2pronoun):
        if pn_end - pn_start > 2 and np.sum(word2pronoun[pn_end - 2 : pn_end + 1]) <= dec_len:
            zp_len = np.sum(word2pronoun[pn_end - 2 : pn_end])
            zp_start = zp_end - zp_len
            pn_start = pn_end - 2
        else:
            zp_len = 0
            zp_start = zp_end
            pn_start = pn_end
        while pn_end < len(word2pronoun) and zp_len + word2pronoun[pn_end] <= dec_len:
            zp_len += word2pronoun[pn_end]
            pn_end += 1
        zp_end = zp_start + zp_len
        pn_slices.append(slice(pn_start, pn_end))
        zp_slices.append(slice(zp_start, zp_end))
    return pn_slices, zp_slices


class MeloTTSAdapter:
    def __init__(self,
                 encoder_path: Path = config.MELO_ENCODER,
                 decoder_path: Path = config.MELO_DECODER,
                 gvec_path:    Path = config.MELO_GVEC,
                 language:     str  = config.MELO_LANG,
                 sample_rate:  int  = config.MELO_SAMPLE_RATE,
                 dec_len:      int  = config.MELO_DEC_LEN,
                 speed:        float= config.MELO_SPEED,
                 outdir:       Path = config.MELO_TMP_OUTDIR):
        self.encoder_path = Path(encoder_path)
        self.decoder_path = Path(decoder_path)
        self.gvec_path    = Path(gvec_path)
        self.language     = "ZH_MIX_EN" if language == "ZH" else language
        self.sample_rate  = sample_rate
        self.dec_len      = dec_len
        self.speed        = speed
        self.outdir       = Path(outdir)

        # Runtime-loaded Melo modules/functions (populated in init_tts)
        self._split_sentence = None
        self._LANG_TO_SYMBOL_MAP = None
        self._clean_text = None
        self._cleaned_text_to_sequence = None

        self.sess_enc = None
        self.sess_dec = None
        self.g_vec = None
        self.symbol_to_id = None

        self._inited = False
        self._id = uuid.uuid4().hex[:6]
        logger.debug(f"[MeloTTS] Adapter instance #{self._id} constructed")

    def init_tts(self):
        """Defer adding Melo's /python to sys.path and importing its modules until here."""
        if self._inited:
            logger.debug(f"[MeloTTS] init_tts() called again for #{self._id}; skipping re-init")
            return
        self.outdir.mkdir(parents=True, exist_ok=True)

        # Append Melo python dir to the END of sys.path to avoid shadowing 'utils' for SenseVoice
        melo_py = str(config.MELO_DIR)
        if melo_py not in sys.path:
            sys.path.append(melo_py)

        # Now import Melo modules safely
        from split_utils import split_sentence
        from symbols import LANG_TO_SYMBOL_MAP
        from text.cleaner import clean_text
        from text import cleaned_text_to_sequence

        self._split_sentence = split_sentence
        self._LANG_TO_SYMBOL_MAP = LANG_TO_SYMBOL_MAP
        self._clean_text = clean_text
        self._cleaned_text_to_sequence = cleaned_text_to_sequence

        # Warm-up language
        try:
            self._clean_text("预热 warmup" if self.language == "ZH_MIX_EN" else "warmup", self.language)
        except Exception:
            pass

        # Build AXEngine decoder first (initializes AX runtime)
        self.sess_dec = axe.InferenceSession(str(self.decoder_path))

        # Encoder on CPU via ORT
        self.sess_enc = ort.InferenceSession(
            str(self.encoder_path),
            providers=["CPUExecutionProvider"],
            sess_options=ort.SessionOptions()
        )

        assert self.gvec_path.exists(), f"Missing conditioning vector: {self.gvec_path}"
        self.g_vec = np.fromfile(self.gvec_path, dtype=np.float32).reshape(1, 256, 1)

        self.symbol_to_id = {s: i for i, s in enumerate(self._LANG_TO_SYMBOL_MAP[self.language])}
        logger.info(f"[MeloTTS] Initialized ({self.language})")

        self._inited = True

    # --- Helpers now bound to instance so they use the imported functions ---
    def _get_text_for_tts_infer(self, text: str):
        norm_text, phone, tone, word2ph = self._clean_text(text, self.language)
        phone, tone, language = self._cleaned_text_to_sequence(phone, tone, self.language, self.symbol_to_id)
        phone = np.array(intersperse(phone, 0), dtype=np.int32)
        tone = np.array(intersperse(tone, 0), dtype=np.int32)
        language = np.array(intersperse(language, 0), dtype=np.int32)
        word2ph = np.array(word2ph, dtype=np.int32) * 2
        if word2ph.size > 0:
            word2ph[0] += 1
        return phone, tone, language, norm_text, word2ph

    def _split_sentences_into_pieces(self, text: str, quiet=True):
        split_lang = "ZH_MIX_EN" if self.language == "EN" else self.language
        return self._split_sentence(text, language_str=split_lang)

    def _synth_sentence(self, text: str) -> np.ndarray:
        # same pre-normalization
        se = text
        if self.language in ["EN", "ZH_MIX_EN"]:
            import re
            se = re.sub(r'([a-z])([A-Z])', r'\1 \2', se)

        phones, tones, lang_ids, norm_text, word2ph = self._get_text_for_tts_infer(se)

        z_p, pronoun_lens, audio_len = self.sess_enc.run(
            None,
            input_feed={
                'phone': phones, 'g': self.g_vec,
                'tone': tones, 'language': lang_ids,
                'noise_scale': np.array([0], dtype=np.float32),
                'length_scale': np.array([1.0 / self.speed], dtype=np.float32),
                'noise_scale_w': np.array([0], dtype=np.float32),
                'sdp_ratio': np.array([0], dtype=np.float32)
            }
        )

        word2pronoun = calc_word2pronoun(word2ph, pronoun_lens)
        pn_slices, zp_slices = generate_slices(word2pronoun, self.dec_len)

        audio_len = int(audio_len[0])
        sub_audio_list = []
        for i, (ps, zs) in enumerate(zip(pn_slices, zp_slices)):
            zp_slice = z_p[..., zs]
            sub_dec_len = zp_slice.shape[-1]
            sub_audio_len = 512 * sub_dec_len
            if sub_dec_len < self.dec_len:
                zp_slice = np.concatenate(
                    (zp_slice, np.zeros((*zp_slice.shape[:-1], self.dec_len - sub_dec_len), dtype=np.float32)),
                    axis=-1
                )
            audio = self.sess_dec.run(None, input_feed={"z_p": zp_slice, "g": self.g_vec})[0].flatten()

            audio_start = 0
            if len(sub_audio_list) > 0 and pn_slices[i - 1].stop > ps.start:
                audio_start = 512 * word2pronoun[ps.start]
            audio_end = sub_audio_len
            if i < len(pn_slices) - 1 and ps.stop > pn_slices[i + 1].start:
                audio_end = sub_audio_len - 512 * word2pronoun[ps.stop - 1]
            audio = audio[audio_start:audio_end]
            sub_audio_list.append(audio)

        return merge_sub_audio(sub_audio_list, 0, audio_len)

    def synth(self, text: str) -> str:
        sens = self._split_sentences_into_pieces(text, quiet=True)
        audio_list = [self._synth_sentence(se) for se in sens]
        audio = audio_numpy_concat(audio_list, sr=self.sample_rate, speed=self.speed)
        out_path = self.outdir / f"tts_{uuid.uuid4().hex[:8]}.wav"
        soundfile.write(str(out_path), audio, self.sample_rate)
        return str(out_path)

    def shutdown(self):
        try:
            if self.sess_enc: del self.sess_enc
        except: pass
        try:
            if self.sess_dec: del self.sess_dec
        except: pass
        self.sess_enc = self.sess_dec = None
        gc.collect()

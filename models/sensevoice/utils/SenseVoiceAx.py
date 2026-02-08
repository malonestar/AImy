# AUTHOR: AXERA-TECH
# https://huggingface.co/AXERA-TECH/SenseVoice/tree/main/python
# this was recently updated, added older version for compatibility
# will update Sensevoice functionality soon and add to downloader script

import axengine as axe
import numpy as np
import librosa
from utils.frontend import WavFrontend
import os
import time
from typing import Any, Dict, Iterable, List, NamedTuple, Set, Tuple, Union
from utils.print_utils import rich_transcription_postprocess

def sequence_mask(lengths, maxlen=None, dtype=np.float32):
    # 如果 maxlen 未指定，则取 lengths 中的最大值
    if maxlen is None:
        maxlen = np.max(lengths)
    
    # 创建一个从 0 到 maxlen-1 的行向量
    row_vector = np.arange(0, maxlen, 1)
    
    # 将 lengths 转换为列向量
    matrix = np.expand_dims(lengths, axis=-1)
    
    # 比较生成掩码
    mask = row_vector < matrix
    if mask.shape[-1] < lengths[0]:
        mask = np.concatenate([mask, np.zeros((mask.shape[0], lengths[0] - mask.shape[-1]), dtype=np.float32)], axis=-1)
    
    # 返回指定数据类型的掩码
    return mask.astype(dtype)[None, ...]
    
def unique_consecutive_np(arr):
    """
    找出数组中连续的唯一值，模拟 torch.unique_consecutive(yseq, dim=-1)
    
    参数:
    arr: 一维numpy数组
    
    返回:
    unique_values: 去除连续重复值后的数组
    """
    if len(arr) == 0:
        return np.array([])
    
    if len(arr) == 1:
        return arr.copy()
    
    # 找出变化的位置
    diff = np.diff(arr)
    change_positions = np.where(diff != 0)[0] + 1
    
    # 添加起始位置
    start_positions = np.concatenate(([0], change_positions))
    
    # 获取唯一值（每个连续段的第一个值）
    unique_values = arr[start_positions]
    
    return unique_values


def longest_common_suffix_prefix_with_tolerance(
    lhs, 
    rhs, 
    tolerate: int = 0
) -> int:
    """
    计算两个数组的最长公共子序列，该子序列必须同时满足：
    - 是 lhs 的后 n 个元素（后缀）
    - 是 rhs 的前 n 个元素（前缀）
    并且允许最多 `tolerate` 个元素不匹配。

    参数:
        lhs: np.ndarray, 第一个数组
        rhs: np.ndarray, 第二个数组
        tolerate: int, 允许的不匹配元素数量（默认为 0，即完全匹配）

    返回:
        int: 最长公共后缀/前缀的长度（如果没有则返回 0）
    """
    max_possible_n = min(len(lhs), len(rhs))
    
    for n in range(max_possible_n, 0, -1):
        mismatches = np.sum(lhs[-n:] != rhs[:n])
        if mismatches <= tolerate:
            return n
    
    return 0

class SenseVoiceAx:
    def __init__(self, model_path, max_len=256, language="auto", use_itn=True, tokenizer=None):
        model_path_root = os.path.join(os.path.dirname(model_path), "..")
        embedding_root = os.path.join(model_path_root, "embeddings")
        self.frontend = WavFrontend(cmvn_file=f"{model_path_root}/am.mvn",
                                    fs=16000, 
                                    window="hamming", 
                                    n_mels=80, 
                                    frame_length=25, 
                                    frame_shift=10,
                                    lfr_m=7,
                                    lfr_n=6,)
        self.model = axe.InferenceSession(model_path)
        self.sample_rate = 16000
        self.tokenizer = tokenizer
        self.blank_id = 0
        self.max_len = max_len
        self.padding = 16

        self.lid_dict = {"auto": 0, "zh": 3, "en": 4, "yue": 7, "ja": 11, "ko": 12, "nospeech": 13}
        self.lid_int_dict = {24884: 3, 24885: 4, 24888: 7, 24892: 11, 24896: 12, 24992: 13}
        self.textnorm_dict = {"withitn": 14, "woitn": 15}
        self.textnorm_int_dict = {25016: 14, 25017: 15}
        self.emo_dict = {"unk": 25009, "happy": 25001, "sad": 25002, "angry": 25003, "neutral": 25004}

        self.position_encoding = np.load(f"{embedding_root}/position_encoding.npy")
        self.language_query = np.load(f"{embedding_root}/{language}.npy")
        self.textnorm_query = np.load(f"{embedding_root}/withitn.npy") if use_itn else np.load(f"{embedding_root}/woitn.npy")
        self.event_emo_query = np.load(f"{embedding_root}/event_emo.npy")
        self.input_query = np.concatenate((self.textnorm_query, self.language_query, self.event_emo_query), axis=1)
        self.query_num = self.input_query.shape[1]

        self.model_path_root = model_path_root
        self.embedding_root = embedding_root
        self.language = language

    @property
    def language_options(self):
        return list(self.lid_dict.keys())
    
    @property
    def textnorm_options(self):
        return list(self.textnorm_dict.keys())
    
    def choose_language(self, language):
        self.language_query = np.load(f"{self.embedding_root}/{language}.npy")
        self.input_query = np.concatenate((self.textnorm_query, self.language_query, self.event_emo_query), axis=1)
        self.language = language

    def load_data(self, filepath: str) -> np.ndarray:
        waveform, _ = librosa.load(filepath, sr=self.sample_rate)
        return waveform.flatten()
    
    @staticmethod
    def pad_feats(feats: List[np.ndarray], max_feat_len: int) -> np.ndarray:
        def pad_feat(feat: np.ndarray, cur_len: int) -> np.ndarray:
            pad_width = ((0, max_feat_len - cur_len), (0, 0))
            return np.pad(feat, pad_width, "constant", constant_values=0)

        feat_res = [pad_feat(feat, feat.shape[0]) for feat in feats]
        feats = np.array(feat_res).astype(np.float32)
        return feats

    def preprocess(self, waveform):
        feats, feats_len = [], []
        for wf in [waveform]:
            speech, _ = self.frontend.fbank(wf)
            feat, feat_len = self.frontend.lfr_cmvn(speech)
            feats.append(feat)
            feats_len.append(feat_len)

        feats = self.pad_feats(feats, np.max(feats_len))
        feats_len = np.array(feats_len).astype(np.int32)
        return feats, feats_len
    
    def postprocess(self, ctc_logits, encoder_out_lens):
        # 提取数据
        x = ctc_logits[0, :encoder_out_lens[0], :]

        # 获取最大值索引
        yseq = np.argmax(x, axis=-1)

        # 去除连续重复元素
        yseq = unique_consecutive_np(yseq)

        # 创建掩码并过滤 blank_id
        mask = yseq != self.blank_id
        token_int = yseq[mask].tolist()

        return token_int
    
    def infer_waveform(self, waveform: np.ndarray, language="auto"):
        if language != self.language:
            self.choose_language(language)

        feat, feat_len = self.preprocess(waveform)

        slice_len = self.max_len - self.query_num
        slice_num = int(np.ceil(feat.shape[1] / slice_len))

        asr_res = []
        for i in range(slice_num):
            if i == 0:
                sub_feat = feat[:, i*slice_len:(i+1)*slice_len, :]
            else:
                sub_feat = feat[:, i*slice_len - self.padding:(i+1)*slice_len - self.padding, :]
            # concat query
            sub_feat = np.concatenate([self.input_query, sub_feat], axis=1)
            real_len = sub_feat.shape[1]
            if real_len < self.max_len:
                sub_feat = np.concatenate([
                        sub_feat, 
                        np.zeros((1, self.max_len - real_len, sub_feat.shape[-1]), dtype=np.float32)
                    ],
                    axis=1)
                
            masks = sequence_mask(np.array([self.max_len], dtype=np.int32), maxlen=real_len, dtype=np.float32)

            outputs = self.model.run(None, {"speech": sub_feat,
                                            "masks": masks,
                                            "position_encoding": self.position_encoding})
            ctc_logits, encoder_out_lens = outputs

            token_int = self.postprocess(ctc_logits, encoder_out_lens)

            if self.tokenizer is not None:
                asr_res.append(self.tokenizer.tokens2text(token_int))
            else:
                asr_res.append(token_int)

        return asr_res
    
    def infer(self, filepath_or_data: Union[np.ndarray, str], language="auto", print_rtf=True):
        if isinstance(filepath_or_data, str):
            waveform = self.load_data(filepath_or_data)
        else:
            waveform = filepath_or_data

        total_time = waveform.shape[-1] / self.sample_rate

        start = time.time()
        asr_res = self.infer_waveform(waveform, language)
        latency = time.time() - start

        if print_rtf:
            rtf = latency / total_time
            print(f"RTF: {rtf}    Latency: {latency}s  Total length: {total_time}s")
        return asr_res

    
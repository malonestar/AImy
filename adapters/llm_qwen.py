# AImy/adapters/llm_qwen.py
import sys, os
import numpy as np
import config
from ml_dtypes import bfloat16
from transformers import AutoTokenizer, AutoConfig
from axengine import InferenceSession

# ====== Constants ======
SYSTEM_PROMPT = config.LLM_SYSTEM_PROMPT
TOPK          = config.LLM_TOPK
TOPP          = config.LLM_TOPP
TEMPERATURE   = config.LLM_TEMPERATURE

INPUT_PREFILL_LEN = 128
KV_MASK_EXPAND_LEN = 128
LAST_N = 2559

def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - x.max()
    e = np.exp(x)
    return e / e.sum()

def _top_p(prob: np.ndarray, p: float) -> np.ndarray:
    idx = np.argsort(prob)[::-1]
    res = prob.copy()
    cum = 0.0
    cutoff = False
    for i in idx:
        cum += res[i]
        if cum >= p:
            cutoff = True
            continue
        if cutoff:
            res[i] = 0.0
    s = res.sum()
    if s > 0:
        res = res / s
    return res

def _sample_logits_to_id(logits: np.ndarray, topk=TOPK, topp=TOPP, temperature=TEMPERATURE) -> int:
    r = logits.astype(np.float32).flatten()
    # topk
    cand_idx = np.argpartition(r, -topk)[-topk:]
    cand_val = r[cand_idx]
    # temp
    cand_val = cand_val / temperature
    # softmax
    prob = _softmax_np(cand_val)
    # topp
    prob = _top_p(prob, topp)
    prob = prob / prob.sum()
    pos = np.random.multinomial(1, prob).argmax()
    next_token = int(cand_idx[pos])
    return next_token

def _gen_slice_indices(token_len: int, prefill=INPUT_PREFILL_LEN, expand=KV_MASK_EXPAND_LEN):
    remaining = max(0, token_len - prefill)
    extra_blocks = (remaining + expand - 1) // expand
    return list(range(extra_blocks + 1))


class QwenAdapter:
    """
    Preloads tokenizer, cfg, embed matrix, all layer sessions, and post session.
    Provides:
      - generate(user_text) -> full string
      - generate_stream(user_text, chunk_cb=callable) -> full string, emits chunks as they’re decoded
    """

    def __init__(self, hf_model_path: str, axmodel_path: str):
        self.hf_model_path = hf_model_path
        self.axmodel_path = axmodel_path

        self.cfg = None
        self.tokenizer = None
        self.embeds = None
        self.k_caches = None
        self.v_caches = None
        self.sessions = None
        self.post_session = None

    # ---------- init ----------
    def init_model(self):
        self.cfg = AutoConfig.from_pretrained(self.hf_model_path, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained(self.hf_model_path, trust_remote_code=True, use_fast=False)

        # embeddings
        self.embeds = np.load(os.path.join(self.axmodel_path, "model.embed_tokens.weight.npy"))

        # caches
        kv_dim = self.cfg.hidden_size // self.cfg.num_attention_heads * self.cfg.num_key_value_heads
        self.k_caches = [np.zeros((1, LAST_N, kv_dim), dtype=bfloat16) for _ in range(self.cfg.num_hidden_layers)]
        self.v_caches = [np.zeros((1, LAST_N, kv_dim), dtype=bfloat16) for _ in range(self.cfg.num_hidden_layers)]

        # layer sessions
        self.sessions = []
        for i in range(self.cfg.num_hidden_layers):
            sess = InferenceSession(os.path.join(self.axmodel_path, f"qwen2_p128_l{i}_together.axmodel"))
            self.sessions.append(sess)

        # postprocess session
        self.post_session = InferenceSession(os.path.join(self.axmodel_path, "qwen2_post.axmodel"))

    # ---------- prompt builder ----------
    def _build_prompt_ids(self, user_text: str):
        messages = [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_text}
        ]
        text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        model_inputs = self.tokenizer([text], return_tensors="pt").to("cpu")
        token_ids = model_inputs.input_ids[0].cpu().numpy().tolist()
        return token_ids

    # ---------- prefill ----------
    def _prefill(self, token_ids: list):
        embeds = self.embeds
        token_len = len(token_ids)
        prefill_data = np.take(embeds, token_ids, axis=0).astype(bfloat16)

        slice_indices = _gen_slice_indices(token_len, INPUT_PREFILL_LEN, KV_MASK_EXPAND_LEN)

        data = None
        for slice_index in slice_indices:
            current_slice_len = INPUT_PREFILL_LEN if slice_index == 0 else KV_MASK_EXPAND_LEN

            indices = np.array(
                list(range(slice_index * INPUT_PREFILL_LEN, (slice_index + 1) * INPUT_PREFILL_LEN)),
                np.uint32
            ).reshape((1, INPUT_PREFILL_LEN))

            mask = (np.zeros((1, INPUT_PREFILL_LEN, current_slice_len * slice_index + INPUT_PREFILL_LEN)) - 65536).astype(bfloat16)

            data = np.zeros((1, INPUT_PREFILL_LEN, self.cfg.hidden_size), dtype=bfloat16)
            for i, t in enumerate(range(slice_index * INPUT_PREFILL_LEN, (slice_index + 1) * INPUT_PREFILL_LEN)):
                if t < token_len:
                    mask[:, i, : slice_index * INPUT_PREFILL_LEN + i + 1] = 0
                    data[:, i:i+1, :] = prefill_data[t].reshape((1,1,self.cfg.hidden_size)).astype(bfloat16)

            if slice_index == slice_indices[-1]:
                curlen_procd = token_len - slice_index * INPUT_PREFILL_LEN
            else:
                curlen_procd = INPUT_PREFILL_LEN

            for i in range(self.cfg.num_hidden_layers):
                input_feed = {
                    "K_cache": (self.k_caches[i][:, 0: current_slice_len * slice_index, :]
                                if slice_index else np.zeros((1, 1, self.cfg.hidden_size), dtype=bfloat16)),
                    "V_cache": (self.v_caches[i][:, 0: current_slice_len * slice_index, :]
                                if slice_index else np.zeros((1, 1, self.cfg.hidden_size), dtype=bfloat16)),
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.sessions[i].run(None, input_feed, shape_group=slice_index + 1)
                self.k_caches[i][:, slice_index * INPUT_PREFILL_LEN : slice_index * INPUT_PREFILL_LEN + curlen_procd, :] = outputs[0][:, :curlen_procd, :]
                self.v_caches[i][:, slice_index * INPUT_PREFILL_LEN : slice_index * INPUT_PREFILL_LEN + curlen_procd, :] = outputs[1][:, :curlen_procd, :]
                data = outputs[2]

        post_inp = data[:, token_len - (len(slice_indices) - 1) * INPUT_PREFILL_LEN - 1, None, :]
        post_out = self.post_session.run(None, {"input": post_inp})[0]
        next_token = _sample_logits_to_id(post_out, TOPK, TOPP, TEMPERATURE)
        return token_len, next_token

    # ---------- decode loop (shared) ----------
    def _decode_tokens(self, token_ids: list, token_len: int, max_steps=LAST_N):
        kv_cache_len = LAST_N

        mask = np.zeros((1, 1, kv_cache_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :kv_cache_len] -= 65536
        if INPUT_PREFILL_LEN > 0:
            mask[:, :, :token_len] = 0

        # iterate positions
        for pos in range(kv_cache_len):
            if INPUT_PREFILL_LEN > 0 and pos < token_len:
                continue

            next_token = token_ids[pos]
            indices = np.array([pos], np.uint32).reshape((1,1))
            data = self.embeds[next_token, :].reshape((1,1,self.cfg.hidden_size)).astype(bfloat16)

            for i in range(self.cfg.num_hidden_layers):
                input_feed = {
                    "K_cache": self.k_caches[i],
                    "V_cache": self.v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.sessions[i].run(None, input_feed, shape_group=0)
                self.k_caches[i][:, pos, :] = outputs[0][:, :, :]
                self.v_caches[i][:, pos, :] = outputs[1][:, :, :]
                data = outputs[2]

            mask[..., pos] = 0

            if pos >= token_len - 1:
                post_out = self.post_session.run(None, {"input": data})[0]
                next_token = _sample_logits_to_id(post_out, TOPK, TOPP, TEMPERATURE)
                token_ids.append(next_token)
                if next_token == self.tokenizer.eos_token_id and next_token > token_len:
                    break

        return token_ids

    # ---------- public: one-shot ----------
    def generate(self, user_text: str) -> str:
        token_ids = self._build_prompt_ids(user_text)
        token_len, next_token = self._prefill(token_ids)
        token_ids.append(next_token)
        token_ids = self._decode_tokens(token_ids, token_len)
        out = self.tokenizer.decode(token_ids[token_len:], skip_special_tokens=True)
        return out.strip()

    # ---------- public: streaming ----------
    def generate_stream(self, user_text: str, chunk_cb=None) -> str:
        """
        Same as generate(), but calls chunk_cb(piece) as tokens come out.
        Returns the full text at the end.
        """
        token_ids = self._build_prompt_ids(user_text)
        token_len, next_token = self._prefill(token_ids)
        token_ids.append(next_token)

        # first piece
        full_text_parts = []
        first_piece = self.tokenizer.decode([next_token], skip_special_tokens=True)
        if first_piece:
            full_text_parts.append(first_piece)
            if chunk_cb:
                chunk_cb(first_piece)

        kv_cache_len = LAST_N
        mask = np.zeros((1, 1, kv_cache_len + 1), dtype=np.float32).astype(bfloat16)
        mask[:, :, :kv_cache_len] -= 65536
        if INPUT_PREFILL_LEN > 0:
            mask[:, :, :token_len] = 0

        for pos in range(kv_cache_len):
            if pos < token_len:
                continue

            last_id = token_ids[pos]
            indices = np.array([pos], np.uint32).reshape((1,1))
            data = self.embeds[last_id, :].reshape((1,1,self.cfg.hidden_size)).astype(bfloat16)

            for i in range(self.cfg.num_hidden_layers):
                input_feed = {
                    "K_cache": self.k_caches[i],
                    "V_cache": self.v_caches[i],
                    "indices": indices,
                    "input": data,
                    "mask": mask,
                }
                outputs = self.sessions[i].run(None, input_feed, shape_group=0)
                self.k_caches[i][:, pos, :] = outputs[0][:, :, :]
                self.v_caches[i][:, pos, :] = outputs[1][:, :, :]
                data = outputs[2]

            mask[..., pos] = 0

            post_out = self.post_session.run(None, {"input": data})[0]
            next_token = _sample_logits_to_id(post_out, TOPK, TOPP, TEMPERATURE)
            token_ids.append(next_token)

            if next_token == self.tokenizer.eos_token_id and next_token > token_len:
                break

            piece = self.tokenizer.decode([next_token], skip_special_tokens=True)
            if piece:
                full_text_parts.append(piece)
                if chunk_cb:
                    chunk_cb(piece)

        return "".join(full_text_parts).strip()

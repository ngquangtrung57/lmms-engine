"""In-loop benchmark evaluation: generate on a verl-format parquet benchmark and
score with a rule-based reward function.

Adapted from verl's RL validation (``RayPPOTrainer._validate`` + ``RLHFDataset`` +
``custom_reward_function``): the benchmark is a parquet with columns
``prompt`` (chat messages, ``<image>`` placeholders), ``images``
(list of ``{bytes, path}``), ``data_source``, ``reward_model{ground_truth, style}``
and ``extra_info``. The reward function is loaded from a configurable file path
with signature ``compute_score(data_source, solution_str, ground_truth, extra_info)``
returning a float or a dict with ``score`` / ``acc_score`` / ``format_reward_score``.

Image-only for now; rows carrying videos raise at load time.
"""

import importlib.util
import io
import math
import random
from collections import defaultdict
from dataclasses import dataclass, fields
from typing import Any, Dict, List, Optional

import pyarrow.parquet as pq
from loguru import logger
from PIL import Image
from torch.utils.data import Dataset

IMAGE_PLACEHOLDER = "<image>"


@dataclass
class BenchmarkEvalConfig:
    dataset_path: str
    reward_fn_path: str
    reward_fn_name: str = "compute_score"
    eval_steps: int = 200
    eval_on_start: bool = True
    per_device_batch_size: int = 8
    # Total sequence cap (prompt + generated). Matches SFT packing_length so the
    # model never decodes past lengths seen in training.
    max_total_len: int = 16384
    # Explicit cap; the effective budget is min(max_new_tokens, max_total_len - prompt_len).
    max_new_tokens: Optional[int] = None
    do_sample: bool = False
    # Stratified per-data_source cap, seeded -> identical subset every eval.
    max_samples: Optional[int] = None
    seed: int = 42
    # How many generations rank 0 dumps to output_dir/bench_eval/step{N}.jsonl.
    log_samples: int = 32
    num_workers: int = 2

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "BenchmarkEvalConfig":
        d = dict(d)
        known = {f.name for f in fields(cls)}
        unknown = set(d) - known
        if unknown:
            logger.warning(f"benchmark_eval: ignoring unknown keys {sorted(unknown)}")
        return cls(**{k: v for k, v in d.items() if k in known})


def load_reward_fn(path: str, name: str = "compute_score"):
    """Load the reward function from a file path (verl custom_reward_function pattern)."""
    spec = importlib.util.spec_from_file_location("lmms_engine_benchmark_reward", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return getattr(module, name)


def _build_messages(prompt: List[Dict[str, str]], images: List[Image.Image]):
    """Replace <image> placeholders with typed content items (verl _build_messages).

    Images are consumed in placeholder order across the whole conversation; the
    PIL objects themselves are passed to the processor separately, so content
    items only carry {"type": "image"} markers.
    """
    messages = []
    used = 0
    for msg in prompt:
        text = msg["content"]
        if IMAGE_PLACEHOLDER not in text:
            messages.append({"role": msg["role"], "content": [{"type": "text", "text": text}]})
            continue
        content = []
        segments = text.split(IMAGE_PLACEHOLDER)
        for i, seg in enumerate(segments):
            if i > 0:
                content.append({"type": "image"})
                used += 1
            if seg:
                content.append({"type": "text", "text": seg})
        messages.append({"role": msg["role"], "content": content})
    if used != len(images):
        raise ValueError(f"<image> placeholder count {used} != number of images {len(images)}")
    return messages


class BenchmarkDataset(Dataset):
    """Map-style reader for a verl-format benchmark parquet.

    Items are served in prompt-length-sorted order: batches then carry similar
    prompt lengths (minimal left-padding waste) and tend to hit EOS together,
    which keeps GPU utilization high during batched greedy generation.
    Original row indices are preserved in ``idx`` for post-gather dedupe.
    """

    def __init__(self, config: BenchmarkEvalConfig, hf_processor):
        self.config = config
        self.hf_processor = hf_processor
        table = pq.read_table(config.dataset_path)

        # Image bytes stay in the single pyarrow table (copy-on-write shared with
        # forked dataloader workers); only light metadata is materialized.
        self._images_col = table.column("images")
        self._prompts = table.column("prompt").to_pylist()
        self._sources = table.column("data_source").to_pylist()
        self._ground_truths = [rm["ground_truth"] for rm in table.column("reward_model").to_pylist()]
        self._extra_infos = (
            table.column("extra_info").to_pylist() if "extra_info" in table.column_names else [None] * len(self._prompts)
        )
        if "videos" in table.column_names:
            videos = table.column("videos").to_pylist()
            bad = next((i for i, v in enumerate(videos) if v), None)
            if bad is not None:
                raise ValueError(
                    f"{config.dataset_path}: row {bad} carries videos; benchmark eval is image-only for now"
                )

        indices = list(range(len(self._prompts)))
        if config.max_samples is not None and config.max_samples < len(indices):
            indices = self._stratified_subset(indices, config.max_samples, config.seed)
        self._order = self._sort_by_prompt_len(indices)
        logger.info(
            f"BenchmarkDataset: {len(self._order)}/{len(self._prompts)} rows from {config.dataset_path}"
        )

    def _stratified_subset(self, indices: List[int], max_samples: int, seed: int) -> List[int]:
        by_src = defaultdict(list)
        for i in indices:
            by_src[self._sources[i]].append(i)
        # Distribute the remainder so the subset totals exactly max_samples
        # (sources may still cap out below their share if they are small).
        per_src, extra = divmod(max_samples, len(by_src))
        rng = random.Random(seed)
        picked = []
        for k, src in enumerate(sorted(by_src)):
            take = per_src + (1 if k < extra else 0)
            src_indices = by_src[src]
            rng.shuffle(src_indices)
            picked.extend(src_indices[:take])
        return sorted(picked)

    def _sort_by_prompt_len(self, indices: List[int]) -> List[int]:
        """Estimate prompt length = text tokens + visual tokens from the image
        header (PIL reads dimensions without decoding pixel data)."""
        tokenizer = getattr(self.hf_processor, "tokenizer", None)
        merge = 28  # qwen-vl: 14px patch * 2x2 spatial merge per visual token
        texts = [" ".join(m["content"] for m in self._prompts[i]) for i in indices]
        if tokenizer is not None:
            # One batched call: fast tokenizers parallelize internally, vs ~10s+
            # of sequential per-row calls on the full 7k-row benchmark.
            text_lens = [len(ids) for ids in tokenizer(texts, add_special_tokens=False)["input_ids"]]
        else:
            text_lens = [len(t) // 4 for t in texts]
        lengths = []
        for i, n_text in zip(indices, text_lens):
            n_visual = 0
            for img in self._images_col[i].as_py() or []:
                try:
                    with Image.open(io.BytesIO(img["bytes"])) as im:
                        w, h = im.size
                    n_visual += math.ceil(h / merge) * math.ceil(w / merge)
                except Exception:
                    n_visual += 1024
            lengths.append(n_text + n_visual)
        overlong = sum(1 for n in lengths if n >= self.config.max_total_len)
        if overlong:
            logger.warning(
                f"BenchmarkDataset: {overlong} rows have estimated prompt length >= "
                f"max_total_len={self.config.max_total_len}; they will generate ~1 token and score 0"
            )
        return [i for _, i in sorted(zip(lengths, indices))]

    def __len__(self):
        return len(self._order)

    def __getitem__(self, i: int) -> Dict[str, Any]:
        j = self._order[i]
        images = [
            Image.open(io.BytesIO(img["bytes"])).convert("RGB") for img in (self._images_col[j].as_py() or [])
        ]
        messages = _build_messages(self._prompts[j], images)
        text = self.hf_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return {
            "text": text,
            "images": images,
            "ground_truth": self._ground_truths[j],
            "data_source": self._sources[j],
            "extra_info": self._extra_infos[j] or {},
            "idx": j,
        }


class GenerationCollator:
    """Left-padded batch for generation; metadata passes through untouched."""

    def __init__(self, hf_processor):
        self.hf_processor = hf_processor

    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        texts = [b["text"] for b in batch]
        # Flattened in batch order; the processor matches them sequentially to
        # the image pad runs in each text.
        images = [img for b in batch for img in b["images"]]
        tokenizer = self.hf_processor.tokenizer
        # Left-pad only for this call and RESTORE: the tokenizer is shared with
        # the training pipeline, and VisionCollator flips input_ids whenever it
        # sees padding_side == "left" — a leaked mutation would corrupt training
        # batches in any process that runs both collators.
        prev_padding_side = tokenizer.padding_side
        tokenizer.padding_side = "left"
        try:
            model_inputs = self.hf_processor(
                text=texts,
                images=images if images else None,
                padding=True,
                return_tensors="pt",
            )
        finally:
            tokenizer.padding_side = prev_padding_side
        meta = [{k: b[k] for k in ("ground_truth", "data_source", "extra_info", "idx")} for b in batch]
        return {"model_inputs": model_inputs, "meta": meta}


# Stored/dumped generation text cap. Scoring always sees the FULL text; the
# truncated copy only rides the rank-0 gather (full generations for thousands
# of rows would push the gather payload toward hundreds of MB).
GENERATION_DUMP_MAX_CHARS = 4000


def score_generation(reward_fn, meta: Dict[str, Any], generation: str) -> Dict[str, Any]:
    """Apply the reward fn to one generation; never raises (a single unparsable
    sample must not kill the training job)."""
    try:
        score = reward_fn(meta["data_source"], generation, meta["ground_truth"], meta.get("extra_info"))
    except Exception as e:
        logger.warning(f"benchmark reward failed on idx={meta['idx']}: {e}")
        score = {"score": 0.0, "acc_score": 0.0, "format_reward_score": 0.0}
    if not isinstance(score, dict):
        score = {"score": float(score), "acc_score": float(score), "format_reward_score": 0.0}
    return {
        "idx": meta["idx"],
        "data_source": meta["data_source"],
        "acc": float(score.get("acc_score", score.get("score", 0.0))),
        "score": float(score.get("score", 0.0)),
        "format": float(score.get("format_reward_score", 0.0)),
        "ground_truth": str(meta["ground_truth"]),
        "generation": generation[:GENERATION_DUMP_MAX_CHARS],
    }


def dedupe_by_idx(results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Drop DistributedSampler pad-by-repeat duplicates (keep first per idx)."""
    deduped = {}
    for r in results:
        deduped.setdefault(r["idx"], r)
    return list(deduped.values())


def aggregate_benchmark_results(results: List[Dict[str, Any]]) -> Dict[str, float]:
    """Dedupe by row idx (DistributedSampler pads by repeating samples), then
    aggregate per-data_source and overall accuracy/format/score means."""
    rows = dedupe_by_idx(results)
    if not rows:
        return {}

    def mean(values):
        return sum(values) / len(values)

    by_src = defaultdict(list)
    for r in rows:
        by_src[r["data_source"]].append(r)
    metrics = {}
    for src, rs in sorted(by_src.items()):
        metrics[f"eval/bench/{src}/acc"] = mean([r["acc"] for r in rs])
    metrics["eval/bench/acc_mean"] = mean([r["acc"] for r in rows])
    metrics["eval/bench/acc_macro"] = mean([metrics[f"eval/bench/{src}/acc"] for src in by_src])
    metrics["eval/bench/format"] = mean([r["format"] for r in rows])
    metrics["eval/bench/score"] = mean([r["score"] for r in rows])
    metrics["eval/bench/num_samples"] = len(rows)
    return metrics

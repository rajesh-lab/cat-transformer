"""
lm-evaluation-harness wrapper for Transformer, CAT_Transformer and Beacon_Transformer models.

Usage:
    python eval/harness.py --model_type vanilla \
    --tasks wikitext,lambada_openai,hellaswag,winogrande,arc_easy,swde,fda,niah_single_2,niah_single_3 --metadata '{"max_seq_lengths":[2048,4096]}' \
    --output_path eval/harness_vanilla.json

    python eval/harness.py --model_type chunked --chunk_size_power 3 \
    --tasks wikitext,lambada_openai,hellaswag,winogrande,arc_easy,swde,fda,niah_single_2,niah_single_3 --metadata '{"max_seq_lengths":[2048,4096]}' \
    --output_path eval/harness_cat_transformer.json

    python eval/harness.py --model_type beacon --chunk_size_power 3 \
    --tasks wikitext,lambada_openai,hellaswag,winogrande,arc_easy,swde,fda,niah_single_2,niah_single_3 --metadata '{"max_seq_lengths":[2048,4096]}' \
    --output_path eval/harness_beacon_transformer.json
"""

from __future__ import annotations

import os
import sys
import argparse
import json

import torch
from transformers import AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from transformer import TransformerConfig, Transformer
from cat_transformer_adaptive import CAT_Config, CAT_Transformer
from beacon_transformer import Beacon_Config, Beacon_Transformer
from cat_lookback_transformer import CAT_Lookback_Transformer

import lm_eval
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM

# fill in the right details below!

BLOCK_SIZE = 4096

MODEL_TYPE_TO_PATH = {
    "vanilla": "/path/to/model",
    "chunked": "/path/to/model",
    "beacon": "/path/to/model",
}

# models that take a chunk_size_power at forward time
ADAPTIVE_MODELS = (CAT_Transformer, CAT_Lookback_Transformer, Beacon_Transformer)

TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50257

def get_model(model_type):
    if model_type == "vanilla":
        config = TransformerConfig(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            dim=1024,
            n_head=16,
            n_layer=12,
            use_qk_norm=True,
            use_fused_ops=True,
        )
        return Transformer(config)

    elif model_type == "vanilla2":
        config = TransformerConfig(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            dim=1024,
            n_head=16,
            n_layer=24,
            use_qk_norm=True,
            use_fused_ops=True,
        )
        return Transformer(config)

    elif model_type == "chunked":
        chunk_size = 32
        compressor_config = CAT_Config(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            chunk_size=chunk_size,
            dim=1024,
            n_head=16,
            n_layer=3,
            dim_fx=2048,
            use_qk_norm=True,
        )
        decoder_config = CAT_Config(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            chunk_size=chunk_size,
            dim=2048,
            n_head=32,
            n_layer=12,
            use_qk_norm=True,
            use_fused_ops=True,
        )
        return CAT_Transformer(decoder_config, compressor_config)

    elif model_type == "chunked_lookback":
        chunk_size = 32
        compressor_config = CAT_Config(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            chunk_size=chunk_size,
            dim=1024,
            n_head=16,
            n_layer=3,
            dim_fx=2048,
            use_qk_norm=True,
        )
        decoder_config = CAT_Config(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            chunk_size=chunk_size,
            dim=2048,
            n_head=32,
            n_layer=12,
            use_qk_norm=True,
            use_fused_ops=True,
        )
        return CAT_Lookback_Transformer(decoder_config, compressor_config)

    elif model_type == "beacon":
        config = Beacon_Config(
            vocab_size=VOCAB_SIZE,
            block_size=BLOCK_SIZE,
            chunk_size=32,
            n_beacons=1,
            # must match the checkpoint being loaded
            rope_position_scheme="chunk_reset",
            dim=1024,
            n_head=16,
            n_layer=12,
            use_qk_norm=False,
            use_fused_ops=True,
        )
        return Beacon_Transformer(config)

    else:
        raise ValueError(f"Unknown model type: {model_type}")


def load_model(model_type, device="cuda"):
    model = get_model(model_type)
    model_path = MODEL_TYPE_TO_PATH[model_type]

    state_dict = torch.load(model_path, map_location="cpu", weights_only=True)
    new_state_dict = {k.replace("module.", ""): v for k, v in state_dict.items()}
    print(model.load_state_dict(new_state_dict, strict=True))

    model.eval()
    model.to(device=device)
    model.setup_cache(device=device)
    return model


# Wrapper to use the harness with cat-transformer
class CATTransformerLM(HFLM):
    """lm-evaluation-harness wrapper for Transformer and CAT_Transformer."""

    def __init__(
        self,
        model_type: str = "chunked",
        chunk_size_power: int = 4,
        device: str = "cuda",
        batch_size: int = 1,
        max_length: int = BLOCK_SIZE,
    ):
        # bypass HFLM.__init__, call the grandparent LM.__init__ directly
        LM.__init__(self)

        self.model_type = model_type
        self.chunk_size_power = chunk_size_power

        # load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        # load model
        self._model = load_model(model_type, device=device)

        # HFLM-expected attributes
        self._device = torch.device(device)
        self._config = None
        self.backend = "causal"
        self._max_length = max_length
        self.vocab_size = self.tokenizer.vocab_size
        self.add_bos_token = False
        self.custom_prefix_token_id = None
        self.truncation = False
        self.logits_cache = True

        self.batch_size_per_gpu = int(batch_size)
        self.batch_schedule = 1
        self.batch_sizes = {}
        self.max_batch_size = None

        self._rank = 0
        self._world_size = 1

        self.softmax_dtype = torch.float32
        self.mixed_precision_dtype = torch.bfloat16
        self.think_end_token = None
        self.chat_template_args = {}

        self.pretrained = model_type
        self.revision = "n/a"
        self.peft = None
        self.delta = None

    def _model_call(self, inps, attn_mask=None, labels=None):
        with torch.no_grad(), torch.autocast(
            device_type=self._device.type,
            dtype=self.mixed_precision_dtype,
        ):
            if isinstance(self._model, ADAPTIVE_MODELS):
                return self._model(inps, chunk_size_power=self.chunk_size_power)
            else:
                return self._model(inps)

    def _model_generate(self, context, max_length, stop, **generation_kwargs):
        max_new_tokens = max_length - context.shape[1]
        if max_new_tokens <= 0:
            return context

        cur = context.clone()
        with torch.no_grad(), torch.autocast(
            device_type=self._device.type,
            dtype=self.mixed_precision_dtype,
        ):
            for _ in range(max_new_tokens):
                if isinstance(self._model, ADAPTIVE_MODELS):
                    logits = self._model(cur, chunk_size_power=self.chunk_size_power)
                else:
                    logits = self._model(cur)
                logits = logits[:, -1, :]
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
                cur = torch.cat([cur, next_token], dim=1)

                # check stop sequences
                if stop:
                    decoded = self.tokenizer.decode(cur[0, context.shape[1]:].tolist())
                    if any(s in decoded for s in stop):
                        break

        return cur


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate with lm-evaluation-harness")
    parser.add_argument("--model_type", type=str, required=True, help="Model type: vanilla, chunked, chunked_lookback, beacon")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task names (e.g. hellaswag,arc_easy)")
    parser.add_argument("--chunk_size_power", type=int, default=4, help="Chunk size power for CAT / beacon (default: 4)")
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (default: 1)")
    parser.add_argument("--device", type=str, default="cuda", help="Device (default: cuda)")
    parser.add_argument("--num_fewshot", type=int, default=0, help="Number of few-shot examples (default: 0)")
    parser.add_argument("--limit", type=float, default=None, help="Limit number of examples per task")
    parser.add_argument("--output_path", type=str, default=None, help="Path to save results JSON")
    parser.add_argument("--metadata", type=str, default=None, help='Task metadata JSON (e.g. \'{"max_seq_lengths":[1024]}\')')
    args = parser.parse_args()

    lm = CATTransformerLM(
        model_type=args.model_type,
        chunk_size_power=args.chunk_size_power,
        device=args.device,
        batch_size=args.batch_size,
    )

    task_list = [t.strip() for t in args.tasks.split(",")]

    metadata = json.loads(args.metadata) if args.metadata else {}
    if "tokenizer" not in metadata and "pretrained" not in metadata:
        metadata["tokenizer"] = TOKENIZER_NAME

    from lm_eval.tasks import TaskManager
    task_manager = TaskManager(metadata=metadata)

    eval_kwargs = dict(
        model=lm,
        tasks=task_list,
        task_manager=task_manager,
        num_fewshot=args.num_fewshot,
        batch_size=args.batch_size,
        limit=args.limit,
    )

    results = lm_eval.simple_evaluate(**eval_kwargs)

    # print results
    print("\n" + "=" * 60)
    print("Results:")
    print("=" * 60)
    for task_name, task_results in results["results"].items():
        print(f"\n  {task_name}:")
        for metric, value in task_results.items():
            if not metric.endswith(",stderr"):
                stderr_key = f"{metric},stderr"
                stderr = task_results.get(stderr_key, "")
                stderr_str = f" +/- {stderr:.4f}" if isinstance(stderr, float) else ""
                if isinstance(value, float):
                    print(f"    {metric}: {value:.4f}{stderr_str}")
                else:
                    print(f"    {metric}: {value}")

    if args.output_path:
        os.makedirs(os.path.dirname(args.output_path) or ".", exist_ok=True)
        from lm_eval.utils import handle_non_serializable
        with open(args.output_path, "w") as f:
            json.dump(results["results"], f, default=handle_non_serializable, indent=2)
        print(f"\nResults saved to {args.output_path}")

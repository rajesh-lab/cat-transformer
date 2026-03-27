"""
lm-evaluation-harness wrapper for Transformer and CAT_Transformer models.

Usage:
    python eval/harness.py --model_type chunked --tasks hellaswag,arc_easy

    python eval/harness.py --model_type chunked --chunk_size_power 2 \
    --tasks niah_single_1,niah_single_2,niah_single_3 --metadata '{"max_seq_lengths":[4096]}' \
    --limit 50 --output_path eval/test.json

    python eval/harness.py --model_type chunked --chunk_size_power 3 --tasks niah_single_1 --metadata '{"max_seq_lengths":[1024]}' --output_path eval/test.json

    # wikitext,lambada_openai,hellaswag,winogrande,arc_easy,swde,fda,niah_single_1
    # swde,fda
    # niah_single_1 --metadata '{"max_seq_lengths":[1024]}'
    # niah_single_2 --metadata '{"max_seq_lengths":[1024]}'

    python eval/harness.py --model_type chunked --chunk_size_power 2 --tasks niah_single_1 --limit 100 --metadata '{"max_seq_lengths":[1024]}'

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

import lm_eval
from lm_eval.api.model import LM
from lm_eval.models.huggingface import HFLM

# BLOCK_SIZE = 2048
BLOCK_SIZE = 4096

MODEL_TYPE_TO_PATH = {
    
    # "vanilla": "/scratch/jp7467/cat-transformer/Results/test-fineweb-1b/2026-03-06/12:07:04.421670/state_dict.pt",
    # "chunked": "/scratch/jp7467/cat-transformer/Results/test-fineweb-1b/2026-03-06/15:40:25.010151/state_dict.pt",

    # llama2 tokenization -- 10B tokens, 2K context, D=1024
    # "vanilla" : "/scratch/jp7467/cat-transformer/Results/test-fineweb-1b/2026-03-08/20:35:53.414785/state_dict.pt",
    # "chunked" : "/scratch/jp7467/cat-transformer/Results/test-fineweb-1b/2026-03-07/17:35:43.210462/state_dict.pt", # chunk_size=4,8,16,32

    # gpt2 -- 15B tokens, 4K context, D=1024
    "vanilla" : "/scratch/jp7467/cat-transformer/Results/fineweb-15b/2026-03-16/13:24:52.608318/state_dict.pt", # 12L
    "vanilla2" : "/scratch/jp7467/cat-transformer/Results/fineweb-15b/2026-03-16/13:34:38.920977/state_dict.pt", # 24L

    "chunked" : "/scratch/jp7467/cat-transformer/Results/fineweb-15b/2026-03-20/00:35:07.052056/state_dict.pt",
}

TOKENIZER_NAME = "gpt2"
VOCAB_SIZE = 50257

# TOKENIZER_NAME = "meta-llama/Llama-2-7b-hf"
# VOCAB_SIZE = 32000

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
            if isinstance(self._model, CAT_Transformer):
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
                if isinstance(self._model, CAT_Transformer):
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
    parser.add_argument("--model_type", type=str, required=True, help="Model type: vanilla, chunked")
    parser.add_argument("--tasks", type=str, required=True, help="Comma-separated task names (e.g. hellaswag,arc_easy)")
    parser.add_argument("--chunk_size_power", type=int, default=4, help="Chunk size power for CAT (default: 4)")
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

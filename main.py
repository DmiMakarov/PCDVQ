from codebooks import Codebook
from quantizer import Quantizer
from quantizer import quantize_linear_inplace

import argparse
from pathlib import Path
import logging
import torch
#import evaluate
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
#import json

from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

number_of_linear = 0

def check_number_of_linear(module):
    """Quantize each linear layer in the input module inplace."""
    for _, child in list(module.named_children()):
        if isinstance(child, torch.nn.Linear):
            global number_of_linear
            number_of_linear += 1
        else:
            quantize_linear_inplace(child)

codebook = Codebook()
codebook.load_codebooks()

quantizer = Quantizer(codebook)

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-4B",
                        help="model id")
parser.add_argument("--dataset_name", type=str, default="wikitext",
                        help="dataset name")
parser.add_argument("--dataset_config", type=str, default="wikitext-2-raw-v1",
                        help="dataset configuration name (e.g. wikitext-2-raw-v1)")
parser.add_argument("--split", type=str, default="validation",
                        help="dataset split")
parser.add_argument("--stride", type=int, default=None,
                        help="window stride")
parser.add_argument("--trust_remote_code", action="store_true",
                        help="enable if the model requires custom code")
parser.add_argument("--quantize_with_pcdvq", action="store_true",
                        help="enable PCDVQ quantization of linear layers")
parser.add_argument("--save_path", type=str, default=None,
                        help="save path for quantized model")
parser.add_argument("--batch_size", type=int, default=8,
                        help="batch size")
parser.add_argument("--device", type=str, default='cuda',
                        help="device")
args = parser.parse_args()

dtype = torch.float16
device = args.device

tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        use_fast=True,
        trust_remote_code=args.trust_remote_code,
)
model = AutoModelForCausalLM.from_pretrained(
        args.model_name,
        dtype=dtype,
        device_map=device,
        trust_remote_code=args.trust_remote_code,
)
model.eval()

check_number_of_linear(model)

logger.info(f"Number of linear layers: {number_of_linear}")

dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
texts = [text for text in dataset["text"] if isinstance(text, str) and text.strip() != ""]

save_path = args.model_name

if args.quantize_with_pcdvq:
    logger.info("Quantizing linear layers with PCDVQ...")
    quantize_linear_inplace(model, quantizer=quantizer)
    logger.info("Quantization done.")

    if args.save_path is None:
        project_path = Path.cwd()
        model_dir = Path("quant_models", args.model_name)
        save_path = project_path / model_dir
        save_path.mkdir(parents=True, exist_ok=True)


TASKS = ["wikitext"]

m = HFLM(
        pretrained=model,
        tokenizer=tokenizer,
        max_length=getattr(tokenizer, "model_max_length", 2048),
        batch_size="auto",
        dtype=dtype,
        trust_remote_code=True,
    )

tm = TaskManager()
results = evaluator.simple_evaluate(
        model=m,
        tasks=TASKS,
        task_manager=tm,
)

print(make_table(results))

#model.save_pretrained(save_path)
#tokenizer.save_pretrained(save_path)
#ppl = evaluate.load("perplexity")
#results = ppl.compute(
#    model_id=save_path,
#    batch_size=args.batch_size,
#    device=device,
#    add_start_token=False,
#    max_length=4096,
#    predictions=texts[:8*8],
#)
#logger.info(results)

#with open("results.json", "w") as f:
#    json.dump(results, f)




from pcdvq.codebooks import PCDVQCodebook
from pcdvq.quantizer import Quantizer, quantize_linear_inplace
from pcdvq.utils import get_linear_layers

import argparse
from pathlib import Path
import logging
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM


from lm_eval import evaluator
from lm_eval.tasks import TaskManager
from lm_eval.models.huggingface import HFLM
from lm_eval.utils import make_table
from pcdvq.filters import qwen3_pcdvq_filter


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)



### parse arguments
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B", help="model id")
parser.add_argument("--dataset_name", type=str, default="wikitext", help="dataset name")
parser.add_argument(
    "--dataset_config", type=str, default="wikitext-2-raw-v1", help="dataset configuration name (e.g. wikitext-2-raw-v1)"
)
parser.add_argument("--split", type=str, default="validation", help="dataset split")
parser.add_argument("--stride", type=int, default=None, help="window stride")
parser.add_argument("--trust_remote_code", action="store_true", help="enable if the model requires custom code")
parser.add_argument("--quantize_with_pcdvq", action="store_true", help="enable PCDVQ quantization of linear layers")
parser.add_argument("--save_path", type=str, default=None, help="save path for quantized model")
parser.add_argument("--batch_size", type=int, default=8, help="batch size")
parser.add_argument("--device", type=str, default="cuda", help="device")
parser.add_argument("--chunk_size", type=int, default=1024, help="chunk size for PCDVQ quantization over codebook entries")
parser.add_argument("--phi_chunk_size", type=int, default=262144, help="chunk size for PCDVQ quantization over phi rows")
parser.add_argument("--codebook_path", type=str, default="codebook.pt", help="codebook path")
args = parser.parse_args()

### init quantizer
codebook = PCDVQCodebook()
codebook.load("codebook.pt")

### init tokenizer and model
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

### count linear layers
number_of_linear, number_of_quantizable_linear = 0, 0

number_of_quantizable_linear = len(get_linear_layers(model, qwen3_pcdvq_filter))
number_of_linear = len(get_linear_layers(model))

logger.info(f"Number of quantizable linear layers: {number_of_quantizable_linear}/{number_of_linear}")

quantizer = Quantizer(codebook, codebook_chunk_size=args.chunk_size, phi_chunk_size=args.phi_chunk_size)
save_path = args.model_name
### optionally quantize
if args.quantize_with_pcdvq:
    logger.info("Quantizing linear layers with PCDVQ...")
    quantize_linear_inplace(
        model,
        quantizer=quantizer,
        filter_fn=qwen3_pcdvq_filter,
    )
    logger.info("Quantization done.")

    if args.save_path is None:
        project_path = Path.cwd()
        model_dir = Path("quant_models", args.model_name)
        save_path = project_path / model_dir
        save_path.mkdir(parents=True, exist_ok=True)

### init dataset and evaluate
dataset = load_dataset(args.dataset_name, args.dataset_config, split=args.split)
texts = [text for text in dataset["text"] if isinstance(text, str) and text.strip() != ""]

TASKS = ["wikitext"]

m = HFLM(
    pretrained=model,
    # tokenizer=tokenizer,
    max_length=2048,
    batch_size="auto",
    dtype=dtype,
    trust_remote_code=args.trust_remote_code,
)

tm = TaskManager()
results = evaluator.simple_evaluate(model=m, tasks=TASKS, task_manager=tm)


print(make_table(results))

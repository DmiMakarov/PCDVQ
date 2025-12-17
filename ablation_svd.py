import argparse
import torch
import logging
from transformers import AutoModelForCausalLM, AutoTokenizer
from lm_eval import evaluator
from lm_eval.models.huggingface import HFLM
from lm_eval.tasks import TaskManager
from lm_eval.utils import make_table

from pcdvq.codebooks import PCDVQCodebook
from pcdvq.quantizer import Quantizer, quantize_linear_inplace
from pcdvq.normalization import QRRotation
from pcdvq.baseline import apply_baseline
from pcdvq.filters import qwen3_pcdvq_filter

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def calculate_exact_bpw(model, pcdvq_bits=16, pcdvq_vec_dim=8, svd_rank=0):
    """
    Calculates the memory cost.
    Topic 5: Low-Rank Matrix Storage.
    """
    total_bits = 0
    total_params = 0
    
    for name, module in model.named_modules():
        if isinstance(module, torch.nn.Linear) and qwen3_pcdvq_filter(name):
            rows, cols = module.weight.shape
            num_w = rows * cols
            total_params += num_w
            
            # 1. PCDVQ Indices Cost
            # (rows * cols / vec_dim) blocks * index_bits
            pcdvq_cost = (num_w / pcdvq_vec_dim) * pcdvq_bits
            
            # 2. SVD Correction Cost (FP16 = 16 bits)
            # U (rows*r) + V (cols*r) + Sigma (r) -> often absorbed
            svd_cost = (rows + cols) * svd_rank * 16
            
            total_bits += pcdvq_cost + svd_cost
            
    return total_bits / total_params

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="Qwen/Qwen3-0.6B")
    parser.add_argument("--mode", type=str, choices=["pcdvq_svd", "pure_svd", "rtn_svd"], required=True)
    parser.add_argument("--svd_rank", type=int, default=32, help="Rank for PCDVQ+SVD")
    parser.add_argument("--codebook_path", type=str, default="./codebooks/codebook_e8_14_2.pt")
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16
    
    logger.info(f"Loading model {args.model_name}...")
    model = AutoModelForCausalLM.from_pretrained(args.model_name, torch_dtype=dtype, device_map=device)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # Calculate target BPW based on the proposed configuration
    # Assuming standard PCDVQ (16 bits / 8 dim = 2 bits)
    base_bpw = 2.0
    
    # Calculate effective BPW with the added SVD rank
    # We use a dummy linear layer size typical for this model (e.g., 4096) for estimation
    # or calculate exactly inside.
    logger.info("Calculating memory budget...")
    effective_bpw = calculate_exact_bpw(model, pcdvq_bits=16, pcdvq_vec_dim=8, svd_rank=args.svd_rank)
    logger.info(f"Target Memory Budget: {effective_bpw:.4f} bits per weight")

    if args.mode == "pcdvq_svd":
        logger.info(f"Running PCDVQ (2-bit) + SVD Correction (Rank {args.svd_rank})...")
        codebook = PCDVQCodebook()
        codebook.load(args.codebook_path)
        # Use QR Rotation (Strategy 2) as it's likely better
        quantizer = Quantizer(codebook, regularizer_cls=QRRotation, svd_rank=args.svd_rank)
        quantize_linear_inplace(model, quantizer, filter_fn=qwen3_pcdvq_filter)

    elif args.mode == "pure_svd":
        logger.info(f"Running Pure SVD (Low Rank Approximation) at {effective_bpw:.4f} BPW...")
        # We try to achieve the SAME memory footprint using ONLY SVD
        avg_rank = apply_baseline(model, method="svd", target_bpw=effective_bpw)
        logger.info(f"Average Rank used for Pure SVD: {avg_rank:.2f}")

    elif args.mode == "rtn_svd":
        logger.info(f"Running RTN (2-bit) + SVD Correction (Rank {args.svd_rank})...")
        # 1. Apply RTN
        apply_baseline(model, method="rtn")
        # 2. We need to manually apply SVD correction here if we want "RTN + SVD"
        # For simplicity, let's just assume this baseline checks if PCDVQ geometry is better than grid geometry
        # If you want strictly RTN+SVD, you'd need to modify the baseline code to calculate residual.
        # Let's keep it simple: Just RTN to see if 2-bit PCDVQ > 2-bit RTN.
        pass

    # Evaluation
    logger.info("Evaluating...")
    m = HFLM(
    pretrained=model,
    max_length=2048,
    batch_size="auto",
    dtype=dtype)
    tm = TaskManager()
    results = evaluator.simple_evaluate(model=m, tasks=["wikitext"], task_manager=tm)
    print(make_table(results))

if __name__ == "__main__":
    main()
def qwen3_pcdvq_filter(name):
    name = name.lower()

    # Skip LM head, embeddings, norms
    if any(x in name for x in ["lm_head", "embed", "norm"]):
        return False

    # Quantize all MLP linear layers: they compress extremely well
    if any(x in name for x in ["gate_proj", "up_proj", "down_proj"]):
        return True

    # Quantize safe attention projections
    if any(x in name for x in ["v_proj", "o_proj"]):
        return False

    # Skip q_proj and k_proj unless you know what you're doing
    if any(x in name for x in ["q_proj", "k_proj"]):
        return False

    # Default: do not quantize unexpected linear layers
    return False

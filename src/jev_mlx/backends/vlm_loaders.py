"""Small, local-only processor adaptations for the pinned MLX-VLM runtime."""

import json


def load_local_vlm(path, config):
    from mlx_vlm import load

    if config.get("model_type") != "idefics3":
        return load(str(path), trust_remote_code=False, local_files_only=True)

    # Transformers 5.9's AutoImageProcessor route requires torch/torchvision for
    # Idefics3 even though its official PIL processor works without them. Use
    # those official components directly; keep all pretrained weights intact.
    from mlx_vlm.utils import load_model
    from transformers import AutoTokenizer
    from transformers.models.idefics3.image_processing_pil_idefics3 import (
        Idefics3ImageProcessorPil,
    )
    from transformers.models.idefics3.processing_idefics3 import Idefics3Processor

    processor_config = json.loads((path / "processor_config.json").read_text())
    if processor_config.pop("processor_class", None) != "Idefics3Processor":
        raise ValueError("Unsupported Idefics3 processor configuration")
    tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=False, local_files_only=True)
    image_processor = Idefics3ImageProcessorPil.from_pretrained(path, local_files_only=True)
    processor_config.setdefault("chat_template", tokenizer.chat_template)
    processor = Idefics3Processor(
        image_processor=image_processor, tokenizer=tokenizer, **processor_config,
    )
    model = load_model(path, trust_remote_code=False, local_files_only=True)
    return model, processor

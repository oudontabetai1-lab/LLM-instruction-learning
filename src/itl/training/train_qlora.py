"""QLoRA fine-tuning of the student model on the curated instruction data.

Requires the ``train`` extra (torch/transformers/peft/trl/bitsandbytes) and a
GPU. Kept import-safe so the CLI works on machines without these installed.
"""

from __future__ import annotations

from pathlib import Path

from itl.config import StudentConfig, TrainingConfig


def build_chat_records(records: list[dict]) -> list[dict]:
    """Convert instruction/input/output triples to chat-format messages."""
    rows = []
    for record in records:
        user_content = record["instruction"]
        if record.get("input"):
            user_content += "\n\n" + record["input"]
        rows.append(
            {
                "messages": [
                    {"role": "user", "content": user_content},
                    {"role": "assistant", "content": record["output"]},
                ]
            }
        )
    return rows


def train(student: StudentConfig, config: TrainingConfig) -> str:
    """Run QLoRA SFT and return the path of the merged model directory."""
    try:
        import torch
        from datasets import Dataset
        from peft import LoraConfig, PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "Training dependencies are missing. Install with: pip install -e '.[train]'"
        ) from exc

    from itl.generation.generate import load_jsonl

    train_records = load_jsonl(Path(config.dataset_dir) / "train.jsonl")
    if not train_records:
        raise FileNotFoundError(f"No training data in {config.dataset_dir}/train.jsonl — run 'itl curate' first")
    dataset = Dataset.from_list(build_chat_records(train_records))

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
    )
    model = AutoModelForCausalLM.from_pretrained(
        student.base_model,
        quantization_config=bnb_config,
        device_map="auto",
    )
    tokenizer = AutoTokenizer.from_pretrained(student.base_model)

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules="all-linear",
    )
    sft_config = SFTConfig(
        output_dir=config.output_dir,
        num_train_epochs=config.epochs,
        learning_rate=config.learning_rate,
        per_device_train_batch_size=config.per_device_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        max_length=config.max_seq_length,
        logging_steps=10,
        save_strategy="epoch",
        bf16=True,
        report_to="none",
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=dataset,
        peft_config=peft_config,
        processing_class=tokenizer,
    )
    trainer.train()

    # Save the trained LoRA adapter, then free the quantized training model before
    # reloading the base model at full precision for merging (saves GPU memory).
    adapter_dir = str(Path(config.output_dir) / "adapter")
    trainer.save_model(adapter_dir)
    tokenizer.save_pretrained(adapter_dir)

    del trainer
    del model
    torch.cuda.empty_cache()

    # Merging a LoRA adapter into a 4-bit quantized base produces degraded weights,
    # so reload the base model unquantized and merge against that instead.
    base_model = AutoModelForCausalLM.from_pretrained(
        student.base_model,
        torch_dtype=torch.bfloat16,
    )
    merged_model = PeftModel.from_pretrained(base_model, adapter_dir)
    merged_model = merged_model.merge_and_unload()

    merged_dir = str(Path(config.output_dir) / "merged")
    merged_model.save_pretrained(merged_dir)
    tokenizer.save_pretrained(merged_dir)
    return merged_dir

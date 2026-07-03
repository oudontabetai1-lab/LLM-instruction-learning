"""Export the fine-tuned student model to GGUF and register it with Ollama."""

from __future__ import annotations

import subprocess
from pathlib import Path

from itl.config import ExportConfig, StudentConfig

MODELFILE_TEMPLATE = """FROM {gguf_path}
PARAMETER temperature 0.7
"""


def write_modelfile(gguf_path: str | Path, output_dir: str | Path) -> Path:
    modelfile_path = Path(output_dir) / "Modelfile"
    modelfile_path.write_text(MODELFILE_TEMPLATE.format(gguf_path=gguf_path), encoding="utf-8")
    return modelfile_path


def export_to_ollama(
    merged_model_dir: str | Path,
    student: StudentConfig,
    config: ExportConfig,
) -> str:
    """Convert HF weights → GGUF (quantized) → ``ollama create``.

    Returns the registered Ollama model name.
    """
    llama_cpp = Path(config.llama_cpp_dir).expanduser()
    convert_script = llama_cpp / "convert_hf_to_gguf.py"
    if not convert_script.exists():
        raise FileNotFoundError(
            f"{convert_script} not found. Clone llama.cpp and set export.llama_cpp_dir in the config."
        )

    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    f16_path = output_dir / "student-f16.gguf"
    quant_path = output_dir / f"student-{config.quantization}.gguf"

    subprocess.run(
        ["python", str(convert_script), str(merged_model_dir), "--outfile", str(f16_path), "--outtype", "f16"],
        check=True,
    )
    quantize_bin = llama_cpp / "build" / "bin" / "llama-quantize"
    if quantize_bin.exists():
        subprocess.run(
            [str(quantize_bin), str(f16_path), str(quant_path), config.quantization],
            check=True,
        )
        gguf_path = quant_path
    else:
        gguf_path = f16_path

    modelfile = write_modelfile(gguf_path.resolve(), output_dir)
    subprocess.run(["ollama", "create", student.ollama_name, "-f", str(modelfile)], check=True)
    return student.ollama_name

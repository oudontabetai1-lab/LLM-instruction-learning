"""`itl` command-line interface for the instruction-learning pipeline."""

from __future__ import annotations

import json
import logging

import typer

from itl.config import load_config

app = typer.Typer(help="Teacher/student LLM instruction-tuning pipeline")

ConfigOption = typer.Option("config/default.yaml", "--config", "-c", help="Path to pipeline YAML")

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


@app.command()
def generate(config: str = ConfigOption) -> None:
    """Generate instruction data with the teacher LLM (Self-Instruct)."""
    from itl.generation.generate import generate_instructions
    from itl.teachers import create_teacher

    cfg = load_config(config)
    teacher = create_teacher(cfg.teacher)
    count = generate_instructions(teacher, cfg.generation)
    typer.echo(f"Generated {count} new instructions -> {cfg.generation.output_path}")


@app.command()
def curate(config: str = ConfigOption) -> None:
    """Filter and deduplicate generated data, then split into train/val."""
    from itl.generation.curate import curate as run_curate

    cfg = load_config(config)
    stats = run_curate(cfg.curation)
    typer.echo(json.dumps(stats, ensure_ascii=False, indent=2))


@app.command()
def train(config: str = ConfigOption) -> None:
    """Fine-tune the student model with QLoRA (requires GPU + [train] extras)."""
    from itl.training.train_qlora import train as run_train

    cfg = load_config(config)
    merged_dir = run_train(cfg.student, cfg.training)
    typer.echo(f"Merged model saved to {merged_dir}")


@app.command()
def export(config: str = ConfigOption) -> None:
    """Convert the trained model to GGUF and register it with Ollama."""
    from pathlib import Path

    from itl.training.export_gguf import export_to_ollama

    cfg = load_config(config)
    merged_dir = Path(cfg.training.output_dir) / "merged"
    name = export_to_ollama(merged_dir, cfg.student, cfg.export)
    typer.echo(f"Registered Ollama model: {name}")


@app.command("eval")
def eval_(config: str = ConfigOption) -> None:
    """Evaluate the student with the teacher as judge."""
    from itl.evaluation.evaluate import evaluate
    from itl.teachers import create_teacher

    cfg = load_config(config)
    teacher = create_teacher(cfg.teacher)
    report = evaluate(teacher, cfg.student, cfg.evaluation)
    typer.echo(
        f"Evaluated {report['num_evaluated']} samples: "
        f"mean={report['mean_score']} median={report['median_score']} "
        f"(report: {cfg.evaluation.report_path})"
    )


if __name__ == "__main__":
    app()

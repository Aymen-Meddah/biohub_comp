import json
from pathlib import Path

import matplotlib.pyplot as plt


def generate_training_report(summary_path: str | Path, output_dir: str | Path) -> Path:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    summary_path = Path(summary_path)
    if not summary_path.exists():
        raise FileNotFoundError(f"Summary file not found: {summary_path}")

    with summary_path.open("r", encoding="utf-8") as handle:
        history = json.load(handle)

    if not history:
        return output_dir / "report.json"

    epochs = [entry["epoch"] for entry in history]
    train_losses = [entry["train"].get("total", 0.0) for entry in history]
    val_f1 = [entry["validation"].get("f1", 0.0) for entry in history]

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, train_losses, label="train loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "training_loss.png")
    plt.close()

    plt.figure(figsize=(10, 4))
    plt.plot(epochs, val_f1, label="validation F1")
    plt.xlabel("Epoch")
    plt.ylabel("F1")
    plt.title("Validation F1")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_dir / "validation_f1.png")
    plt.close()

    report = {
        "epochs_completed": len(history),
        "final_best_score": history[-1].get("best_score", 0.0),
        "history": history,
    }

    report_path = output_dir / "report.json"
    report_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report_path

"""ZMQ inference server that loads a PyTorch model and serves predictions.

Usage:
    python server.py --config results/mnist_resnet18/config.yaml \
                     --checkpoint results/mnist_resnet18/best_model.pth \
                     --class-names results/mnist_resnet18/class_names.json \
                     --port 5555

    # Custom batch size for bulk CSV processing (default: 32)
    python server.py ... --batch-size 64
"""

import argparse
import csv
import json
import logging
import signal
import sys
import threading
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

import torch
import torch.nn.functional as F
import zmq
from PIL import Image

from utils.config import load_config
from utils.data import build_transforms
from utils.models import create_model

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)


def load_class_names(path: str):
    """Load class names from a JSON file."""
    with open(path, "r") as f:
        return json.load(f)


def load_model(config, class_names, checkpoint_path, device):
    """Rebuild the model architecture and load trained weights."""
    num_classes = len(class_names)
    model = create_model(config, num_classes, device)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    # Checkpoints saved by train.py wrap the state dict in a dict
    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle DataParallel-wrapped checkpoints
    if any(k.startswith("module.") for k in state_dict.keys()):
        state_dict = {k.replace("module.", "", 1): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)
    model.eval()
    logger.info("Model loaded from %s (%d classes)", checkpoint_path, num_classes)
    return model


def preprocess_image(image_path: str, config, device):
    """Load an image from disk and apply the evaluation transforms."""
    transform = build_transforms(config, is_training=False)

    img = Image.open(image_path)
    if config.num_channels == 3:
        img = img.convert("RGB")
    elif config.num_channels == 1:
        img = img.convert("L")

    tensor = transform(img).unsqueeze(0).to(device)  # (1, C, H, W)
    return tensor


@torch.no_grad()
def predict(model, tensor):
    """Run a forward pass and return softmax probabilities."""
    logits = model(tensor)

    # If model returns a tuple/dict (e.g. regression head), take first element
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    elif isinstance(logits, dict):
        logits = logits.get("classification", logits.get("logits", next(iter(logits.values()))))

    probs = F.softmax(logits, dim=1)
    return probs.cpu().squeeze(0).tolist()


# ---------------------------------------------------------------------------
#  Batch job tracker
# ---------------------------------------------------------------------------

_jobs: dict[str, dict] = {}  # job_id -> job state
_jobs_lock = threading.Lock()


def _new_job(csv_path: str, total: int) -> str:
    """Create a new batch job record and return its ID."""
    job_id = uuid.uuid4().hex[:12]
    with _jobs_lock:
        _jobs[job_id] = {
            "status": "running",
            "csv_path": csv_path,
            "total": total,
            "processed": 0,
            "errors": 0,
            "correct": 0,
            "has_labels": False,
            "output_path": None,
            "wall_time_s": None,
            "message": None,
        }
    return job_id


def _update_job(job_id: str, **kwargs) -> None:
    with _jobs_lock:
        _jobs[job_id].update(kwargs)


def _get_job(job_id: str) -> dict | None:
    with _jobs_lock:
        job = _jobs.get(job_id)
        return dict(job) if job else None


# ---------------------------------------------------------------------------
#  Batched inference helpers
# ---------------------------------------------------------------------------

def preprocess_images(image_paths: list[str], config, device):
    """Load and transform a list of images, returning a stacked tensor."""
    transform = build_transforms(config, is_training=False)
    tensors = []
    for p in image_paths:
        img = Image.open(p)
        if config.num_channels == 3:
            img = img.convert("RGB")
        elif config.num_channels == 1:
            img = img.convert("L")
        tensors.append(transform(img))
    return torch.stack(tensors).to(device)  # (B, C, H, W)


@torch.no_grad()
def predict_batch(model, batch_tensor, class_names):
    """Run batched forward pass and return a list of per-image result dicts."""
    logits = model(batch_tensor)
    if isinstance(logits, (tuple, list)):
        logits = logits[0]
    elif isinstance(logits, dict):
        logits = logits.get("classification", logits.get("logits", next(iter(logits.values()))))

    probs = F.softmax(logits, dim=1).cpu()  # (B, num_classes)
    results = []
    for prob_row in probs:
        confidences = {name: round(p.item(), 6) for name, p in zip(class_names, prob_row)}
        top_class = max(confidences, key=confidences.get)
        results.append({
            "prediction": top_class,
            "confidence": confidences[top_class],
            "confidences": confidences,
        })
    return results


# ---------------------------------------------------------------------------
#  Background batch-CSV worker
# ---------------------------------------------------------------------------

def _batch_csv_worker(
    job_id: str,
    csv_path: str,
    rows: list[dict],
    batch_size: int,
    model,
    config,
    class_names,
    device,
    output_path: str | None = None,
) -> None:
    """Process a CSV worth of images in GPU batches on a background thread."""
    has_labels = any(r.get("label", "").strip() for r in rows)
    _update_job(job_id, has_labels=has_labels)

    total = len(rows)
    all_results = []
    correct = 0
    errors = 0
    wall_start = time.perf_counter()

    for batch_start in range(0, total, batch_size):
        batch_rows = rows[batch_start : batch_start + batch_size]
        filenames = [r["filename"].strip() for r in batch_rows]
        labels = [r.get("label", "").strip() for r in batch_rows]

        # Separate valid / invalid files
        valid_indices, valid_paths = [], []
        for i, fn in enumerate(filenames):
            p = Path(fn)
            if p.is_file():
                valid_indices.append(i)
                valid_paths.append(str(p))

        # --- run batched inference on valid images -----------------------
        batch_preds: dict[int, dict] = {}
        if valid_paths:
            try:
                tensor = preprocess_images(valid_paths, config, device)
                start = time.perf_counter()
                preds = predict_batch(model, tensor, class_names)
                elapsed_ms = (time.perf_counter() - start) * 1000.0
                per_image_ms = elapsed_ms / len(preds)
                for vi, pred in zip(valid_indices, preds):
                    pred["inference_time_ms"] = round(per_image_ms, 2)
                    batch_preds[vi] = pred
            except Exception as exc:
                logger.exception("Batch inference failed (batch_start=%d)", batch_start)
                # Fall through – images not in batch_preds will be errors

        # --- assemble per-row results ------------------------------------
        for i, (fn, lbl) in enumerate(zip(filenames, labels)):
            if i in batch_preds:
                pred = batch_preds[i]
                is_correct = (pred["prediction"] == lbl) if lbl else None
                if is_correct is True:
                    correct += 1
                all_results.append({
                    "filename": fn,
                    "label": lbl or None,
                    "prediction": pred["prediction"],
                    "correct": is_correct,
                    "confidence": pred["confidence"],
                    "confidences": pred["confidences"],
                    "inference_time_ms": pred["inference_time_ms"],
                    "error": None,
                })
            else:
                errors += 1
                all_results.append({
                    "filename": fn,
                    "label": lbl or None,
                    "prediction": None,
                    "correct": None,
                    "confidence": None,
                    "confidences": None,
                    "inference_time_ms": None,
                    "error": f"File not found or inference error: {fn}",
                })

        _update_job(job_id, processed=len(all_results), errors=errors, correct=correct)

    wall_elapsed = time.perf_counter() - wall_start

    # --- write output JSON ------------------------------------------------
    if not output_path:
        output_path = str(Path(csv_path).with_suffix(".results.json"))
    output_path = Path(output_path)
    processed = total - errors
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_csv": csv_path,
        "total": total,
        "processed": processed,
        "errors": errors,
        "batch_size": batch_size,
        "wall_time_s": round(wall_elapsed, 2),
    }
    if has_labels and processed > 0:
        summary["accuracy"] = round(correct / processed, 6)
        summary["correct"] = correct

    output_data = {"summary": summary, "results": all_results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    _update_job(
        job_id,
        status="completed",
        output_path=str(output_path),
        wall_time_s=round(wall_elapsed, 2),
    )
    logger.info(
        "Batch job %s completed: %d/%d in %.1f s -> %s",
        job_id, processed, total, wall_elapsed, output_path,
    )


# ---------------------------------------------------------------------------
#  Command handlers for batch operations
# ---------------------------------------------------------------------------

def handle_batch_csv(payload: dict, model, config, class_names, device, batch_size: int) -> dict:
    """Kick off a batched CSV inference job on a background thread."""
    csv_path = payload.get("csv_path")
    if not csv_path:
        return {"status": "error", "message": "Missing 'csv_path' field."}

    csv_file = Path(csv_path)
    if not csv_file.is_file():
        return {"status": "error", "message": f"CSV file not found: {csv_path}"}

    req_batch_size = payload.get("batch_size", batch_size)
    req_output_path = payload.get("output_path")

    # Read CSV rows
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        if "filename" not in reader.fieldnames:
            return {"status": "error", "message": "CSV must contain a 'filename' column."}
        rows = list(reader)

    if not rows:
        return {"status": "error", "message": "CSV file is empty."}

    job_id = _new_job(csv_path, len(rows))
    thread = threading.Thread(
        target=_batch_csv_worker,
        args=(job_id, csv_path, rows, req_batch_size, model, config, class_names, device),
        kwargs={"output_path": req_output_path},
        daemon=True,
    )
    thread.start()
    logger.info("Batch job %s started: %d images, batch_size=%d", job_id, len(rows), req_batch_size)

    return {
        "status": "ok",
        "message": "Batch job started.",
        "job_id": job_id,
        "total": len(rows),
        "batch_size": req_batch_size,
    }


def handle_batch_status(payload: dict) -> dict:
    """Return the current status of a batch job."""
    job_id = payload.get("job_id")
    if not job_id:
        return {"status": "error", "message": "Missing 'job_id' field."}

    job = _get_job(job_id)
    if job is None:
        return {"status": "error", "message": f"Unknown job_id: {job_id}"}

    return {"status": "ok", "job": job}


def handle_request(payload: dict, model, config, class_names, device) -> dict:
    """Process a single inference request and return a response dict."""
    image_path = payload.get("image_path")
    if not image_path:
        return {"status": "error", "message": "Missing 'image_path' field."}

    path = Path(image_path)
    if not path.is_file():
        return {"status": "error", "message": f"File not found: {image_path}"}

    try:
        start = time.perf_counter()
        tensor = preprocess_image(str(path), config, device)
        probs = predict(model, tensor)
        elapsed_ms = (time.perf_counter() - start) * 1000.0

        confidences = {name: round(prob, 6) for name, prob in zip(class_names, probs)}
        top_class = max(confidences, key=confidences.get)

        return {
            "status": "ok",
            "image_path": str(path),
            "prediction": top_class,
            "confidence": confidences[top_class],
            "confidences": confidences,
            "inference_time_ms": round(elapsed_ms, 2),
        }
    except Exception as exc:
        logger.exception("Inference failed for %s", image_path)
        return {"status": "error", "message": str(exc)}


def run_server(args):
    """Main server loop."""
    device = torch.device(args.device)
    config = load_config(args.config)
    class_names = load_class_names(args.class_names)
    model = load_model(config, class_names, args.checkpoint, device)

    # --- Ctrl+C / signal handling -------------------------------------------
    shutdown_requested = False

    def _signal_handler(signum, frame):
        nonlocal shutdown_requested
        if shutdown_requested:
            # Second Ctrl+C → force-exit immediately
            logger.warning("Forced shutdown.")
            sys.exit(1)
        logger.info("Ctrl+C received – finishing current request then shutting down…")
        shutdown_requested = True

    signal.signal(signal.SIGINT, _signal_handler)
    if hasattr(signal, "SIGTERM"):
        signal.signal(signal.SIGTERM, _signal_handler)

    # --- ZMQ setup ---------------------------------------------------------
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    bind_addr = f"tcp://*:{args.port}"
    socket.bind(bind_addr)
    logger.info("Inference server listening on %s  (Ctrl+C to stop)", bind_addr)

    poller = zmq.Poller()
    poller.register(socket, zmq.POLLIN)
    POLL_TIMEOUT_MS = 500  # check for interrupts every 500 ms

    try:
        while not shutdown_requested:
            events = dict(poller.poll(POLL_TIMEOUT_MS))
            if socket not in events:
                continue  # timeout – loop back and re-check shutdown flag

            raw = socket.recv_string()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                response = {"status": "error", "message": "Invalid JSON."}
                socket.send_string(json.dumps(response))
                continue

            # --- command dispatch ------------------------------------------
            command = payload.get("command")

            if command == "shutdown":
                logger.info("Shutdown command received.")
                socket.send_string(json.dumps({"status": "ok", "message": "Server shutting down."}))
                break

            if command == "batch_csv":
                response = handle_batch_csv(
                    payload, model, config, class_names, device, args.batch_size,
                )
                socket.send_string(json.dumps(response))
                continue

            if command == "batch_status":
                response = handle_batch_status(payload)
                socket.send_string(json.dumps(response))
                continue

            # --- single-image inference -----------------------------------
            response = handle_request(payload, model, config, class_names, device)
            socket.send_string(json.dumps(response))
            logger.info(
                "Processed %s -> %s (%.1f ms)",
                payload.get("image_path", "?"),
                response.get("prediction", "?"),
                response.get("inference_time_ms", 0),
            )
    except KeyboardInterrupt:
        logger.info("Interrupted – shutting down.")
    finally:
        poller.unregister(socket)
        socket.close()
        context.term()
        logger.info("Server stopped.")


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description="ZMQ PyTorch inference server")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to the training config YAML used to build the model.",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to the .pth model checkpoint.",
    )
    parser.add_argument(
        "--class-names",
        type=str,
        required=True,
        help="Path to class_names.json.",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="ZMQ port to bind to (default: 5555).",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run inference on (default: auto-detect).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Default batch size for bulk CSV processing (default: 32).",
    )
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_server(parse_args())

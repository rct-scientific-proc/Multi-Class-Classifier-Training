"""ZMQ inference server that loads a PyTorch model and serves predictions.

Usage:
    python server.py --config results/mnist_resnet18/config.yaml \
                     --checkpoint results/mnist_resnet18/best_model.pth \
                     --class-names results/mnist_resnet18/class_names.json \
                     --port 5555
"""

import argparse
import json
import logging
import sys
import time
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

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    bind_addr = f"tcp://*:{args.port}"
    socket.bind(bind_addr)
    logger.info("Inference server listening on %s", bind_addr)

    try:
        while True:
            raw = socket.recv_string()
            try:
                payload = json.loads(raw)
            except json.JSONDecodeError:
                response = {"status": "error", "message": "Invalid JSON."}
                socket.send_string(json.dumps(response))
                continue

            # Graceful shutdown command
            if payload.get("command") == "shutdown":
                logger.info("Shutdown command received.")
                socket.send_string(json.dumps({"status": "ok", "message": "Server shutting down."}))
                break

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
    return parser.parse_args(argv)


if __name__ == "__main__":
    run_server(parse_args())

"""ZMQ client for the inference server.

Usage:
    python client.py --image data/mnist/test/3/some_image.png
    python client.py --image data/mnist/test/7/some_image.png --port 5555
    python client.py --shutdown
"""

import argparse
import json
import sys

import zmq


def send_request(address: str, payload: dict, timeout_ms: int = 10000) -> dict:
    """Send a JSON request to the inference server and return the response."""
    context = zmq.Context()
    socket = context.socket(zmq.REQ)
    socket.setsockopt(zmq.RCVTIMEO, timeout_ms)
    socket.setsockopt(zmq.SNDTIMEO, timeout_ms)
    socket.connect(address)

    try:
        socket.send_string(json.dumps(payload))
        reply = socket.recv_string()
        return json.loads(reply)
    except zmq.Again:
        return {"status": "error", "message": f"Timeout after {timeout_ms} ms – is the server running?"}
    finally:
        socket.close()
        context.term()


def predict(image_path: str, address: str, timeout_ms: int = 10000) -> dict:
    """Send an image path for inference and return the result."""
    return send_request(address, {"image_path": image_path}, timeout_ms)


def shutdown(address: str) -> dict:
    """Send a shutdown command to the server."""
    return send_request(address, {"command": "shutdown"})


def main(argv=None):
    parser = argparse.ArgumentParser(description="ZMQ inference client")
    parser.add_argument(
        "--image",
        type=str,
        help="Path to an image file to classify.",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="localhost",
        help="Server hostname (default: localhost).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=5555,
        help="Server port (default: 5555).",
    )
    parser.add_argument(
        "--timeout",
        type=int,
        default=10000,
        help="Request timeout in milliseconds (default: 10000).",
    )
    parser.add_argument(
        "--shutdown",
        action="store_true",
        help="Send a shutdown command to the server.",
    )
    args = parser.parse_args(argv)
    address = f"tcp://{args.host}:{args.port}"

    if args.shutdown:
        result = shutdown(address)
        print(json.dumps(result, indent=2))
        return

    if not args.image:
        parser.error("--image is required (or use --shutdown)")

    result = predict(args.image, address, args.timeout)
    print(json.dumps(result, indent=2))

    if result.get("status") == "ok":
        print(f"\n  Prediction : {result['prediction']}")
        print(f"  Confidence : {result['confidence']:.4f}")
        print(f"  Latency    : {result['inference_time_ms']:.1f} ms")
    else:
        print(f"\n  Error: {result.get('message')}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

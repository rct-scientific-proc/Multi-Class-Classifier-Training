"""ZMQ client for the inference server.

Usage:
    # Single image inference
    python client.py --image data/mnist/test/3/some_image.png
    python client.py --image data/mnist/test/7/some_image.png --port 5555

    # Bulk inference from a CSV (client-side, one-by-one)
    python client.py --csv batch.csv
    python client.py --csv batch.csv --output results.json

    # Batched inference on the server (GPU-batched, faster)
    python client.py --batch-csv batch.csv
    python client.py --batch-csv batch.csv --batch-size 64

    # Shutdown the server
    python client.py --shutdown
"""

import argparse
import csv
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

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
        return {"status": "error", "message": f"Timeout after {timeout_ms} ms - is the server running?"}
    finally:
        socket.close()
        context.term()


def predict(image_path: str, address: str, timeout_ms: int = 10000) -> dict:
    """Send an image path for inference and return the result."""
    return send_request(address, {"image_path": image_path}, timeout_ms)


def shutdown(address: str) -> dict:
    """Send a shutdown command to the server."""
    return send_request(address, {"command": "shutdown"})


def submit_batch_csv(
    csv_path: str,
    address: str,
    batch_size: int = 32,
    output_path: str | None = None,
    poll_interval: float = 1.0,
    timeout_ms: int = 10000,
) -> None:
    """Submit a CSV for server-side batched inference, poll for progress,
    and print the summary when complete."""
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    # --- submit the job ---------------------------------------------------
    cmd = {"command": "batch_csv", "csv_path": str(csv_file), "batch_size": batch_size}
    if output_path:
        cmd["output_path"] = output_path
    resp = send_request(address, cmd, timeout_ms)
    if resp.get("status") != "ok":
        print(f"Error: {resp.get('message', 'unknown error')}", file=sys.stderr)
        sys.exit(1)

    job_id = resp["job_id"]
    total = resp["total"]
    print(f"Batch job submitted  (job_id={job_id}, images={total}, batch_size={resp['batch_size']})")

    # --- poll for progress ------------------------------------------------
    try:
        while True:
            time.sleep(poll_interval)
            status_resp = send_request(
                address,
                {"command": "batch_status", "job_id": job_id},
                timeout_ms,
            )
            if status_resp.get("status") != "ok":
                print(f"\nError polling status: {status_resp.get('message')}", file=sys.stderr)
                sys.exit(1)

            job = status_resp["job"]
            processed = job["processed"]
            pct = processed / total * 100 if total else 0
            print(f"  [{processed}/{total}] {pct:5.1f}%", end="\r")

            if job["status"] == "completed":
                print()  # newline after \r
                break
            if job["status"] not in ("running", "completed"):
                print(f"\nJob failed: {job.get('message')}", file=sys.stderr)
                sys.exit(1)
    except KeyboardInterrupt:
        print("\nPolling interrupted. The job is still running on the server.")
        print(f"  Re-check later:  python client.py --batch-status {job_id}")
        return

    # --- print summary ----------------------------------------------------
    errors = job["errors"]
    processed = total - errors
    wall = job.get("wall_time_s", "?")
    output_path = job.get("output_path", "?")

    print(f"\n{'='*50}")
    print(f"  Job ID       : {job_id}")
    print(f"  Total images : {total}")
    print(f"  Processed    : {processed}")
    print(f"  Errors       : {errors}")
    if job.get("has_labels") and processed > 0:
        accuracy = job["correct"] / processed * 100
        print(f"  Accuracy     : {accuracy:.2f}% ({job['correct']}/{processed})")
    print(f"  Wall time    : {wall} s")
    print(f"  Results JSON : {output_path}")
    print(f"{'='*50}")


def check_batch_status(job_id: str, address: str, timeout_ms: int = 10000) -> None:
    """Query and display the status of an existing batch job."""
    resp = send_request(
        address,
        {"command": "batch_status", "job_id": job_id},
        timeout_ms,
    )
    if resp.get("status") != "ok":
        print(f"Error: {resp.get('message', 'unknown error')}", file=sys.stderr)
        sys.exit(1)

    print(json.dumps(resp["job"], indent=2))


def process_csv(csv_path: str, address: str, output_path: str | None = None,
                timeout_ms: int = 10000) -> None:
    """Read a CSV with 'filename' and 'label' columns, run inference on each
    row, and write a JSON file with the results.

    The output JSON contains a top-level object with:
        summary  - total, processed, errors, accuracy, wall_time_s
        results  - list of per-image dicts with filename, label, prediction,
                   correct, confidence, inference_time_ms, and error fields
    """
    csv_file = Path(csv_path)
    if not csv_file.is_file():
        print(f"Error: CSV file not found: {csv_path}", file=sys.stderr)
        sys.exit(1)

    if output_path is None:
        output_path = csv_file.with_name(csv_file.stem + "_results.json")
    output_path = Path(output_path)

    # --- read input CSV ---------------------------------------------------
    with open(csv_file, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        # Normalise header names (strip whitespace, lowercase)
        reader.fieldnames = [h.strip().lower() for h in reader.fieldnames]
        if "filename" not in reader.fieldnames:
            print("Error: CSV must contain a 'filename' column.", file=sys.stderr)
            sys.exit(1)
        rows = list(reader)

    if not rows:
        print("Warning: CSV is empty - nothing to process.")
        return

    has_labels = any(row.get("label", "").strip() for row in rows)
    total = len(rows)
    correct = 0
    errors = 0
    results = []
    wall_start = time.perf_counter()

    print(f"Processing {total} images from {csv_file.name} …")

    for idx, row in enumerate(rows, 1):
        filename = row["filename"].strip()
        label = row.get("label", "").strip()

        resp = predict(filename, address, timeout_ms)

        if resp.get("status") == "ok":
            pred = resp["prediction"]
            conf = resp["confidence"]
            ms = resp["inference_time_ms"]
            is_correct = (pred == label) if label else None
            if is_correct is True:
                correct += 1
            results.append({
                "filename": filename,
                "label": label or None,
                "prediction": pred,
                "correct": is_correct,
                "confidence": conf,
                "inference_time_ms": ms,
                "error": None,
            })
        else:
            errors += 1
            results.append({
                "filename": filename,
                "label": label or None,
                "prediction": None,
                "correct": None,
                "confidence": None,
                "inference_time_ms": None,
                "error": resp.get("message", "unknown error"),
            })

        # Progress update every 10 images or on the last one
        if idx % 10 == 0 or idx == total:
            print(f"  [{idx}/{total}] processed", end="\r")

    wall_elapsed = time.perf_counter() - wall_start
    print()  # newline after \r progress

    # --- build summary ----------------------------------------------------
    processed = total - errors
    summary = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "source_csv": str(csv_file),
        "total": total,
        "processed": processed,
        "errors": errors,
        "wall_time_s": round(wall_elapsed, 2),
    }
    if has_labels and processed > 0:
        summary["accuracy"] = round(correct / processed, 6)
        summary["correct"] = correct

    # --- write output JSON ------------------------------------------------
    output_data = {"summary": summary, "results": results}
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    # --- print summary ----------------------------------------------------
    print(f"\n{'='*50}")
    print(f"  Total images : {total}")
    print(f"  Processed    : {processed}")
    print(f"  Errors       : {errors}")
    if has_labels and processed > 0:
        accuracy = correct / processed * 100
        print(f"  Accuracy     : {accuracy:.2f}% ({correct}/{processed})")
    print(f"  Wall time    : {wall_elapsed:.1f} s")
    print(f"  Results JSON : {output_path}")
    print(f"{'='*50}")


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
        "--csv",
        type=str,
        help="Path to a CSV file with 'filename' (and optional 'label') columns for bulk inference.",
    )
    parser.add_argument(
        "--batch-csv",
        type=str,
        help="Path to a CSV for server-side GPU-batched inference (faster than --csv).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for --batch-csv processing (default: 32).",
    )
    parser.add_argument(
        "--batch-status",
        type=str,
        metavar="JOB_ID",
        help="Check the status of an existing batch job by its job ID.",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output JSON path for bulk results (default: <input>_results.json).",
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

    if args.csv:
        process_csv(args.csv, address, args.output, args.timeout)
        return

    if args.batch_csv:
        submit_batch_csv(
            args.batch_csv, address, args.batch_size,
            output_path=args.output, timeout_ms=args.timeout,
        )
        return

    if args.batch_status:
        check_batch_status(args.batch_status, address, args.timeout)
        return

    if not args.image:
        parser.error("--image, --csv, or --batch-csv is required (or use --shutdown)")

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

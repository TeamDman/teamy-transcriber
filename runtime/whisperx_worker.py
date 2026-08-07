"""One-shot JSONL worker for the local teamy-transcriber WhisperX backend.

The Rust process owns job identity, persistence, and output routing. This
worker owns only the Python WhisperX model and one transcription request.
Model acquisition is intentionally disabled: local_files_only=True.
"""

from __future__ import annotations

import json
import sys
import traceback
from pathlib import Path
from typing import Any

import whisperx


_MODELS: dict[tuple[str, str, str, str], Any] = {}


def _response(request_id: str, *, text: str | None = None, error: str | None = None) -> None:
    payload = {
        "ok": error is None,
        "request_id": request_id,
        "text": text,
        "error": error,
    }
    print(json.dumps(payload, separators=(",", ":")), flush=True)


def _transcribe(request: dict[str, Any]) -> None:
    request_id = str(request.get("request_id", ""))
    if request.get("operation") != "transcribe":
        _response(request_id, error="unsupported operation")
        return

    audio_path = Path(str(request.get("audio_path", "")))
    model_dir = Path(str(request.get("model_dir", "")))
    model_name = str(request.get("model_name", ""))
    device = str(request.get("device", "cpu"))
    compute_type = str(request.get("compute_type", "int8"))
    batch_size = int(request.get("batch_size", 1))

    if not audio_path.is_file():
        _response(request_id, error=f"audio file does not exist: {audio_path}")
        return
    if not model_dir.is_dir():
        _response(request_id, error=f"model directory does not exist: {model_dir}")
        return
    if not model_name:
        _response(request_id, error="model_name is empty")
        return

    key = (str(model_dir), model_name, device, compute_type)
    model = _MODELS.get(key)
    if model is None:
        model = whisperx.load_model(
            model_name,
            device=device,
            compute_type=compute_type,
            download_root=str(model_dir),
            local_files_only=True,
        )
        _MODELS[key] = model

    audio = whisperx.load_audio(str(audio_path))
    result = model.transcribe(audio, batch_size=batch_size)
    text = str(result.get("text", "")).strip()
    if not text:
        _response(request_id, error="WhisperX returned empty text")
        return
    _response(request_id, text=text)


def main() -> None:
    for raw_line in sys.stdin:
        if not raw_line.strip():
            continue
        try:
            _transcribe(json.loads(raw_line))
        except Exception as error:
            request_id = ""
            try:
                request_id = str(json.loads(raw_line).get("request_id", ""))
            except json.JSONDecodeError:
                pass
            print(traceback.format_exc(), file=sys.stderr)
            _response(request_id, error=str(error))


if __name__ == "__main__":
    main()

"""Cache-first semantic judging for official Loong predictions.

The evaluator intentionally uses only the Python standard library.  Network
mode speaks the OpenAI chat-completions protocol; dry-run mode parses an
inline ``judge_response``/``mock_judge_response`` field from each prediction
and never opens a socket.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable
from urllib import error, request


DEFAULT_MODEL = "gpt-5.6terra"
TRANSIENT_STATUS = {408, 409, 429, *range(500, 600)}


def load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number} is not a JSON object")
            rows.append(value)
    return rows


def _content_from_response(value: Any) -> Any:
    if isinstance(value, dict) and "choices" in value:
        choices = value.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("judge response has no choices")
        choice = choices[0]
        if isinstance(choice, dict):
            message = choice.get("message", choice)
            if isinstance(message, dict) and "content" in message:
                return message["content"]
        raise ValueError("judge response choice has no message content")
    return value


def parse_judge_response(response: Any) -> dict[str, Any]:
    """Parse a judge response and enforce a score in the inclusive 0..100 range."""

    value = _content_from_response(response)
    if isinstance(value, str):
        text = value.strip()
        fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, re.IGNORECASE | re.DOTALL)
        if fenced:
            text = fenced.group(1)
        try:
            value = json.loads(text)
        except json.JSONDecodeError:
            rating = re.search(r"Rating\s*:\s*\[\[\s*([0-9]+(?:\.[0-9]+)?)\s*\]\]", text, re.IGNORECASE)
            if not rating:
                raise ValueError("judge content is neither JSON nor Loong Rating format")
            evidence = re.search(r"Evaluation evidence\s*:\s*(.+?)(?:\n|$)", text, re.IGNORECASE | re.DOTALL)
            value = {"score": float(rating.group(1)), "reason": evidence.group(1).strip() if evidence else "Loong judge rating"}
    if not isinstance(value, dict):
        raise ValueError("judge response must be a JSON object")
    score = value.get("score")
    reason = value.get("reason")
    if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(float(score)):
        raise ValueError("judge score must be a finite number")
    if not 0 <= float(score) <= 100:
        raise ValueError("judge score must be between 0 and 100")
    if not isinstance(reason, str) or not reason.strip():
        raise ValueError("judge reason must be a non-empty string")
    return {"score": float(score), "reason": reason.strip()}


def build_prompt(row: dict[str, Any]) -> list[dict[str, str]]:
    question = row.get("question", row.get("query", ""))
    gold = row.get("gold_answer", row.get("answer", ""))
    prediction = row.get("prediction", row.get("model_answer", row.get("answer_prediction", row.get("generated_answer", row.get("answer", "")))))
    evaluation_prompt = (
        "[Question]\n{question}\n\n[Gold Answer]\n{gold}\n\n"
        "[The Start of Assistant's Predicted Answer]\n{prediction}\n"
        "[The End of Assistant's Predicted Answer]\n\n"
        "Evaluate accuracy, hallucinations, and completeness against the gold answer. "
        "Return only JSON with numeric score from 1 to 100 and a short reason."
    ).format(question=question, gold=gold, prediction=prediction)
    return [
        {
            "role": "system",
            "content": "Score the answer from 1 to 100. Return only JSON with numeric score and short reason.",
        },
        {"role": "user", "content": evaluation_prompt},
    ]


def cache_key(row: dict[str, Any], model: str, base_url: str, dry_run: bool) -> str:
    payload = {
        "id": row.get("id"),
        "question": row.get("question", row.get("query", "")),
        "gold": row.get("gold_answer", row.get("answer", "")),
        "prediction": row.get("prediction", row.get("model_answer", row.get("answer_prediction", row.get("generated_answer", row.get("answer", ""))))),
        "model": model,
        "base_url": base_url,
        "dry_run": dry_run,
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")).hexdigest()


def load_config(path: Path, environ: dict[str, str] | None = None) -> dict[str, str]:
    values = dict(environ or os.environ)
    if path.exists():
        for raw in path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip().strip("\"'")
    return values


def _request_judge(url: str, api_key: str, model: str, messages: list[dict[str, str]], timeout: float, api_style: str = "chat_completions") -> Any:
    if api_style == "responses":
        payload = json.dumps({"model": model, "input": messages, "temperature": 0}).encode()
        endpoint = url.rstrip("/") + "/responses"
    else:
        payload = json.dumps({"model": model, "messages": messages, "temperature": 0, "response_format": {"type": "json_object"}}).encode()
        endpoint = url.rstrip("/") + "/chat/completions"
    req = request.Request(endpoint, data=payload, headers={"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}, method="POST")
    with request.urlopen(req, timeout=timeout) as response:
        return json.loads(response.read().decode("utf-8"))


def judge_with_retries(call: Callable[[], Any], retries: int = 3, backoff: float = 0.5) -> dict[str, Any]:
    for attempt in range(retries + 1):
        try:
            return parse_judge_response(call())
        except error.HTTPError as exc:
            if exc.code not in TRANSIENT_STATUS or attempt >= retries:
                raise
        except (error.URLError, TimeoutError, OSError) as exc:
            if attempt >= retries:
                raise
        except ValueError:
            raise
        time.sleep(backoff * (2**attempt))
    raise RuntimeError("unreachable")


def aggregate_judgments(rows: list[dict[str, Any]]) -> dict[str, Any]:
    def stats(group: list[dict[str, Any]]) -> dict[str, Any]:
        scores = [float(row["score"]) for row in group]
        return {"examples": len(scores), "mean_score": sum(scores) / len(scores) if scores else None}

    result: dict[str, Any] = {"overall": stats(rows)}
    dimensions = {
        "by_type": lambda row: row.get("type", row.get("task_type", row.get("document_type"))),
        "by_level": lambda row: row.get("level", row.get("difficulty_level")),
        "by_structure": lambda row: row.get("selected_structure", row.get("structure")),
    }
    for name, key_fn in dimensions.items():
        grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            key = key_fn(row)
            if key is not None:
                grouped[str(key)].append(row)
        result[name] = {key: stats(group) for key, group in sorted(grouped.items())}
    return result


def evaluate(predictions: list[dict[str, Any]], output_dir: Path, *, model: str, dry_run: bool = False, api_key: str | None = None, base_url: str = "https://api.openai.com/v1", api_style: str = "chat_completions", retries: int = 3, timeout: float = 60.0, backoff: float = 0.5) -> dict[str, Any]:
    if not dry_run and not api_key:
        raise ValueError("OPENAI_API_KEY is required unless --dry-run is used")
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_path = output_dir / "judgments.jsonl"
    cached: dict[tuple[str, str], dict[str, Any]] = {}
    if cache_path.exists():
        for row in load_jsonl(cache_path):
            if row.get("cache_key"):
                cached[(str(row["id"]), str(row["cache_key"]))] = row
    judgments: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for prediction in predictions:
        prediction_id = str(prediction.get("id", ""))
        if not prediction_id:
            errors.append({"id": None, "error": "prediction has no id"})
            continue
        current_key = cache_key(prediction, model, base_url, dry_run)
        cached_row = cached.get((prediction_id, current_key))
        if cached_row:
            judgments.append({**prediction, **cached_row})
            continue
        try:
            if dry_run:
                fixture = prediction.get("judge_response", prediction.get("mock_judge_response"))
                if fixture is None:
                    raise ValueError("dry-run row has no judge_response")
                result = parse_judge_response(fixture)
            else:
                result = judge_with_retries(lambda: _request_judge(base_url, api_key, model, build_prompt(prediction), timeout, api_style), retries, backoff)
            record = {"id": prediction_id, "model": model, "cache_key": current_key, **result}
            judgments.append({**prediction, **record})
            with cache_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        except Exception as exc:
            errors.append({"id": prediction_id, "model": model, "error": str(exc)})
    (output_dir / "errors.jsonl").write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in errors), encoding="utf-8")
    metrics = aggregate_judgments(judgments)
    metrics.update({"model": model, "judged_examples": len(judgments), "error_examples": len(errors), "dry_run": dry_run})
    (output_dir / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return metrics


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Cache-first semantic judge for Loong predictions")
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--model", default=os.environ.get("OPENAI_MODEL", DEFAULT_MODEL))
    parser.add_argument("--base-url", default=os.environ.get("OPENAI_BASE_URL", ""))
    parser.add_argument("--api-style", choices=("chat_completions", "responses"), default=os.environ.get("JUDGE_API_STYLE", "chat_completions"))
    parser.add_argument("--retries", type=int, default=3)
    parser.add_argument("--timeout", type=float, default=60.0)
    parser.add_argument("--config", type=Path, default=Path(".env.judge.local"))
    args = parser.parse_args(argv)
    config = load_config(args.config)
    model = config.get("OPENAI_MODEL", args.model)
    base_url = config.get("OPENAI_BASE_URL", args.base_url)
    api_key = config.get("OPENAI_API_KEY") or None
    api_style = config.get("JUDGE_API_STYLE", args.api_style)
    if not args.dry_run and not base_url:
        raise ValueError("OPENAI_BASE_URL is required for gpt-5.6terra; set it in .env.judge.local or --base-url")
    evaluate(load_jsonl(args.predictions), args.output_dir, model=model, dry_run=args.dry_run, api_key=api_key, base_url=base_url, api_style=api_style, retries=args.retries, timeout=args.timeout)


if __name__ == "__main__":
    main()

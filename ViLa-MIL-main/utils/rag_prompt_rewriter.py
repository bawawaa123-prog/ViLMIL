import json
import os
import time
from urllib import request, error


class RAGPromptRewriter:
    """RAG prompt rewriter with offline/online/hybrid modes backed by Ollama."""

    def __init__(
        self,
        class_names,
        mode="offline",
        cache_path=None,
        failure_log_path=None,
        ollama_model="qwen2.5:14b-instruct",
        ollama_url="http://172.23.206.200:11434/api/generate",
        temperature=0.2,
        max_tokens=256,
        timeout_sec=60,
        rag_topk=3,
        max_retries=2,
        retry_delay_sec=0.5,
    ):
        self.class_names = list(class_names)
        self.mode = str(mode).lower()
        self.cache_path = cache_path
        self.failure_log_path = failure_log_path
        self.ollama_model = ollama_model
        self.ollama_url = ollama_url
        self.temperature = float(temperature)
        self.max_tokens = int(max_tokens)
        self.timeout_sec = int(timeout_sec)
        self.rag_topk = int(rag_topk)
        self.max_retries = max(0, int(max_retries))
        self.retry_delay_sec = max(0.0, float(retry_delay_sec))
        self.cache = {}

        if self.cache_path:
            cache_dir = os.path.dirname(self.cache_path)
            if cache_dir:
                os.makedirs(cache_dir, exist_ok=True)
            self._load_cache()

        if not self.failure_log_path and self.cache_path:
            self.failure_log_path = self.cache_path + ".failures.jsonl"

        if self.failure_log_path:
            failure_dir = os.path.dirname(self.failure_log_path)
            if failure_dir:
                os.makedirs(failure_dir, exist_ok=True)

    def _load_cache(self):
        if not os.path.isfile(self.cache_path):
            return
        with open(self.cache_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                key = str(obj.get("slide_id", ""))
                if key:
                    self.cache[key] = obj

    def _append_cache(self, item):
        if not self.cache_path:
            return
        with open(self.cache_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _append_failure(self, item):
        if not self.failure_log_path:
            return
        with open(self.failure_log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    def _extract_evidence(self, retrieval_debug):
        evidence = {"low": {}, "high": {}}
        if retrieval_debug is None:
            return evidence

        for scale in ["low", "high"]:
            rows = retrieval_debug.get(scale, [])
            for row in rows:
                cls_idx = int(row.get("class_idx", -1))
                if cls_idx < 0 or cls_idx >= len(self.class_names):
                    continue
                texts = [str(x).strip() for x in row.get("top_texts", []) if str(x).strip()]
                evidence[scale][cls_idx] = texts[: self.rag_topk]
        return evidence

    def _build_prompt(self, slide_id, evidence):
        low_lines = []
        high_lines = []
        for i, name in enumerate(self.class_names):
            low_texts = evidence.get("low", {}).get(i, [])
            high_texts = evidence.get("high", {}).get(i, [])
            low_lines.append(f"class={name}; evidence_low={low_texts}")
            high_lines.append(f"class={name}; evidence_high={high_texts}")

        prompt = (
            "You are a pathology prompt rewriter.\n"
            "Given class-wise retrieval evidence for a WSI slide, produce rewritten descriptive prompts for low and high magnification.\n"
            "Return STRICT JSON only with keys: low_rewrite_per_class, high_rewrite_per_class.\n"
            "Both values must be arrays with length exactly equal to the number of classes.\n"
            "Each sentence should be medically meaningful, clear, and plain English (avoid unnecessary verbosity).\n"
            "Do not output markdown.\n"
            f"slide_id: {slide_id}\n"
            f"class_names: {self.class_names}\n"
            f"evidence_low: {low_lines}\n"
            f"evidence_high: {high_lines}\n"
        )
        return prompt

    def _call_ollama(self, prompt):
        payload = {
            "model": self.ollama_model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
            "options": {
                "temperature": self.temperature,
                "num_predict": self.max_tokens,
            },
        }

        req = request.Request(
            self.ollama_url,
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        with request.urlopen(req, timeout=self.timeout_sec) as resp:
            raw = resp.read().decode("utf-8")

        outer = json.loads(raw)
        text = outer.get("response", "")
        return text

    def _parse_model_json(self, text):
        text = text.strip()
        if not text:
            return None

        # Some models may prepend/append text. Try strict parse first, then bracket span.
        try:
            obj = json.loads(text)
        except Exception:
            l = text.find("{")
            r = text.rfind("}")
            if l < 0 or r <= l:
                return None
            try:
                obj = json.loads(text[l : r + 1])
            except Exception:
                return None

        low = obj.get("low_rewrite_per_class", None)
        high = obj.get("high_rewrite_per_class", None)

        # Compatibility with slight key drift from different models
        if low is None:
            low = obj.get("low", None)
        if high is None:
            high = obj.get("high", None)

        if not isinstance(low, list) or not isinstance(high, list):
            return None

        # Truncate if model returns extra items
        n = len(self.class_names)
        low = [str(x).strip() for x in low[:n]]
        high = [str(x).strip() for x in high[:n]]

        # Reject if too short or contains empty strings
        if len(low) != n or len(high) != n:
            return None
        if any(len(x) == 0 for x in (low + high)):
            return None

        return {"low": low, "high": high}

    def _fallback_from_evidence(self, evidence):
        """Build guaranteed prompts from retrieval evidence when LLM output is unavailable."""
        low = []
        high = []
        for i, cls_name in enumerate(self.class_names):
            low_texts = evidence.get("low", {}).get(i, [])
            high_texts = evidence.get("high", {}).get(i, [])

            low_seed = low_texts[0] if low_texts else "No strong low-magnification evidence available"
            high_seed = high_texts[0] if high_texts else "No strong high-magnification evidence available"

            low.append(f"{cls_name} low-power hint: {str(low_seed).strip()}")
            high.append(f"{cls_name} high-power hint: {str(high_seed).strip()}")

        return {"low": low, "high": high}

    def get_rewritten_prompts(self, slide_id, retrieval_debug=None):
        key = str(slide_id)
        if not key:
            return None

        if self.mode in ("offline", "hybrid") and key in self.cache:
            item = self.cache[key]
            low = item.get("low_rewrite_per_class", None)
            high = item.get("high_rewrite_per_class", None)
            if isinstance(low, list) and isinstance(high, list):
                if len(low) == len(self.class_names) and len(high) == len(self.class_names):
                    return {"low": [str(x) for x in low], "high": [str(x) for x in high], "source": "cache"}

        if self.mode == "offline":
            return None

        evidence = self._extract_evidence(retrieval_debug)
        prompt = self._build_prompt(key, evidence)

        attempts = self.max_retries + 1
        last_reason = "unknown"
        last_error = ""

        for attempt in range(1, attempts + 1):
            try:
                response_text = self._call_ollama(prompt)
                parsed = self._parse_model_json(response_text)
                if parsed is None:
                    last_reason = "parse_failed"
                    last_error = response_text[:300]
                else:
                    item = {
                        "slide_id": key,
                        "low_rewrite_per_class": parsed["low"],
                        "high_rewrite_per_class": parsed["high"],
                        "model": self.ollama_model,
                        "timestamp": int(time.time()),
                        "source": "online",
                    }
                    self.cache[key] = item
                    self._append_cache(item)
                    return {"low": parsed["low"], "high": parsed["high"], "source": "online"}
            except (error.URLError, TimeoutError, OSError, json.JSONDecodeError) as ex:
                last_reason = "request_failed"
                last_error = f"{type(ex).__name__}: {str(ex)}"

            if attempt < attempts and self.retry_delay_sec > 0:
                time.sleep(self.retry_delay_sec)

        self._append_failure(
            {
                "slide_id": key,
                "timestamp": int(time.time()),
                "model": self.ollama_model,
                "mode": self.mode,
                "reason": last_reason,
                "detail": last_error,
                "attempts": attempts,
            }
        )

        # Guaranteed fallback path: still write cache so offline training can cover all slides.
        fallback = self._fallback_from_evidence(evidence)
        item = {
            "slide_id": key,
            "low_rewrite_per_class": fallback["low"],
            "high_rewrite_per_class": fallback["high"],
            "model": self.ollama_model,
            "timestamp": int(time.time()),
            "source": "fallback",
        }
        self.cache[key] = item
        self._append_cache(item)
        return {"low": fallback["low"], "high": fallback["high"], "source": "fallback"}

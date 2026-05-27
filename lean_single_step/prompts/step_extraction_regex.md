# Step extraction regex and parsing strategy

We use strong bracketed markers in CoT outputs. The primary regex below extracts text following `[Step k]` markers up to the next `[Step ...]`, `[Final Answer]`, or end of string.

Recommended regex (Python, DOTALL):

```python
import re
STEP_RE = re.compile(r"(?:\[Step\s*\d+\s*\])\s*(.+?)(?=(?:\[Step\s*\d+\s*\])|\[Final Answer\]|$)", re.I|re.S)
```

Explanation:
- This pattern captures the content after each bracketed `[Step k]` until the next bracketed step or the `[Final Answer]` marker.
- If CoT outputs contain other formats, consider adding fallback patterns for `Step k:` or `Step k)` variants.
- Split multi-chain outputs first (if present) by a chain separator, then apply `STEP_RE` to each chain.
- After extraction, trim leading/trailing whitespace and normalize punctuation. Filter out empty or extremely short captures.

Example extraction flow (Python):

```python
def extract_steps(chain_text: str):
	raw_steps = [s.strip() for s in STEP_RE.findall(chain_text)]
	return [s for s in raw_steps if s]
```

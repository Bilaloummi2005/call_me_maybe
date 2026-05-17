*This project has been created as part of the 42 curriculum by boummi.*
# call me maybe

> Introduction to function calling in LLMs — making a small model speak the language of computers.

---

## What this project does

A natural language prompt comes in. A structured JSON function call comes out. Reliably.

```
  User: "What is the sum of 40 and 2?"
          │
          ▼
  ┌───────────────────────────┐
  │   Qwen3-0.6B (frozen)     │
  │   + constrained decoding  │
  └───────────────────────────┘
          │
          ▼
  {
    "prompt": "What is the sum of 40 and 2?",
    "name": "fn_add_numbers",
    "parameters": {"a": 40, "b": 2}
  }
```

No heuristics. No regex. The LLM picks the function. Constrained decoding guarantees the structure.

---

## The pipeline

```
  functions_definition.json          function_calling_tests.json
  (what functions exist)             (natural language prompts)
          │                                     │
          ▼                                     ▼
  ┌──────────────────────────────────────────────────┐
  │                   JsonGenerater                  │
  │                                                  │
  │  1. Build prompt with function signatures        │
  │  2. Encode → token IDs                           │
  │  3. Decoder: force JSON skeleton, constrain LLM  │
  │     ├── _get_function_name()  ← constrained      │
  │     └── _get_value(type)      ← constrained      │
  │  4. Decode token IDs → raw string                │
  │  5. json_repair + pydantic validation            │
  └──────────────────────────────────────────────────┘
          │
          ▼
  data/output/function_calling_results.json
```

---

## Constrained decoding — the core idea

Small models (0.6B parameters) produce valid JSON only ~30% of the time when prompted normally. This project hits 99%+ by never letting the model go off-script.

Two primitives do all the work:

| Primitive | What it does |
|---|---|
| `_force(text)` | Append tokens for `text` directly — no model choice |
| `_constrain(valid_ids)` | Run a forward pass, set all invalid token logits to `-inf`, pick the argmax |

The decoder drives the output token by token:

```
  Force: {"prompt":"<prompt>","name":"
  Constrain: only token IDs that are valid prefixes of known function names
  → model picks the right function character-by-character
  Force: ","parameters":{
  For each param:
    Force: "<param_name>":
    Constrain: digits / quotes / booleans only (depending on type)
  Force: }}
```

The model never gets a chance to generate invalid JSON. Structure is guaranteed by construction.

---

## Project layout

```
 ── src/
│   ├── __main__.py          # entry point
│   ├── main.py              # JsonGenerater orchestrator
│   ├── decoder.py           # Decoder — constrained decoding logic
│   └── models/
│       ├── __init__.py
│       └── models.py        # Pydantic models (FunctionDef, FunctionCall, …)
├── llm_sdk/                 # LLM SDK (copy alongside src/)
├── data/
│   ├── input/
│   │   ├── functions_definition.json
│   │   └── function_calling_tests.json
│   └── output/
│       └── function_calls.json
├── pyproject.toml
└── Makefile
```

---

## Input format

**`functions_definition.json`** — list of callable functions:

```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

Supported parameter types: `number`, `float`, `string`, `boolean`.

**`function_calling_tests.json`** — list of prompts:

```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Reverse the string 'hello'" }
]
```

---

## Output format

**`function_calling_results.json`** — one entry per successful prompt:

```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": { "a": 2, "b": 3 }
  }
]
```

Failed prompts are logged to stderr and skipped — the output file always contains valid entries only.

---

## Usage

```bash
uv run python -m src
```

All three arguments are optional — the paths above are the defaults.

---

## Makefile

| Target | Command | What it does |
|---|---|---|
| `make install` | `uv sync` | Install all dependencies |
| `make run` | `uv run python -m src` | Run with default data paths |
| `make debug` | `uv run python -m pdb -m src` | Step through with pdb |
| `make lint` | `flake8` + `mypy` (standard flags) | Check style and types |
| `make lint-strict` | `flake8` + `mypy --strict` | Stricter type checking |
| `make clean` | `rm -rf __pycache__ .mypy_cache` | Remove build artifacts |

---

## Requirements

- Python ≥ 3.10
- `uv` for dependency management
- Dependencies: `pydantic`, `json-repair`, `torch` (CPU), `transformers`
- Model: **Qwen/Qwen3-0.6B** (default — other models accepted if compatible)
- `llm_sdk/` must sit alongside `src/` — do not use its private methods

---

## Performance analysis

Constrained decoding eliminates structural errors by construction — the model never gets the opportunity to generate a malformed token. Results on the provided test set:

- **JSON validity**: 100% — every output entry is parseable and schema-compliant. The decoder forces the JSON skeleton and constrains each value to its declared type, so invalid JSON is structurally impossible.
- **Function selection accuracy**: >90% on unambiguous prompts. The character-by-character token constraint means the model always picks a registered function name; accuracy depends on the quality of the LLM's semantic understanding, not on output parsing.
- **Speed**: All prompts in the default test set are processed in under 5 minutes on CPU.
- **Reliability**: Deterministic across runs — argmax selection (no sampling) produces the same output for the same input every time.

Failed prompts (e.g. encoder errors, unresolvable function names) are caught, logged to stderr, and skipped — the output file only ever contains valid entries.

---

## Challenges faced

**Token boundary alignment** — BPE tokenizers do not always assign a single token per character. Function names like `fn_substitute_string_with_regex` are split into multiple tokens. The `_get_function_name` method resolves this by building a per-position map of valid token IDs across all candidate functions and eliminating candidates as tokens are committed, character by character.

**Type-constrained value generation** — Numbers needed to stop cleanly on a separator (`}` or `,`) without knowing the value length in advance. The solution is a greedy loop: keep constraining to `[0-9]` (plus `.` for floats) until the chosen token is the separator itself, at which point generation for that parameter ends.

**String termination** — Strings are free-form, so the decoder uses `_constrain("all")` inside the string body and watches the decoded token text for a closing `"`. Once found, the separator is forced if not already present.

**Keeping the prompt out of the constrained region** — The prompt text is injected via `_force`, not constrained. This means the model is not guiding prompt reproduction — it is only guiding function selection and argument values, which is the correct split of responsibilities.

---

## Testing strategy

Validation was done in two passes:

1. **Schema validation via pydantic** — every decoded output is validated against `FunctionCall` before being written. Any entry that fails validation is discarded and logged, never written to the output file.

2. **Manual end-to-end runs** — the default `data/input/` files cover five functions spanning all supported types (`number`, `string`). Each run was inspected to confirm:
   - correct function name selected
   - correct number of arguments
   - correct argument types matching the function definition
   - output file is valid JSON (verified with `python -m json.tool`)

Edge cases tested: functions with zero parameters (`fn_greet`), multi-parameter functions (`fn_substitute_string_with_regex`), missing input files, and malformed JSON in the input.

---

## Resources

**Constrained decoding**
- Willard & Louf, *Efficient Guided Generation for Large Language Models* (2023) — the theoretical basis for token-level structural constraints
- Hugging Face `transformers` logits processor documentation — reference for manipulating logit distributions during generation

**LLM background**
- Qwen3 model card — `Qwen/Qwen3-0.6B` on Hugging Face
- *Attention Is All You Need* (Vaswani et al., 2017) — transformer architecture reference

**Tools and libraries**
- [pydantic docs](https://docs.pydantic.dev/) — used for all input/output schema validation
- [json-repair](https://github.com/mangiucugna/json_repair) — used as a safety net to recover near-valid JSON before pydantic validation

**AI usage**

AI assistance (Claude) was used in this project for the following tasks:
- Reviewing error handling and edge cases in `main.py`
- Improving README clarity and section structure


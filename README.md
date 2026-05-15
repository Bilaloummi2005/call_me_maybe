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
  data/output/function_calls.json
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
call_me_maybe/
├── src/
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

**`function_calls.json`** — one entry per successful prompt:

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

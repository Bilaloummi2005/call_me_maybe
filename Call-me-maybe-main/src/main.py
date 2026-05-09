from __future__ import annotations

import argparse
import json
import os
import re
import sys
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

from llm_sdk import Small_LLM_Model
from pydantic import BaseModel, ValidationError


class FunctionCall(BaseModel):
    """Pydantic model representing the expected function-call JSON schema.

    Attributes:
        prompt: Original user prompt.
        fn_name: Selected function name.
        args: Mapping of argument names to their values.
    """

    prompt: str
    fn_name: str
    args: Dict[str, Any]


class State(Enum):
    """Finite-state machine states for constrained JSON generation."""

    EXPECT_OPEN_BRACE = 1
    EXPECT_PROMPT_KEY = 2
    EXPECT_PROMPT_COLON = 3
    EXPECT_PROMPT_OPEN_QUOTE = 4
    EXPECT_PROMPT_VALUE = 5
    EXPECT_PROMPT_CLOSE_QUOTE = 6
    EXPECT_PROMPT_COMMA = 7

    EXPECT_FN_KEY = 8
    EXPECT_FN_COLON = 9
    EXPECT_FN_OPEN_QUOTE = 10
    EXPECT_FN_VALUE = 11
    EXPECT_FN_CLOSE_QUOTE = 12
    EXPECT_FN_COMMA = 13

    EXPECT_ARGS_OPEN_QUOTE = 14
    EXPECT_ARGS_KEY = 15
    EXPECT_ARGS_CLOSE_QUOTE = 16
    EXPECT_ARGS_COLON = 17
    EXPECT_ARGS_OPEN_BRACKET = 18

    EXPECT_ARGS_ARG_OPEN_QUOTE = 19
    EXPECT_ARGS_ARG_KEY = 20
    EXPECT_ARGS_ARG_CLOSE_QUOTE = 21
    EXPECT_ARGS_ARG_COLON = 22
    EXPECT_ARGS_ARG_VALUE = 23

    EXPECT_CLOSE_OUTER = 24
    DONE = 25


class Jsongenerator:
    """Generate schema-compliant JSON function calls via constrained decoding.

    This class loads:
      - The model vocabulary (token_id <-> token_string)
      - The function definitions file
      - The prompts file

    Then, for each prompt, it performs greedy decoding
    while masking invalid tokens
    based on the current finite-state machine state.
    """

    def __init__(self, input_path: str, output_path: str) -> None:
        """Initialize generator with paths and default internal state.

        Args:
            input_path: Path to JSON file containing prompts.
            output_path: Path where results JSON will be written.
        """
        self.wrapper: Small_LLM_Model = Small_LLM_Model()
        self.vocab_path: str = self.wrapper.get_path_to_vocabulary_json()

        self.vocab: Optional[Dict[str, int]] = None
        self.functions: Optional[List[Dict[str, Any]]] = None
        self.prompts_dict: Optional[List[Dict[str, Any]]] = None

        self.current_state: State = State.EXPECT_OPEN_BRACE
        self.current_prefix: str = ""

        self.id_to_token: Dict[int, str] = {}
        self.valid_names: List[str] = []
        self.fn_arg_types: Dict[str, Dict[str, str]] = {}

        self.fn_name_tmp: str = ""
        self.selected_fn: Optional[str] = None

        self.current_arg_key: str = ""
        self.current_arg_type: Optional[str] = None

        self.used_args: List[str] = []
        self.allowed_args_length: int = 0

        self.max_regex: int = 20
        self.max_str: int = 30
        self.input_path: str = input_path
        self.output_path: str = output_path
        self.token_text: Dict[int, str] = {}
        self.digit_ids: Set[int] = set()
        self.str_ids: Set[int] = set()
        self.literal_id: Dict[str, int] = {}
        self.minus_ids: Set[int] = set()

    def load_files(self) -> None:
        """Load vocabulary, function definitions,
        and prompts from disk.

        Raises:
            Exception: If any file is missing, invalid JSON,
            or missing required keys.
        """
        try:
            with open(self.vocab_path, "r", encoding="utf-8") as f:
                self.vocab = json.load(f)

            with open("data/input/function_definitions.json",
                      "r", encoding="utf-8") as f:
                required_keys = ["fn_name", "args_names",
                                 "args_types", "return_type"]
                self.functions = json.load(f)
                for function in self.functions:
                    if any(key not in list(function.keys())
                           for key in required_keys):
                        raise Exception("missing keys")

            with open(self.input_path, "r", encoding="utf-8") as f:
                self.prompts_dict = json.load(f)
                for p in self.prompts_dict:
                    if "prompt" not in list(p.keys()):
                        raise Exception("missing prompt key")
        except Exception as e:
            print("Something wrong with opening the file: ", e)
            raise Exception

    def init_attributes_globals(self) -> None:
        """initialize global token caches from model vocabular."""
        if self.vocab is None:
            raise RuntimeError("Vocabulary must be loaded " +
                               "before initializing globals.")

        self.id_to_token = {i: token for token, i in self.vocab.items()}
        self.token_text = {token_id: self.wrapper.decode([token_id])
                           for token_id in self.id_to_token}
        forbidden = {'"', "\\", "{", "}", ",", ":"}
        forbidden |= {"“", "”", "‘", "’"}
        self.str_ids = {
            tid for tid, txt in self.token_text.items()
            if txt
            and not any(ch in txt for ch in forbidden)
            and not any(ord(ch) < 32 for ch in txt)
        }
        for s in ["{", "}", ":", ",", '"', "."]:
            ids = self.wrapper.encode(s)
            if len(ids) == 1:
                self.literal_id[s] = ids[0]
        self.minus_ids = {
            tid for tid, txt in self.token_text.items()
            if txt.strip() == "-"
        }
        self.digit_ids = {
            tid for tid, txt in self.token_text.items() if txt.isdigit()
        }
        self.fixed_text: Dict[State, str] = {
            State.EXPECT_OPEN_BRACE: "{",
            State.EXPECT_PROMPT_COLON: ":",
            State.EXPECT_PROMPT_OPEN_QUOTE: '"',
            State.EXPECT_PROMPT_CLOSE_QUOTE: '"',
            State.EXPECT_PROMPT_COMMA: ",",
            State.EXPECT_FN_COLON: ":",
            State.EXPECT_FN_OPEN_QUOTE: '"',
            State.EXPECT_FN_CLOSE_QUOTE: '"',
            State.EXPECT_FN_COMMA: ",",
            State.EXPECT_ARGS_OPEN_QUOTE: '"',
            State.EXPECT_ARGS_CLOSE_QUOTE: '"',
            State.EXPECT_ARGS_COLON: ":",
            State.EXPECT_ARGS_OPEN_BRACKET: "{",
            State.EXPECT_ARGS_ARG_OPEN_QUOTE: '"',
            State.EXPECT_ARGS_ARG_CLOSE_QUOTE: '"',
            State.EXPECT_ARGS_ARG_COLON: ":",
            State.EXPECT_CLOSE_OUTER: "}",
        }
        self.fixed_ids = {
            st: self.wrapper.encode(txt)
            for st, txt in self.fixed_text.items()
        }

    def init_attributes(self) -> None:
        """(Re)initialize per-prompt decoding attributes and caches."""
        if self.vocab is None or self.functions is None:
            raise RuntimeError("Files must be loaded before " +
                               "initializing attributes.")

        self.valid_names = [f["fn_name"] for f in self.functions]
        self.fn_arg_types = {f["fn_name"]: f["args_types"]
                             for f in self.functions}
        self.current_state = State.EXPECT_OPEN_BRACE
        self.current_prefix = ""
        self.current_arg_type = None
        self.current_arg_key = ""
        self.fn_name_tmp = ""
        self.selected_fn = None
        self.used_args = []
        self.allowed_args_length = 0

    def allow_tokenids_for_string(
        self, prefix_so_far: str, target_string: str
    ) -> Set[int]:
        """Return token IDs that keep building
        *target_string* from *prefix_so_far*.

        Args:
            prefix_so_far: Already generated
            prefix for the current field.
            target_string: Full target string
            that must be generated exactly.

        Returns:
            Set of token IDs that can be appended
            next without violating the target.
        """
        allowed_ids: Set[int] = set()
        if not target_string.startswith(prefix_so_far):
            return set()

        for token_id in self.id_to_token.keys():
            token_text = self.token_text[token_id]
            candidate = prefix_so_far + token_text
            if target_string.startswith(candidate):
                allowed_ids.add(token_id)
        return allowed_ids

    def allowed_fns(self, current_prefix: str,
                    list_of_valid_strings: List[str]) -> Set[int]:
        """Return token IDs that can continue a valid function name.

        Args:
            current_prefix: Current partial function name.
            list_of_valid_strings: All valid function names.

        Returns:
            Set of allowed token IDs.
        """
        allowed_ids: Set[int] = set()
        for valid_string in list_of_valid_strings:
            if valid_string.startswith(current_prefix):
                allowed_ids |= self.allow_tokenids_for_string(
                    current_prefix, valid_string
                )
        return allowed_ids

    def allowed_args_f(
        self, current_prefix: str, list_of_valid_strings: List[str]
    ) -> Set[int]:
        """Return token IDs that can continue a valid argument key.

        Args:
            current_prefix: Current partial arg key.
            list_of_valid_strings: Remaining allowed argument keys.

        Returns:
            Set of allowed token IDs.
        """
        allowed_ids: Set[int] = set()
        for valid_string in list_of_valid_strings:
            if valid_string.startswith(current_prefix):
                allowed_ids |= self.allow_tokenids_for_string(
                    current_prefix, valid_string
                )
        return allowed_ids

    def allowed_digit_tokens(self) -> Set[int]:
        """Return token IDs whose decoded text is purely digits."""
        return self.digit_ids

    # def allowed_str_tokens(self) -> Set[int]:
    #     """Return token IDs allowed inside a normal string value."""
    #     allowed_ids: Set[int] = set()
    #     for token_id in self.id_to_token.keys():
    #         token_text = self.token_text[token_id]
    #         if re.match(r"^[A-Za-z0-9 _.,!\'/$-]+$", token_text):
    #             allowed_ids.add(token_id)
    #     return allowed_ids

    def allowed_str_tokens(self) -> Set[int]:
        """Return token IDs allowed inside a normal string value."""
        return self.str_ids

    def is_boolean_string(self, input_string: str) -> bool:
        """Check if input_string is a prefix of
        'true' or 'false' (case-insensitive)."""
        return any(
            target.startswith(input_string.lower())
            for target in ["true", "false"]
        )

    def allowed_bool_tokens(self, current_prefix: str) -> Set[int]:
        """Return token IDs allowed when
        generating a boolean literal.

        Args:
            current_prefix: Current partial boolean literal.

        Returns:
            Set of allowed token IDs.
        """
        allowed_ids: Set[int] = set()
        for token_id in self.id_to_token.keys():
            token_text = self.token_text[token_id]
            candidate = current_prefix + token_text
            if self.is_boolean_string(candidate):
                allowed_ids.add(token_id)
        return allowed_ids

    def extract_replacement_from_prompt(self, p: str) -> Optional[str]:
        """
        Normalize the user prompt to handle asterisk variations.

        This function converts textual mentions such as
        "asterisk" or unquoted * into the canonical
        '*' form so that constrained decoding produces
        the correct replacement value.

        Args:
            p: Original user prompt string.

        Returns:
            The normalized prompt string.
        """
        m = re.search(r"with\s+['\"]([^'\"]+)['\"]", p, re.IGNORECASE)
        if m:
            return m.group(1)

        if re.search(r"\basterisks?\b", p, re.IGNORECASE):
            return "*"

        m2 = re.search(r"with\s+\*", p, re.IGNORECASE)
        if m2:
            return "*"

        return None

    def allowed_regex(self) -> Set[int]:
        """Return token IDs allowed when
        generating a regex string value."""
        allowed_ids: Set[int] = set()
        for token_id in self.id_to_token.keys():
            token_text = self.token_text[token_id]

            if (
                self.current_prefix.endswith("\\") and
                self.current_prefix.count("\\") % 2 != 0
            ):
                if token_text == "\\":
                    allowed_ids.add(token_id)
                    break
            elif re.escape(token_text):
                allowed_ids.add(token_id)
        return allowed_ids

    def allowed_token_by_state(
        self,
        state: State,
        current_prefix: str,
        user_input: str,
        allowed_args: Optional[List[str]],
        expected_replacement: Optional[str]
    ) -> Set[int]:
        """Return allowed token IDs given current FSM state.

        Args:
            state: Current state in the FSM.
            current_prefix: Current partial field content for the state.
            user_input: The current user prompt text
            (used for "prompt" field).
            allowed_args: List of allowed argument
            keys for the selected function.

        Returns:
            Set of allowed token IDs for the next token.
        """
        if state == State.EXPECT_OPEN_BRACE:
            return {self.literal_id["{"]}

        if state == State.EXPECT_PROMPT_KEY:
            return self.allow_tokenids_for_string(current_prefix, '"prompt"')

        if state == State.EXPECT_PROMPT_COLON:
            return {self.literal_id[":"]}

        if state == State.EXPECT_PROMPT_OPEN_QUOTE:
            return {self.literal_id['"']}

        if state == State.EXPECT_PROMPT_VALUE:
            return self.allow_tokenids_for_string(current_prefix, user_input)

        if state == State.EXPECT_PROMPT_CLOSE_QUOTE:
            return {self.literal_id['"']}

        if state == State.EXPECT_PROMPT_COMMA:
            return {self.literal_id[","]}

        if state == State.EXPECT_FN_KEY:
            return self.allow_tokenids_for_string(current_prefix, '"fn_name"')

        if state == State.EXPECT_FN_COLON:
            return {self.literal_id[":"]}

        if state == State.EXPECT_FN_OPEN_QUOTE:
            return {self.literal_id['"']}

        if state == State.EXPECT_FN_VALUE:
            if "substitute" in user_input.lower():
                if self.count_quoted_chunks(user_input) < 2:
                    raise RuntimeError("Error: incomplete prompt")
            return self.allowed_fns(current_prefix, self.valid_names)

        if state == State.EXPECT_FN_CLOSE_QUOTE:
            return {self.literal_id['"']}

        if state == State.EXPECT_FN_COMMA:
            return {self.literal_id[","]}

        if state == State.EXPECT_ARGS_OPEN_QUOTE:
            return {self.literal_id['"']}

        if state == State.EXPECT_ARGS_KEY:
            return self.allow_tokenids_for_string(current_prefix, "args")

        if state == State.EXPECT_ARGS_CLOSE_QUOTE:
            return {self.literal_id['"']}

        if state == State.EXPECT_ARGS_COLON:
            return {self.literal_id[":"]}

        if state == State.EXPECT_ARGS_OPEN_BRACKET:
            return {self.literal_id["{"]}

        if state == State.EXPECT_ARGS_ARG_OPEN_QUOTE:
            return {self.literal_id['"']}

        if state == State.EXPECT_ARGS_ARG_KEY:
            if not allowed_args:
                return set()
            remaining = [a for a in allowed_args if a not in self.used_args]
            return self.allowed_args_f(current_prefix, remaining)

        if state == State.EXPECT_ARGS_ARG_CLOSE_QUOTE:
            return {self.literal_id['"']}

        if state == State.EXPECT_ARGS_ARG_COLON:
            return {self.literal_id[":"]}

        if state == State.EXPECT_ARGS_ARG_VALUE:
            if self.current_arg_type == "int":
                digit_ids = self.allowed_digit_tokens()
                norm = current_prefix.strip()

                if norm == "":
                    return digit_ids | self.minus_ids
                if norm == "-":
                    return digit_ids

                if norm.lstrip("-").isdigit():
                    allowed_int = set(digit_ids)
                    if len(self.used_args) < self.allowed_args_length:
                        allowed_int |= {self.literal_id[","]}
                    if len(self.used_args) == self.allowed_args_length:
                        allowed_int |= {self.literal_id["}"]}
                    return allowed_int

            if self.current_arg_type == "float":
                digit_ids = self.allowed_digit_tokens()
                norm = current_prefix.strip()

                if norm == "":
                    return digit_ids | self.minus_ids
                if norm == "-":
                    return digit_ids

                if "." not in norm:
                    if norm.lstrip("-").isdigit():
                        return digit_ids | {self.literal_id["."]}
                    return set()

                left, right = norm.split(".", 1)
                if left in ["", "-"]:
                    return digit_ids

                if right == "" or right.isdigit():
                    allowed_float = set(digit_ids)
                    if len(self.used_args) < self.allowed_args_length:
                        allowed_float |= {self.literal_id[","]}
                    if len(self.used_args) == self.allowed_args_length:
                        allowed_float |= {self.literal_id["}"]}
                    return allowed_float

            if self.current_arg_type == "str":
                if (
                    self.current_arg_key == "replacement"
                    and expected_replacement
                ):
                    target = expected_replacement

                    if current_prefix == "":
                        return {self.literal_id['"']}

                    if (
                        current_prefix.startswith('"')
                        and not current_prefix.endswith('"')
                    ):
                        inner = current_prefix[1:]

                        if target.startswith(inner):
                            return (
                                self.allow_tokenids_for_string(inner, target) |
                                {self.literal_id['"']})
                        return set()

                    if current_prefix == f'"{target}"':
                        allowed_str: Set[int] = set()
                        if len(self.used_args) < self.allowed_args_length:
                            allowed_str |= {self.literal_id[","]}
                        if len(self.used_args) == self.allowed_args_length:
                            allowed_str |= {self.literal_id["}"]}
                        return allowed_str
                if self.current_arg_key == "regex":
                    if current_prefix == "":
                        return {self.literal_id['"']}

                    if (
                        current_prefix.startswith('"')
                        and not current_prefix[1:].endswith('"')
                        and len(current_prefix) >= self.max_regex
                    ):
                        return (
                            self.allowed_regex().union({self.literal_id['"']}))

                    if (
                        current_prefix.startswith('"')
                        and current_prefix[1:].endswith('"')
                    ):
                        allowed_str = set()
                        if len(self.used_args) < self.allowed_args_length:
                            allowed_str |= {self.literal_id[","]}
                        if len(self.used_args) == self.allowed_args_length:
                            allowed_str |= {self.literal_id["}"]}
                        return allowed_str

                    return self.allowed_regex()

                if current_prefix == "":
                    return {self.literal_id['"']}

                if (
                    current_prefix.startswith('"')
                    and not current_prefix[1:].endswith('"')
                ):
                    return (
                        self.allowed_str_tokens().union({self.literal_id['"']})
                    )

                if (
                    current_prefix.startswith('"')
                    and current_prefix[1:].endswith('"')
                    and len(current_prefix) >= 2
                ):
                    allowed_str = set()
                    if len(self.used_args) < self.allowed_args_length:
                        allowed_str |= {self.literal_id[","]}
                    if len(self.used_args) == self.allowed_args_length:
                        allowed_str |= {self.literal_id["}"]}
                    return allowed_str

                return self.allowed_str_tokens()

            if self.current_arg_type == "bool":
                allowed_bool = self.allowed_bool_tokens(current_prefix)
                if current_prefix in ["true", "false"]:
                    if len(self.used_args) < self.allowed_args_length:
                        allowed_bool |= {self.literal_id[","]}
                    if len(self.used_args) == self.allowed_args_length:
                        allowed_bool |= {self.literal_id["}"]}
                return allowed_bool

            return set()

        if state == State.EXPECT_CLOSE_OUTER:
            return {self.literal_id["}"]}

        if state == State.DONE:
            return set()

        return set()

    def count_quoted_chunks(self, p: str) -> int:
        singles = re.findall(r"'([^'\\]*(?:\\.[^'\\]*)*)'", p)
        doubles = re.findall(r'"([^"\\]*(?:\\.[^"\\]*)*)"', p)
        return len(singles) + len(doubles)

    def advance_state_if_done(
        self,
        state: State,
        prefix: str,
        user_input: str,
        allowed_args: Optional[List[str]],
    ) -> Tuple[State, str]:
        """Advance FSM state when the current
        state's expected content is complete.

        Args:
            state: Current FSM state.
            prefix: Current accumulated prefix for the state.
            user_input: Original user prompt.
            allowed_args: Current allowed args for selected function.

        Returns:
            A tuple of (new_state, new_prefix).
        """
        if state == State.EXPECT_OPEN_BRACE and prefix == "{":
            return State.EXPECT_PROMPT_KEY, ""

        if state == State.EXPECT_PROMPT_KEY and prefix == '"prompt"':
            return State.EXPECT_PROMPT_COLON, ""

        if state == State.EXPECT_PROMPT_COLON and prefix == ":":
            return State.EXPECT_PROMPT_OPEN_QUOTE, ""

        if state == State.EXPECT_PROMPT_OPEN_QUOTE and prefix == '"':
            return State.EXPECT_PROMPT_VALUE, ""

        if state == State.EXPECT_PROMPT_VALUE and prefix == user_input:
            return State.EXPECT_PROMPT_CLOSE_QUOTE, ""

        if state == State.EXPECT_PROMPT_CLOSE_QUOTE and prefix == '"':
            return State.EXPECT_PROMPT_COMMA, ""

        if state == State.EXPECT_PROMPT_COMMA and prefix == ",":
            return State.EXPECT_FN_KEY, ""

        if state == State.EXPECT_FN_KEY and prefix == '"fn_name"':
            return State.EXPECT_FN_COLON, ""

        if state == State.EXPECT_FN_COLON and prefix == ":":
            return State.EXPECT_FN_OPEN_QUOTE, ""

        if state == State.EXPECT_FN_OPEN_QUOTE and prefix == '"':
            return State.EXPECT_FN_VALUE, ""

        if state == State.EXPECT_FN_VALUE and prefix in self.valid_names:
            self.fn_name_tmp = prefix
            return State.EXPECT_FN_CLOSE_QUOTE, ""

        if state == State.EXPECT_FN_CLOSE_QUOTE and prefix == '"':
            return State.EXPECT_FN_COMMA, ""

        if state == State.EXPECT_FN_COMMA and prefix == ",":
            return State.EXPECT_ARGS_OPEN_QUOTE, ""

        if state == State.EXPECT_ARGS_OPEN_QUOTE and prefix == '"':
            return State.EXPECT_ARGS_KEY, ""

        if state == State.EXPECT_ARGS_KEY and prefix == "args":
            return State.EXPECT_ARGS_CLOSE_QUOTE, ""

        if state == State.EXPECT_ARGS_CLOSE_QUOTE and prefix == '"':
            return State.EXPECT_ARGS_COLON, ""

        if state == State.EXPECT_ARGS_COLON and prefix == ":":
            return State.EXPECT_ARGS_OPEN_BRACKET, ""

        if state == State.EXPECT_ARGS_OPEN_BRACKET and prefix == "{":
            return State.EXPECT_ARGS_ARG_OPEN_QUOTE, ""

        if state == State.EXPECT_ARGS_ARG_OPEN_QUOTE and prefix == '"':
            return State.EXPECT_ARGS_ARG_KEY, ""

        if (
            state == State.EXPECT_ARGS_ARG_KEY and
            allowed_args and
            prefix in allowed_args
        ):
            self.current_arg_key = prefix
            if self.selected_fn is None:
                raise RuntimeError("selected_fn must be set" +
                                   "before selecting args.")
            self.current_arg_type = self.fn_arg_types[self.selected_fn][prefix]
            self.used_args.append(prefix)
            return State.EXPECT_ARGS_ARG_CLOSE_QUOTE, ""

        if state == State.EXPECT_ARGS_ARG_CLOSE_QUOTE and prefix == '"':
            return State.EXPECT_ARGS_ARG_COLON, ""

        if state == State.EXPECT_ARGS_ARG_COLON and prefix == ":":
            return State.EXPECT_ARGS_ARG_VALUE, ""

        if state == State.EXPECT_ARGS_ARG_VALUE:
            if self.current_arg_type == "int" and prefix.isdigit():
                return state, prefix

            if self.current_arg_type == "float":
                try:
                    float(prefix)
                    return state, prefix
                except Exception:
                    pass

            if self.current_arg_type == "str":
                if (
                    prefix.startswith('"') and
                    prefix.endswith('"') and
                    len(prefix) >= 2
                ):
                    return state, prefix

            if self.current_arg_type == "bool":
                if prefix in ["true", "false"]:
                    return state, prefix

            if prefix.endswith(","):
                return State.EXPECT_ARGS_ARG_OPEN_QUOTE, ""

            if prefix.endswith("}"):
                return State.EXPECT_CLOSE_OUTER, ""

            return state, prefix

        if state == State.EXPECT_CLOSE_OUTER and prefix == "}":
            return State.DONE, ""

        return state, prefix

    def argmax(self, logits: List[float]) -> Optional[Tuple[int, float]]:
        """Return index and value of the maximum logit.

        Args:
            logits: List of logit values.

        Returns:
            (best_index, best_value) or None if no valid maximum exists.
        """
        best = float("-inf")
        best_i: Optional[int] = None
        for i, logit in enumerate(logits):
            if logit > best:
                best = logit
                best_i = i
        if best_i is None or best == float("-inf"):
            return None
        return best_i, best

    def generate(self) -> None:
        """Generate constrained JSON outputs
        for all prompts and write results to disk."""
        try:
            self.init_attributes_globals()
            result: List[Dict[str, Any]] = []
            if self.prompts_dict is None:
                raise RuntimeError("Prompts file must be" +
                                   "loaded before generation.")

            prompts: List[str] = [prom["prompt"] for prom in self.prompts_dict]

            for p in prompts:
                expected_replacement = self.extract_replacement_from_prompt(p)
                if p.strip() == "":
                    raise Exception("Function not found")
                prompt = f"""
                Return JSON with:
                prompt, fn_name, args.

                Functions:
                fn_add_numbers(a: float, b: float)
                fn_get_square_root(a: int)
                fn_greet(name: str)
                fn_is_even(n: int)
                fn_multiply_numbers(a: float, b: float)
                fn_reverse_string(s: str)
                fn_substitute_string_with_regex(source_string: str, regex: str, replacement: str)

                User: {p}
                JSON:
                """ # noqa

                self.init_attributes()
                full_output = ""
                ids: List[int] = self.wrapper.encode(prompt)

                allowed_args: Optional[List[str]] = None

                while True:
                    if self.current_state == State.DONE:
                        break

                    if self.current_state in self.fixed_ids:
                        toks = self.fixed_ids[self.current_state]
                        text = self.fixed_text[self.current_state]

                        ids.extend(toks)
                        self.current_prefix += text
                        full_output += text

                        (self.current_state,
                         self.current_prefix) = self.advance_state_if_done(
                            self.current_state,
                            self.current_prefix,
                            p,
                            allowed_args,
                        )
                        continue

                    if self.current_state == State.EXPECT_PROMPT_VALUE:
                        toks = self.wrapper.encode(p)
                        ids.extend(toks)
                        self.current_prefix += p
                        full_output += p
                        (self.current_state,
                         self.current_prefix) = self.advance_state_if_done(
                            self.current_state,
                            self.current_prefix,
                            p,
                            allowed_args,
                        )
                        continue

                    logits = self.wrapper.get_logits_from_input_ids(ids)
                    allowed_ids = self.allowed_token_by_state(
                        self.current_state,
                        self.current_prefix,
                        p,
                        allowed_args,
                        expected_replacement
                    )

                    best_i = None
                    best_v = float("-inf")

                    for i in allowed_ids:
                        v = logits[i]
                        if v > best_v:
                            best_v = v
                            best_i = i

                    if best_i is None:
                        raise RuntimeError("No valid token available ...")

                    idx = best_i
                    ids.append(idx)

                    decoded_token = self.token_text[idx]
                    self.current_prefix += decoded_token
                    full_output += decoded_token

                    if self.fn_name_tmp:
                        self.selected_fn = self.fn_name_tmp
                        allowed_args = list(
                            self.fn_arg_types[self.selected_fn].keys()
                        )
                        self.allowed_args_length = len(allowed_args)
                        self.fn_name_tmp = ""

                    (self.current_state,
                     self.current_prefix) = self.advance_state_if_done(
                        self.current_state,
                        self.current_prefix,
                        p,
                        allowed_args,
                    )

                try:
                    parsed = json.loads(full_output)
                    validated = FunctionCall(**parsed)
                    result.append(validated.model_dump())
                except (json.JSONDecodeError, ValidationError) as e:
                    raise RuntimeError(f"Invalid generated output: {e}")

            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            with open(self.output_path, "w", encoding="utf-8") as file:
                json.dump(result, file, indent=4)
                file.write("\n")

        except (RuntimeError, Exception) as e:
            print("Something Wrong", e)


def main() -> None:
    """CLI entrypoint: parse args, load files, run generation."""
    try:
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument(
                "--input",
                default="data/input/function_calling_tests.json",
                help="Path to input file",
            )
            parser.add_argument(
                "--output",
                default="data/output/function_calling_results.json",
                help="Path to output file",
            )
            args = parser.parse_args()
            if not args.input or not args.output:
                raise Exception
        except Exception:
            print("Something wrong with the arguments")
            raise Exception

        jsonwrapper = Jsongenerator(args.input, args.output)
        jsonwrapper.load_files()
        jsonwrapper.generate()
    except (RuntimeError, Exception):
        sys.exit()

import re
from typing import Any, Literal, cast

from .models import FunctionDef


class DecoderError(Exception):
    pass


class TypeNotFound(DecoderError):
    pass


class FunctionNotFound(DecoderError):
    pass


class Decoder:
    MAX_ITERATIONS = 1000

    def __init__(
        self, llm: Any, ids: list[int], functions: dict[str, FunctionDef]
    ) -> None:
        if llm is None:
            raise ValueError("llm cannot be None")
        if ids is None:
            raise ValueError("ids cannot be None")
        if functions is None:
            raise ValueError("functions cannot be None")
        self.llm = llm
        self.ids = ids
        self.functions = functions

    def _force(self, text: str, escape: bool = False) -> None:
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text).__name__}")
        if escape:
            text = text.replace("\\", "\\\\").replace('"', '\\"')
        try:
            ids_to_add = self.llm.encode(text).tolist()[0]
        except (IndexError, AttributeError) as e:
            raise DecoderError(f"Failed to encode '{text}': {e}") from e
        for id in ids_to_add:
            self.ids.append(id)

    def _constrain(self, valid_ids: list[int] | Literal["all"]) -> int:
        try:
            logits = cast(
                list[float], self.llm.get_logits_from_input_ids(self.ids)
            )
        except Exception as e:
            raise DecoderError(f"Failed to get logits: {e}") from e
        if not logits:
            raise DecoderError("Received empty logits from model")
        if valid_ids != "all":
            logits = [
                l if i in valid_ids else float("-inf")
                for i, l in enumerate(logits)
            ]
        max_logit = max(logits)
        if max_logit == float("-inf"):
            raise DecoderError(
                "No valid token found: all constrained logits are -inf"
            )
        return logits.index(max_logit)

    def _get_function_name(self, functions: list[str]) -> str:
        if not functions:
            raise FunctionNotFound("No functions provided")
        try:
            functions_ids = {
                f: self.llm.encode(f).tolist()[0] for f in functions
            }
        except (IndexError, AttributeError) as e:
            raise DecoderError(
                f"Failed to encode function names: {e}"
            ) from e

        i = 0
        while i < self.MAX_ITERATIONS:
            valid_ids = list(
                {ids[i] for ids in functions_ids.values() if i < len(ids)}
            )
            if not valid_ids:
                raise DecoderError(
                    f"No valid token IDs at position {i}"
                    " for remaining functions"
                )
            chosen_id = self._constrain(valid_ids)
            self.ids.append(chosen_id)
            functions_ids = {
                f: ids for f, ids in functions_ids.items()
                if i < len(ids) and ids[i] == chosen_id
            }
            if not functions_ids:
                raise FunctionNotFound(
                    "No matching function found for "
                    "the generated token sequence"
                )
            key, value = next(iter(functions_ids.items()))
            if len(functions_ids) == 1 and i == len(value) - 1:
                return key
            i += 1

        raise DecoderError(
            f"Function name resolution exceeded "
            f"{self.MAX_ITERATIONS} iterations"
        )

    def _get_value(self, value_type: str, sep: str, prompt: str) -> None:
        if value_type in ("number", "float", "integer"):
            numbers = re.findall(r'-?\d+\.?\d*', prompt)
            if numbers:
                max_digit = max(numbers, key=len) + "00"
            else:
                max_digit = "000"
            try:
                number_ids = self.llm.encode("0123456789").tolist()[0]
                minus_id = self.llm.encode("-").tolist()[0]
                sep_id = self.llm.encode(sep).tolist()[0]
                dot_id = self.llm.encode(".").tolist()[0]
            except (IndexError, AttributeError) as e:
                raise DecoderError(
                    f"Failed to encode numeric tokens: {e}"
                ) from e
            next_id = self._constrain(number_ids + minus_id)
            self.ids.append(next_id)
            if value_type in ("number", "float"):
                valid_ids = number_ids + dot_id + sep_id
            else:
                valid_ids = number_ids + sep_id
            for _ in max_digit:
                next_id = self._constrain(valid_ids)
                if next_id in sep_id:
                    if dot_id[0] in valid_ids:
                        self._force(".0")
                    self.ids.append(next_id)
                    return
                if next_id in dot_id:
                    valid_ids = number_ids + sep_id
                self.ids.append(next_id)
            self._force(sep)
            return

        if value_type == "string":
            self._force('"')
            escaped = False
            for _ in range(self.MAX_ITERATIONS):
                next_id = self._constrain("all")
                try:
                    token_text: str = self.llm.decode([next_id])
                except Exception as e:
                    raise DecoderError(
                        f"Failed to decode token {next_id}: {e}"
                    ) from e
                close_idx = None
                for j, ch in enumerate(token_text):
                    if escaped:
                        escaped = False
                        continue
                    if ch == '\\':
                        escaped = True
                    elif ch == '"':
                        close_idx = j
                        break
                if close_idx is not None:
                    self._force(token_text[:close_idx + 1] + sep)
                    return
                self.ids.append(next_id)
            raise DecoderError(
                f"String decoding exceeded {self.MAX_ITERATIONS} iterations"
            )

        if value_type == "boolean":
            try:
                boolean_ids = self.llm.encode("true").tolist()[0]
                boolean_ids.extend(self.llm.encode("false").tolist()[0])
                sep_id = self.llm.encode(sep).tolist()[0]
            except (IndexError, AttributeError) as e:
                raise DecoderError(
                    f"Failed to encode boolean tokens: {e}"
                ) from e
            next_id = self._constrain(number_ids)
            self.ids.append(next_id)
            next_id = self._constrain(sep_id)
            self.ids.append(next_id)
            return

        raise TypeNotFound(f"Unsupported type: '{value_type}'")

    def decode(self, prompt: str, functions: list[str]) -> None:
        if not isinstance(prompt, str):
            raise ValueError(
                f"prompt must be a string, got {type(prompt).__name__}"
            )
        if not functions:
            raise FunctionNotFound("No functions provided to decode")
        missing = [f for f in functions if f not in self.functions]
        if missing:
            raise FunctionNotFound(f"Functions not registered: {missing}")

        self._force('{"prompt":"')
        self._force(prompt, True)
        self._force('","name":"')
        f = self._get_function_name(functions)

        if f not in self.functions:
            raise FunctionNotFound(
                f"Decoded function '{f}' is not registered"
            )

        self._force('","parameters":{')
        params = self.functions[f].parameters or {}
        for i, param in enumerate(params):
            self._force(f'"{param}":')
            sep = "}" if i == len(params) - 1 else ","
            param_type = getattr(params[param], "type", None)
            if param_type is None:
                raise DecoderError(
                    f"Parameter '{param}' of function '{f}'"
                    " has no type defined"
                )
            self._get_value(param_type, sep, prompt)
        if not params:
            self._force("}")
        self._force("}")

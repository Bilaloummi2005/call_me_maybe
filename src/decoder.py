from .models import FunctionCall


class DecoderError(Exception):
    pass


class TypeNotFound(DecoderError):
    pass


class FunctionNotFound(DecoderError):
    pass


class Decoder:
    MAX_ITERATIONS = 1000

    def __init__(self, llm, ids, functions):
        if llm is None:
            raise ValueError("llm cannot be None")
        if ids is None:
            raise ValueError("ids cannot be None")
        if functions is None:
            raise ValueError("functions cannot be None")
        self.llm = llm
        self.ids = ids
        self.functions = functions

    def _force(self, text):
        if not isinstance(text, str):
            raise ValueError(f"Expected string, got {type(text).__name__}")
        try:
            ids_to_add = self.llm.encode(text).tolist()[0]
        except (IndexError, AttributeError) as e:
            raise DecoderError(f"Failed to encode '{text}': {e}") from e
        for id in ids_to_add:
            self.ids.append(id)

    def _constrain(self, valid_ids):
        try:
            logits = self.llm.get_logits_from_input_ids(self.ids)
        except Exception as e:
            raise DecoderError(f"Failed to get logits: {e}") from e
        if not logits:
            raise DecoderError("Received empty logits from model")
        if valid_ids != "all":
            logits = [l if i in valid_ids else float("-inf") for i, l in enumerate(logits)]
        max_logit = max(logits)
        if max_logit == float("-inf"):
            raise DecoderError("No valid token found: all constrained logits are -inf")
        return logits.index(max_logit)

    def _get_function_name(self, functions):
        if len(functions) == 0:
            raise FunctionNotFound("No functions provided")
        try:
            functions_ids = {
                f: self.llm.encode(f).tolist()[0] for f in functions
            }
        except (IndexError, AttributeError) as e:
            raise DecoderError(f"Failed to encode function names: {e}") from e

        i = 0
        while i < self.MAX_ITERATIONS:
            valid_ids = list({ids[i] for ids in functions_ids.values() if i < len(ids)})
            if not valid_ids:
                raise DecoderError(f"No valid token IDs at position {i} for remaining functions")
            chosen_id = self._constrain(valid_ids)
            self.ids.append(chosen_id)
            functions_ids = {
                f: ids for f, ids in functions_ids.items()
                if i < len(ids) and ids[i] == chosen_id
            }
            if not functions_ids:
                raise FunctionNotFound("No matching function found for the generated token sequence")
            key, value = next(iter(functions_ids.items()))
            if len(functions_ids) == 1 and i == len(value) - 1:
                return key
            i += 1

        raise DecoderError(f"Function name resolution exceeded {self.MAX_ITERATIONS} iterations")

    def _get_value(self, type, sep):
        if type == "number" or type == "float":
            try:
                number_ids = self.llm.encode("0123456789").tolist()[0]
                minus_id = self.llm.encode("-").tolist()[0]
                sep_id = self.llm.encode(sep).tolist()[0]
                dot_id = self.llm.encode(".").tolist()[0]
            except (IndexError, AttributeError) as e:
                raise DecoderError(f"Failed to encode numeric tokens: {e}") from e
            next_id = self._constrain(number_ids + minus_id)
            self.ids.append(next_id)
            valid_ids = number_ids + dot_id + sep_id if type == "float" else number_ids + sep_id
            for _ in range(self.MAX_ITERATIONS):
                next_id = self._constrain(valid_ids)
                self.ids.append(next_id)
                if next_id in sep_id:
                    return
            raise DecoderError(f"Number decoding exceeded {self.MAX_ITERATIONS} iterations")

        if type == "string":
            self._force('"')
            for _ in range(self.MAX_ITERATIONS):
                next_id = self._constrain("all")
                try:
                    is_last: str = self.llm.decode([next_id])
                except Exception as e:
                    raise DecoderError(f"Failed to decode token {next_id}: {e}") from e
                self.ids.append(next_id)
                if '"' in is_last:
                    if not is_last.endswith(sep):
                        self._force(sep)
                    return
            raise DecoderError(f"String decoding exceeded {self.MAX_ITERATIONS} iterations")

        if type == "boolean":
            try:
                number_ids = self.llm.encode("01").tolist()[0]
                sep_id = self.llm.encode(sep).tolist()[0]
            except (IndexError, AttributeError) as e:
                raise DecoderError(f"Failed to encode boolean tokens: {e}") from e
            next_id = self._constrain(number_ids)
            self.ids.append(next_id)
            next_id = self._constrain(sep_id)
            return

        raise TypeNotFound(f"Unsupported type: '{type}'")

    def decode(self, prompt, functions):
        if not isinstance(prompt, str):
            raise ValueError(f"prompt must be a string, got {type(prompt).__name__}")
        if not functions:
            raise FunctionNotFound("No functions provided to decode")
        missing = [f for f in functions if f not in self.functions]
        if missing:
            raise FunctionNotFound(f"Functions not registered: {missing}")

        self._force('{"prompt":"')
        self._force(prompt)
        self._force('","name":"')
        f = self._get_function_name(functions)

        if f not in self.functions:
            raise FunctionNotFound(f"Decoded function '{f}' is not registered")

        self._force('","parameters":{')
        params = self.functions[f].parameters
        for i, param in enumerate(params):
            self._force(f'"{param}":')
            sep = "}" if i == len(params) - 1 else ","
            param_type = getattr(params[param], "type", None)
            if param_type is None:
                raise DecoderError(f"Parameter '{param}' of function '{f}' has no type defined")
            self._get_value(param_type, sep)
        if len(params) == 0:
            self._force("}")
        self._force("}")

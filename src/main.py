# mypy: allow-untyped-defs

from .models import FunctionCall, FunctionDef, TestPrompt
from typing import Any
from . import Small_LLM_Model
import sys
import json
from pathlib import Path
from .decoder import Decoder, DecoderError
import argparse
from pydantic import ValidationError


class JsonGenerater:
    def __init__(self) -> None:
        try:
            self.llm = Small_LLM_Model()
        except Exception as e:
            raise RuntimeError(f"Failed to load LLM model: {e}") from e
        self.fun_def_path, self.prompt_path, self.output_file = (
            self.parse_arguments()
        )
        self.funcs, self.funcs_description = self.get_funcs()
        self.functions_name = list(self.funcs.keys())
        self.prompts = self.get_prompts()

    def get_prompts(self) -> list[str]:
        try:
            with open(self.prompt_path) as file_obj:
                content: list[dict[str, Any]] = json.load(file_obj)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Prompts file not found: '{self.prompt_path}'"
            )
        except PermissionError:
            raise PermissionError(
                f"Permission denied reading: '{self.prompt_path}'"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in '{self.prompt_path}': {e}"
            ) from e

        prompts: list[str] = []
        for i, prompt in enumerate(content):
            try:
                p = TestPrompt.model_validate(prompt)
                prompts.append(p.prompt)
            except ValidationError as e:
                raise ValueError(f"Invalid prompt at index {i}: {e}") from e
        return prompts

    def parse_arguments(self) -> tuple[str, str, str]:
        parser = argparse.ArgumentParser(
            description="Function calling system using constrained decoding."
        )
        parser.add_argument(
            "--functions_definition",
            type=str,
            default="data/input/functions_definition.json",
            help="Path to function definitions JSON file."
        )
        parser.add_argument(
            "--input",
            type=str,
            default="data/input/function_calling_tests.json",
            help="Path to input prompts JSON file."
        )
        parser.add_argument(
            "--output",
            type=str,
            default="data/output/function_calling_results.json",
            help="Path to output JSON file."
        )
        args = parser.parse_args()
        return args.functions_definition, args.input, args.output

    def get_funcs(self) -> tuple[dict[str, FunctionDef], str]:
        try:
            with open(self.fun_def_path) as file_obj:
                content: list[dict[str, Any]] = json.load(file_obj)
        except FileNotFoundError:
            raise FileNotFoundError(
                f"Functions definition file not found: '{self.fun_def_path}'"
            )
        except PermissionError:
            raise PermissionError(
                f"Permission denied reading: '{self.fun_def_path}'"
            )
        except json.JSONDecodeError as e:
            raise ValueError(
                f"Invalid JSON in '{self.fun_def_path}': {e}"
            ) from e

        funcs: dict[str, FunctionDef] = {}
        funcs_description = ""
        for i, fun in enumerate(content):
            try:
                func_def = FunctionDef.model_validate(fun)
                funcs_description += func_def.fun_description()
                funcs[func_def.name] = func_def
            except ValidationError as e:
                raise ValueError(
                    f"Invalid function definition at index {i}: {e}"
                ) from e

        if not funcs:
            raise ValueError(
                f"No function definitions found in '{self.fun_def_path}'"
            )

        return funcs, funcs_description

    def generate(self) -> None:
        json_output: list[dict[str, Any]] = []
        errors: list[str] = []

        for i, p in enumerate(self.prompts):
            prompt = f"""
Return JSON with:
prompt, fn_name, args.

Functions:
{self.funcs_description}
User: {p}
JSON:
"""
            try:
                ids = self.llm.encode(prompt).tolist()[0]
            except (IndexError, AttributeError) as e:
                errors.append(f"Prompt {i}: failed to encode prompt: {e}")
                continue

            try:
                decode = Decoder(self.llm, ids, self.funcs)
                decode.decode(p, self.functions_name)
            except DecoderError as e:
                errors.append(
                    f"Prompt {i} ('{p[:40]}'): decoder error: {e}"
                )
                continue

            try:
                output = self.llm.decode(ids)
            except Exception as e:
                errors.append(f"Prompt {i}: failed to decode output: {e}")
                continue

            print(output[len(prompt):])

            raw = output[len(prompt):]
            try:
                parsed = json.loads(raw)
                if not isinstance(parsed, dict):
                    errors.append(
                        f"Prompt {i+1} ('{p[:40]}'): "
                        "invalid function call output: not a JSON object"
                    )
                    continue
                FunctionCall.model_validate(parsed)

                json_output.append(parsed)
            except ValidationError as e:
                errors.append(
                    f"Prompt {i} ('{p[:40]}'): "
                    f"invalid function call output: {e}"
                )
            except Exception as e:
                errors.append(
                    f"Prompt {i} ('{p[:40]}'): "
                    f"failed to parse output: {e}"
                )

        if errors:
            print(f"\n{len(errors)} prompt(s) failed:", file=sys.stderr)
            for err in errors:
                print(f"  - {err}", file=sys.stderr)

        try:
            Path(self.output_file).parent.mkdir(parents=True, exist_ok=True)
            with open(self.output_file, "w") as f:
                json.dump(json_output, f, indent=4)
        except PermissionError:
            raise PermissionError(
                f"Permission denied writing to: '{self.output_file}'"
            )
        except OSError as e:
            raise OSError(
                f"Failed to write output to '{self.output_file}': {e}"
            ) from e


def main() -> None:
    try:
        gen = JsonGenerater()
        gen.generate()
    except (FileNotFoundError, PermissionError, ValueError) as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)
    except RuntimeError as e:
        print(f"Initialization error: {e}", file=sys.stderr)
        sys.exit(1)
    except KeyboardInterrupt:
        print("\nInterrupted.", file=sys.stderr)
        sys.exit(130)

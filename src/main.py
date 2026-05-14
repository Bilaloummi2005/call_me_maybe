from .models import FunctionCall, FunctionDef, ParameterSchema, TestPrompt
from .vocab import Vocab
from . import Small_LLM_Model
import sys
import json
from .decoder import Decoder
import json_repair


class JsonGenerater:
    def __init__(self, fun_def_path, prompt_path):
        self.llm = Small_LLM_Model()
        self.funcs, self.funcs_description = self.get_funcs(fun_def_path)
        self.functions_name = [f for f in self.funcs.keys()]
        self.prompts = self.get_prompts(prompt_path)

    def get_prompts(self, prompt_path):
        prompts = []
        content = []
        with open(prompt_path) as f:
            content = json.load(f)
        for prompt in content:
            p = TestPrompt.model_validate(prompt)
            prompts.append(p.prompt)
        return prompts

    def get_funcs(self, fun_def_path):
        funcs = {}
        content = []
        funcs_description = ""
        with open(fun_def_path) as f:
            content = json.load(f)
        for fun in content:
            f = FunctionDef.model_validate(fun)
            funcs_description += f.fun_description()
            funcs[f.name] = f
        return funcs, funcs_description

    def generate(self):
        json_output = []
        for p in self.prompts:
            prompt = f"""
Return JSON with:
prompt, fn_name, args.

Functions:
{self.funcs_description}
User: {p}
JSON:
"""
            print(len(prompt))
            ids = self.llm.encode(prompt).tolist()[0]
            decode = Decoder(self.llm, ids, self.funcs)
            decode.decode(p, self.functions_name)
            output = self.llm.decode(ids)
            print(output[len(prompt):])
            FunctionCall.model_validate(json_repair.loads(output[len(prompt):]))
            json_output.append(json_repair.loads(output[len(prompt):]))
        print(json_output)
        with open("data/output/function_calling_results.json", "w") as f:
            json.dump(json_output, f, indent=4)


def main():
    # test = '{"prompt":"Replace all numbers in "Hello 34 I\'m 233 years old" with NUMBERS","name":"fn_substitute_string_with_regex","parameters":{"source_string":"Hello 34 I\'m 233 years old","regex":"\\d+","replacement":" NUMBERS "}}'
    
    # print(json.loads(test))
    # exit()

    gen = JsonGenerater(sys.argv[1], sys.argv[2])
    gen.generate()
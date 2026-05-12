from .models import FunctionCall, FunctionDef, ParameterSchema, TestPrompt
from .vocab import Vocab
from . import Small_LLM_Model
import sys
import json
from .decoder import Decoder


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
            prompt.append(p)
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
        for p in self.prompts:
            prompt = f"""
                Return JSON with:
                prompt, fn_name, args.

                Functions:
                {self.funcs_description}
                User: {p}
                JSON:
            """
            ids = self.llm.encode(prompt).tolist()[0]
            decode = Decoder(self.llm, ids, self.funcs)
            decode.decode(p, self.functions_name)
            print(output)



def main():
#     llm = Small_LLM_Model()
#     content = None
#     with open(sys.argv[1]) as f:
#         content = json.load(f)
    
    
#     funcs = {}
#     function_description = ""
#     for fun in content:
#         f = FunctionDef.model_validate(fun)
#         function_description += f.fun_description()
#         funcs[f.name] = f
#     # function_description += funcs["fn_substitute_string_with_regex"].fun_description()
#     p = "Replace all vowels in 'Programming is fun' with asterisks"
#     prompt = f"""
# Return JSON with:
# prompt, fn_name, args.

# Functions:
# {function_description}
# User: {p}
# JSON:
# """
#     ids = llm.encode(prompt).tolist()[0]
#     decode = Decoder(llm, ids, funcs)
#     functions_name = [f for f in funcs.keys()]
#     decode.decode(p, functions_name)
#     output = llm.decode(ids)
#     print(output)
    gen = JsonGenerater(sys.argv[1], sys.argv[2])
    gen.generate()
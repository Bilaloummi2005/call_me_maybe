from .models import FunctionCall, FunctionDef, ParameterSchema, TestPrompt
from .vocab import Vocab
from . import Small_LLM_Model
import sys
import json
from .decoder import Decoder


# class JsonGenerater:
#     def __init__(self):
        
        


def main():
    llm = Small_LLM_Model()
    content = None
    with open(sys.argv[1]) as f:
        content = json.load(f)
    
    
    funcs = {}
    function_description = ""
    for fun in content:
        f = FunctionDef.model_validate(fun)
        function_description += f.fun_description()
        funcs[f.name] = f
    # function_description += funcs["fn_substitute_string_with_regex"].fun_description()
    p = "Replace all vowels in 'Programming is fun' with asterisks"
    prompt = f"""
Return JSON with:
prompt, fn_name, args.

Functions:
{function_description}
User: {p}
JSON:
"""
    ids = llm.encode(prompt).tolist()[0]
    decode = Decoder(llm, ids, funcs)
    functions_name = [f for f in funcs.keys()]
    decode.decode(p, functions_name)
    output = llm.decode(ids)
    print(output)
    
    
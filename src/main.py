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
    print(content)
    funcs = {}
    for fun in content:
        f = FunctionDef.model_validate(fun)
        funcs[f.name] = f
    print(funcs)
    p = "Replace all vowels in 'Programming is fun' with asterisks"
    prompt = f"""
    Return JSON with:
    prompt, fn_name, args.

    Functions:
    {content}
    User: {p}
    JSON:
    """
    ids = llm.encode(prompt).tolist()[0]
    decode = Decoder(llm, ids, funcs)
    functions_name = [f for f in funcs.keys()]
    decode.decode(p, functions_name)
    output = llm.decode(ids)
    print(output)
    
    
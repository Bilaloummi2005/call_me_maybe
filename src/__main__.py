from llm_sdk.llm_sdk import Small_LLM_Model
import json

state = 0
json_guid = ['{', '"', 'prompt', '"', ":", '"', "x", '"',",", '"', "name", '"', ":"
                '"', 'x', '"', ",", 'parameters', '"', ":", "{", "x", "}",'}']

llm = Small_LLM_Model()
p = "What is the sum of 2 and 3?"
prompt = f"""
Return JSON with:
prompt, fn_name, args.

Functions:
  {{
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {{
      "a": {{
        "type": "number"
      }},
      "b": {{
        "type": "number"
      }}
    }},
    "returns": {{
      "type": "number"
    }}
  }},
  {{
    "name": "fn_greet",
    "description": "Generate a greeting message for a person by name.",
    "parameters": {{
      "name": {{
        "type": "string"
      }}
    }},
    "returns": {{
      "type": "string"
    }}
  }}
User: {p}
JSON:
""" 
ids = llm.encode(prompt)

ids_list = ids.tolist()[0]
promt_ids = llm.encode(p).tolist()[0]
print(ids_list)
# vocab_path = llm.get_path_to_vocab_file()
# with open(vocab_path, encoding="utf-8") as f:
#     json_vocab, _ = json.JSONDecoder().raw_decode(f.read())

# print(json_vocab)

def get_fn_name(ids_list, fn_name=["fn_add_numbers", "fn_greet"]):
    fn_name = {
        name: llm.encode(name).tolist()[0] for name in fn_name
    }
    print(fn_name)
    i = 0
    while i < 3:
        logits = llm.get_logits_from_input_ids(ids_list)
        allowed_ids = {
            value[i] for value in fn_name.values()
        }
        logits = [l if i in allowed_ids else float("-inf") for i, l in enumerate(logits)]
        max_logit = logits.index(max(logits))
        print(max_logit, fn_name)
        ids_list.append(max_logit)
        fn_name = {
            name: value for name, value in fn_name.items() if value[i] == max_logit
        }
        i += 1
    return 2

def get_parames(ids_list, n_parames):
    allowed_ids = set(llm.encode(p).tolist()[0])
    

while state < len(json_guid):
    allowed_ids = set(llm.encode(json_guid[state]).tolist()[0])
    print(allowed_ids)
    if state == 6:
        ids_list += promt_ids
    elif state == 13:
        n_parames = get_fn_name(ids_list)
    elif state ==20:
        get_parames(ids_list, n_parames)
    else:
        logits = llm.get_logits_from_input_ids(ids_list)
        logits = [l if i in allowed_ids else float("-inf") for i, l in enumerate(logits)]
        ids_list.append(logits.index(max(logits)))
    # print(state, json_guid[state])
    state += 1


print(llm.decode(ids_list))

    


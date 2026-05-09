from llm_sdk.llm_sdk import Small_LLM_Model
import json

state = 0
state_position = ['{', '"', "prompt", '"' ":", '"', "x", '"',",", '"', "name", '"', ":"
                '"', 'x', '"', '}']

llm = Small_LLM_Model()

ids = llm.encode("What is the sum of 2 and 3?")

ids_list = ids.tolist()[0]

print(ids_list)
vocab_path = llm.get_path_to_vocab_file()
# with open(vocab_path, encoding="utf-8") as f:
#     json_vocab, _ = json.JSONDecoder().raw_decode(f.read())

# print(json_vocab)


while state < len(state_position):
    logits = llm.get_logits_from_input_ids(ids_list)
    allowed_tokens = llm.encode(state_position[state]).tolist()[0]
    print(allowed_tokens)
    logits = [l if i in allowed_tokens else float("-inf") for i, l in enumerate(logits)]
    state += 1

    ids_list.append(logits.index(max(logits)))

print(llm.decode(ids_list))

    


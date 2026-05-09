from llm_sdk.llm_sdk import Small_LLM_Model


llm = Small_LLM_Model()

ids = llm.encode("{")

ids_list = ids.tolist()
print(ids_list)
while True:
    logits = llm.get_logits_from_input_ids(ids_list[0])

    max_logits = logits.index(max(logits))
    ids_list[0].append(max_logits)
    print(max_logits, llm.decode(ids_list))

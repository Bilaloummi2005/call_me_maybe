from llm_sdk.llm_sdk import Small_LLM_Model
# import torch
from time import sleep

llm = Small_LLM_Model()



input_ids = llm.encode("what is the capital of france")
print(input_ids)
ids = input_ids.tolist()
while True:
    logit = llm.get_logits_from_input_ids(ids[0])
    print(input_ids)
    print(ids)
    m_id = logit.index(max(logit))
    print(len(logit))
    ids[0].append(m_id)
    print(llm.decode(ids[0]))
    sleep(0.2)


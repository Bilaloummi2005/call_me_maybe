from llm_sdk.llm_sdk import Small_LLM_Model


llm = Small_LLM_Model()

print(llm.encode("hello world"))
print(llm.decode([14990,  1879]))

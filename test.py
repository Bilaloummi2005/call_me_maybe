from llm_sdk.llm_sdk import Small_LLM_Model

llm = Small_LLM_Model()

test = llm.encode("false").tolist()[0]

print(test)
test_to_modify = llm.decode(test)

test_to_modify = "\\\\\\\\".join(test_to_modify.split("\\"))



print(llm.decode(test))

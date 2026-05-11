from . import Small_LLM_Model
import json

class Vocab:
    def __init__(self, llm: Small_LLM_Model):
        vocab_path = llm.get_path_to_vocab_file()
        self.conetent = []
        with open(vocab_path, encoding="utf-8") as f:
            self.conetent: dict[str, int] = json.load(f)
        print("test")
        self.token_to_id = [token for token in self.conetent.keys()]
        self.ids_to_tokens = self.conetent
        print(self.token_to_id[90])
        print(self.ids_to_tokens["{"])


    

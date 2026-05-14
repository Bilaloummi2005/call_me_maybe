from .models import FunctionCall


class Decoder:
    def __init__(self, llm, ids, functions):
        self.llm = llm
        self.ids = ids
        self.functions = functions

    def _force(self, text):
        ids_to_add = self.llm.encode(text).tolist()[0]
        for id in ids_to_add:
            self.ids.append(id)

    def _constrain(self, valid_ids):
        logits = self.llm.get_logits_from_input_ids(self.ids)
        if valid_ids != "all":
            logits = [l if i in valid_ids else float("-inf") for i, l in enumerate(logits)]
        return logits.index(max(logits))

    def _get_function_name(self, functions):
        i = 0
        functions_ids = {
            f: self.llm.encode(f).tolist()[0] for f in functions
        }
        while True:
            valid_ids = list({tokens[i] for tokens in functions_ids.values()})
            chosen_id = self._constrain(valid_ids)
            self.ids.append(chosen_id)
            functions_ids = {
                f: tokens for f, tokens in functions_ids.items() if tokens[i] == chosen_id
            }
            key, value = next(iter(functions_ids.items()))
            if len(functions_ids) == 1 and i == len(value) - 1:
                return key
            i += 1

    def _get_value(self, type, sep, prompt):
        if type == "number" or type == "float":
            number_ids = self.llm.encode("0123456789").tolist()[0]
            minus_id = self.llm.encode("-").tolist()[0]
            sep_id = self.llm.encode(sep).tolist()[0]
            dot_id = self.llm.encode(".").tolist()[0]
            next_id = self._constrain(number_ids + minus_id)
            self.ids.append(next_id)
            if type == "float":
                valid_ids = number_ids + dot_id + sep_id
            else:
                valid_ids = number_ids + sep_id
            while True:
                next_id = self._constrain(valid_ids)
                self.ids.append(next_id)
                if next_id in sep_id:
                    return

        if type == "string":
            allowed_ids = "all"
            self._force('"')
            while True:
                next_id = self._constrain(allowed_ids)
                is_last: str = self.llm.decode([next_id])
                self.ids.append(next_id)
                if '"' in is_last:
                    if not is_last.endswith(sep):
                        self._force(sep)
                    return

        if type == "boolean":
            number_ids = self.llm.encode("01").tolist()[0]
            sep_id = self.llm.encode(sep).tolist()[0]
            next_id = self._constrain(number_ids)
            self.ids.append(next_id)
            next_id = self._constrain(sep_id)
            return
            


    def decode(self, prompt, functions):
        self._force('{"prompt":"')
        self._force(prompt)
        self._force('","name":"')
        f = self._get_function_name(functions)
        self._force('","parameters":{')
        params = self.functions[f].parameters
        for i, param in enumerate(params):
            self._force(f'"{param}":')
            sep = "}" if i == len(params) - 1 else ","
            self._get_value(params[param].type, sep, prompt)
        if len(params) == 0:
            self._force('}')
        self._force('}')

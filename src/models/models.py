from pydantic import BaseModel
from typing import Any, Optional

class ParameterSchema(BaseModel):
    type: str

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: Optional[dict[str, ParameterSchema]]
    returns: ParameterSchema

    def param_names(self) -> list[str]:
        return list(self.parameters.keys())

    def param_type(self, param_name: str) -> str:
        return self.parameters[param_name].type

    def fun_description(self):
        output = self.name + "("
        for key, value in self.parameters.items():
            output += key + f": {value.type}, "
        if len(self.parameters.items()) == 0:
            output += "gg"
        return output[:-2] + ')\n'

class TestPrompt(BaseModel):
    prompt: str

class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]

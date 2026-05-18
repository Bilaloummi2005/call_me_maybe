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
        if self.parameters is None:
            return []
        return list(self.parameters.keys())

    def param_type(self, param_name: str) -> str | None:
        if self.parameters is None:
            return None
        param = self.parameters.get(param_name)
        if param is None:
            return None
        return param.type

    def fun_description(self) -> str:
        params = self.parameters or {}
        output = self.name + "("
        for key, value in params.items():
            output += key + f": {value.type}, "
        if len(params.items()) == 0:
            return output + ")"
        return output[:-2] + ')\n'

class TestPrompt(BaseModel):
    prompt: str

class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]

from pydantic import BaseModel
from typing import Any

class ParameterSchema(BaseModel):
    type: str

class FunctionDef(BaseModel):
    name: str
    description: str
    parameters: dict[str, ParameterSchema]
    returns: ParameterSchema

    def param_names(self) -> list[str]:
        return list(self.parameters.keys())

    def param_type(self, param_name: str) -> str:
        return self.parameters[param_name].type

class TestPrompt(BaseModel):
    prompt: str

class FunctionCall(BaseModel):
    prompt: str
    name: str
    parameters: dict[str, Any]

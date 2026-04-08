from datetime import date

from pydantic import BaseModel, ConfigDict


class ProposalCreate(BaseModel):
    id: int
    name: str
    category: str
    description: str
    date: date
    owner: str


class ProposalResponse(BaseModel):
    model_config = ConfigDict(from_attributes=True)

    id: int
    name: str
    category: str
    description: str
    date: date
    owner: str

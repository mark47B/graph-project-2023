from typing import Optional
from pydantic import BaseModel


class Node(BaseModel):
    number: int
    node_degree: Optional[int]


class Edge(BaseModel):
    start_node: Node
    end_node: Node
    weight: int

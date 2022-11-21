from dataclasses import dataclass
from revolve2.core.database import SerializableFrozenStruct


@dataclass
class EnvironmentName(SerializableFrozenStruct, table_name="environment_name"):
    body_num: int
    ruggedness_num: int
    bowlness_num: int

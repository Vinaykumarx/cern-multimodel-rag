# core/config.py

from dataclasses import dataclass
import os


@dataclass
class LanceDBConfig:
    uri: str = "lancedb"
    table_name: str = "cern_demo"


@dataclass
class RAGConfig:
    embed_model: str = "sentence-transformers/all-MiniLM-L6-v2"
    enable_blip: bool = True
    device: str = "cpu"


def load_lancedb_config() -> LanceDBConfig:
    uri = os.getenv("LANCEDB_URI", "lancedb")
    table = os.getenv("LANCEDB_TABLE", "cern_demo")
    return LanceDBConfig(uri=uri, table_name=table)

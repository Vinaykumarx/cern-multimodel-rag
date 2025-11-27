# scripts/debug_env.py

import sys
import os
import lancedb
import pyarrow as pa

def main():
    print("=== Python ===")
    print(sys.version)
    print("Executable:", sys.executable)
    print()

    print("=== Env ===")
    print("VIRTUAL_ENV:", os.environ.get("VIRTUAL_ENV"))
    print("LANCEDB_URI:", os.getenv("LANCEDB_URI", "lancedb"))
    print("LANCEDB_TABLE:", os.getenv("LANCEDB_TABLE", "cern_demo"))
    print()

    print("=== LanceDB ===")
    db = lancedb.connect(os.getenv("LANCEDB_URI", "lancedb"))
    print("Tables:", db.table_names())
    if "cern_demo" in db.table_names():
        table = db.open_table("cern_demo")
        print("Schema:")
        print(table.schema)

if __name__ == "__main__":
    main()

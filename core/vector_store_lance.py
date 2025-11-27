# core/vector_store_lance.py

import pyarrow as pa
import lancedb


class LanceVectorStore:
    """
    Wrapper around LanceDB for stable vector storage + retrieval.
    Ensures vector column is a fixed-size list (required for ANN search).
    """

    def __init__(self, db_uri="lancedb", table_name="cern_demo", dim=384):
        self.db_uri = db_uri
        self.table_name = table_name
        self.dim = dim

        print(f"[LanceDB] Connecting to: {db_uri}")
        self.db = lancedb.connect(db_uri)

        self.table = self._ensure_table()

    # ------------------------------------------------------------------
    # TABLE SCHEMA SETUP
    # ------------------------------------------------------------------

    def _schema(self):
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            # ❗ FIXED: use list_(float32, dim) since fixed_size_list() doesn't exist
            pa.field("vector", pa.list_(pa.float32(), self.dim)),
        ])

    def _ensure_table(self):
        """
        Ensures a table exists with correct vector schema.
        If schema mismatches, drop + recreate.
        """

        existing_tables = self.db.table_names()

        # -------------------------------------------------------------
        # Table does not exist → create
        # -------------------------------------------------------------
        if self.table_name not in existing_tables:
            print(f"[LanceDB] Creating new table: {self.table_name}")
            return self.db.create_table(self.table_name, schema=self._schema())

        # -------------------------------------------------------------
        # Table exists → validate schema
        # -------------------------------------------------------------
        table = self.db.open_table(self.table_name)
        schema = table.schema

        try:
            vec_field = schema.field("vector")
            typ = vec_field.type

            # typ should be FixedSizeListType(size=dim)
            if not pa.types.is_fixed_size_list(typ):
                print("[LanceDB] Incorrect vector dtype → Recreating table.")
                self.db.drop_table(self.table_name)
                return self.db.create_table(self.table_name, schema=self._schema())

            if typ.list_size != self.dim:
                print("[LanceDB] Wrong vector dimension → Recreating table.")
                self.db.drop_table(self.table_name)
                return self.db.create_table(self.table_name, schema=self._schema())

        except Exception:
            print("[LanceDB] Failed schema check → Recreating table.")
            self.db.drop_table(self.table_name)
            return self.db.create_table(self.table_name, schema=self._schema())

        print(f"[LanceDB] Using existing table: {self.table_name}")
        return table

    # ------------------------------------------------------------------
    # INSERT
    # ------------------------------------------------------------------

    def add(self, rows):
        """Insert rows into LanceDB"""
        self.table.add(rows)

    # ------------------------------------------------------------------
    # SEARCH
    # ------------------------------------------------------------------

    def search(self, vector, top_k=5):
        """Perform ANN search."""
        return (
            self.table.search(vector, vector_column_name="vector")
            .limit(top_k)
            .to_list()
        )

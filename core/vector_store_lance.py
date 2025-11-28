# core/vector_store_lance.py

import pyarrow as pa
import lancedb


class LanceVectorStore:
    def __init__(self, db_uri="lancedb", table_name="cern_demo", dim=384):
        self.db = lancedb.connect(db_uri)
        self.table_name = table_name
        self.dim = dim
        self.table = self._ensure()

    def _schema(self):
        return pa.schema([
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("source", pa.string()),
            pa.field("page", pa.int32()),
            pa.field("chunk_index", pa.int32()),
            pa.field("vector", pa.list_(pa.float32(), self.dim)),
            pa.field("figure_captions", pa.list_(pa.string())),
            pa.field("figure_paths", pa.list_(pa.string())),
            pa.field("has_figures", pa.bool_()),
        ])

    def _ensure(self):
        names = self.db.table_names()

        if self.table_name not in names:
            return self.db.create_table(self.table_name, schema=self._schema())

        table = self.db.open_table(self.table_name)
        if "figure_captions" not in table.schema.names:
            self.db.drop_table(self.table_name)
            return self.db.create_table(self.table_name, schema=self._schema())

        return table

    def add(self, rows):
        self.table.add(rows)

    def search(self, vector, top_k=5):
        return (
            self.table.search(vector, vector_column_name="vector")
            .limit(top_k)
            .to_list()
        )

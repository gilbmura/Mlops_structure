"""A worker-local write buffer; API stores continue to commit each request immediately."""

from src.storage.uploads import UploadStore


class BufferedUploadStore(UploadStore):
    def __init__(self, database):
        super().__init__(database)
        self.pending: list[dict] = []

    def add(self, record: dict) -> None:
        self.pending.append(record)

    def flush(self) -> None:
        self.add_many(self.pending)
        self.pending.clear()

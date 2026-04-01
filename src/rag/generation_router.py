from __future__ import annotations

from langchain_core.documents import Document

class GenerationRouter:

    def route_query(self, query: str) -> str:
        pass

    def rewrite_query(self, query: str, route:str) -> str:
        pass

    def build_answer(self, query: str, route: str, parents: list[Document]) -> str:
        pass
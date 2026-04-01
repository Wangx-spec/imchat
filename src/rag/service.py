from rag.types import RetrievalResult, AnswerResult

class RAGService:
    def initialize(self, force_rebuild: bool = False) -> None:
        # config -> load docs -> chunk -> load/build index -> init retriever
        pass

    def retrieve(self, query: str) -> RetrievalResult:
        pass

    def route_query(self, query: str) -> str:
        pass

    def answer(self, query: str) -> AnswerResult:
        pass
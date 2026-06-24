"""RAG ingestion package.

Import concrete ingestion modules directly to avoid loading optional dependencies
such as marker-pdf or langchain during lightweight PDF parsing.
"""

__all__: list[str] = []

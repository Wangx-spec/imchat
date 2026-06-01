from actions.web_search_tools import web_search as _web_search_tool


def web_search(query: str) -> str:
    return _web_search_tool.invoke({"query": query or ""})

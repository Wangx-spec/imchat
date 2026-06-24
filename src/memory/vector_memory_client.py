from __future__ import annotations

import json
import logging
from datetime import datetime
from typing import Any

from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from db.messages import list_recent_user_messages
from llms.openai_chat import build_openai_chat_model

logger = logging.getLogger("memory.vector")


class VectorMemoryClient:
    """Qdrant-backed long-term memory with the same recall/remember shape as mem0."""

    def __init__(self, settings: Any) -> None:
        self.settings = settings
        self.top_k = max(1, int(getattr(settings, "vector_memory_top_k", 5) or 5))
        self.profile_update_every = max(
            1, int(getattr(settings, "vector_memory_profile_update_every", 3) or 3)
        )
        self.episodic_collection = (
            getattr(settings, "vector_memory_episodic_collection", "memory_episodic")
            or "memory_episodic"
        )
        self.profile_collection = (
            getattr(settings, "vector_memory_profile_collection", "memory_profile")
            or "memory_profile"
        )
        qdrant_url = (getattr(settings, "qdrant_url", None) or "").strip()
        if not qdrant_url:
            raise ValueError("QDRANT_URL is required for vector memory")

        self.client = QdrantClient(
            url=qdrant_url,
            api_key=getattr(settings, "qdrant_api_key", None),
        )
        self.embeddings = OpenAIEmbeddings(
            api_key=getattr(settings, "rag_embedding_api_key", None),
            model=getattr(settings, "rag_embedding_model", "text-embedding-v4"),
            base_url=getattr(
                settings,
                "rag_embedding_base_url",
                "https://dashscope.aliyuncs.com/compatible-mode/v1",
            ),
            dimensions=int(getattr(settings, "rag_embedding_dimensions", 1024) or 1024),
            check_embedding_ctx_length=False,
            chunk_size=10,
        )
        self._ensure_collection(self.episodic_collection)
        self._ensure_collection(self.profile_collection)
        self.episodic_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.episodic_collection,
            embedding=self.embeddings,
        )
        self.profile_store = QdrantVectorStore(
            client=self.client,
            collection_name=self.profile_collection,
            embedding=self.embeddings,
        )
        self._remember_counts: dict[str, int] = {}
        self._profile_llm = None

    def recall(self, user_id: str, query: str, limit: int = 5) -> list[str]:
        uid = (user_id or "").strip()
        q = (query or "").strip()
        if not uid or not q:
            return []
        top_k = max(1, min(int(limit or self.top_k), 10))
        flt = _user_filter(uid)
        out: list[str] = []
        try:
            profile_docs = self.profile_store.similarity_search(q, k=1, filter=flt)
            for doc in profile_docs:
                text = (doc.page_content or "").strip()
                if text:
                    out.append(f"用户画像：{text}")
        except Exception as exc:
            logger.warning("[VECTOR_MEMORY] profile recall failed user_id=%s error=%s", uid, exc)
        try:
            history_docs = self.episodic_store.similarity_search(q, k=top_k, filter=flt)
            for doc in history_docs:
                text = (doc.page_content or "").strip()
                if text:
                    out.append(f"相关历史：{text}")
        except Exception as exc:
            logger.warning("[VECTOR_MEMORY] episodic recall failed user_id=%s error=%s", uid, exc)
        return out[:top_k]

    def remember(
        self,
        user_id: str,
        messages: list[dict],
        session_id: str | None = None,
    ) -> None:
        uid = (user_id or "").strip()
        if not uid or not messages:
            return
        user_text, ai_text = _extract_turn(messages)
        if not user_text or not ai_text:
            return
        sid = (session_id or "").strip()
        now = datetime.utcnow().isoformat()
        text = f"用户：{user_text}\n助手：{ai_text}"
        metadata = {
            "user_id": uid,
            "session_id": sid,
            "type": "episodic",
            "created_at": now,
        }
        try:
            self.episodic_store.add_documents([Document(page_content=text, metadata=metadata)])
            logger.info("[VECTOR_MEMORY] episodic stored user_id=%s session_id=%s", uid, sid)
        except Exception as exc:
            logger.warning("[VECTOR_MEMORY] episodic store failed user_id=%s error=%s", uid, exc)
            return

        self._remember_counts[uid] = self._remember_counts.get(uid, 0) + 1
        if self._remember_counts[uid] % self.profile_update_every == 0:
            self._update_profile(uid)

    def _ensure_collection(self, name: str) -> None:
        try:
            self.client.get_collection(name)
            return
        except Exception:
            pass
        self.client.create_collection(
            collection_name=name,
            vectors_config=qmodels.VectorParams(
                size=int(getattr(self.settings, "rag_embedding_dimensions", 1024) or 1024),
                distance=qmodels.Distance.COSINE,
            ),
        )

    def _update_profile(self, user_id: str) -> None:
        rows = list_recent_user_messages(user_id=user_id, limit=20)
        if not rows:
            return
        transcript = "\n".join(
            f"{row.get('role', '')}: {row.get('content', '')}" for row in rows[-20:]
        )
        prompt = (
            "请从以下医学问答对话中提炼用户长期画像，只记录用户明确陈述的信息，"
            "不要推测疾病、年龄或身份。返回 JSON，字段为 preferences、medical_context、concerns、style。\n\n"
            f"{transcript}"
        )
        try:
            if self._profile_llm is None:
                self._profile_llm = build_openai_chat_model(self.settings)
            resp = self._profile_llm.invoke(
                [
                    SystemMessage(content="你负责提炼长期用户画像，必须谨慎处理医疗隐私。"),
                    HumanMessage(content=prompt),
                ]
            )
            raw = str(getattr(resp, "content", "") or "").strip()
            profile = _extract_json_object(raw) or {"summary": raw[:1000]}
            profile_text = json.dumps(profile, ensure_ascii=False)
        except Exception as exc:
            logger.warning("[VECTOR_MEMORY] profile generation failed user_id=%s error=%s", user_id, exc)
            return

        try:
            self.client.delete(
                collection_name=self.profile_collection,
                points_selector=qmodels.FilterSelector(filter=_user_filter(user_id)),
            )
        except Exception:
            pass
        try:
            self.profile_store.add_documents(
                [
                    Document(
                        page_content=profile_text,
                        metadata={
                            "user_id": user_id,
                            "type": "profile",
                            "updated_at": datetime.utcnow().isoformat(),
                        },
                    )
                ]
            )
            logger.info("[VECTOR_MEMORY] profile updated user_id=%s", user_id)
        except Exception as exc:
            logger.warning("[VECTOR_MEMORY] profile store failed user_id=%s error=%s", user_id, exc)


def _extract_turn(messages: list[dict]) -> tuple[str, str]:
    user_text = ""
    ai_text = ""
    for item in messages:
        if not isinstance(item, dict):
            continue
        role = str(item.get("role", "")).strip()
        content = str(item.get("content", "")).strip()
        if role == "user" and content:
            user_text = content
        elif role == "assistant" and content:
            ai_text = content
    return user_text, ai_text


def _user_filter(user_id: str) -> qmodels.Filter:
    return qmodels.Filter(
        must=[
            qmodels.FieldCondition(
                key="metadata.user_id",
                match=qmodels.MatchValue(value=user_id),
            )
        ]
    )


def _extract_json_object(text: str) -> dict[str, Any] | None:
    start = text.find("{")
    end = text.rfind("}") + 1
    if start < 0 or end <= start:
        return None
    try:
        parsed = json.loads(text[start:end])
    except Exception:
        return None
    return parsed if isinstance(parsed, dict) else None

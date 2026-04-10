CREATE TABLE IF NOT EXISTS conversations (
    session_id   TEXT PRIMARY KEY,            -- 即 LangGraph thread_id
    user_id      TEXT NOT NULL DEFAULT 'default-user',  -- 预留多用户，当前固定值
    title        TEXT,                         -- 对话标题（取首条消息摘要）
    created_at   TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at   TIMESTAMPTZ NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_conv_user_id ON conversations(user_id);
CREATE INDEX IF NOT EXISTS idx_conv_updated ON conversations(updated_at DESC);

CREATE TABLE IF NOT EXISTS rag_index_meta (
    id              SERIAL PRIMARY KEY,
    source_dirs     JSONB NOT NULL,
    embedding_model TEXT NOT NULL,
    children_count  INT NOT NULL,
    built_at        TIMESTAMPTZ NOT NULL DEFAULT now()
);
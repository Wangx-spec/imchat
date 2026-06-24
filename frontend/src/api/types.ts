export type Role = "user" | "assistant" | "system";

export type Conversation = {
  session_id: string;
  title?: string | null;
  updated_at?: string | null;
};

export type BackendMessage = {
  role: Role;
  content: string;
  created_at?: string | null;
};

export type HealthStatus = {
  ok: boolean;
  rag_ready: boolean;
  rag_reason: string;
  runtime: string;
};

export type SseEventName = "chunk" | "done" | "error" | "hitl" | string;

export type SsePayload = {
  text?: string;
  replace?: boolean;
  answer?: string;
  detail?: string;
  interrupt?: HitlInterrupt;
};

export type HitlInterrupt = {
  type?: string;
  reason?: string;
  draft_answer?: string;
  [key: string]: unknown;
};

export type StreamHandlers = {
  onChunk?: (text: string, replace: boolean) => void;
  onDone?: (answer: string) => void;
  onError?: (detail: string) => void;
  onHitl?: (interrupt: HitlInterrupt) => void;
};

export type PreviewImage = {
  id: string;
  file: File;
  objectUrl: string;
};

export type ChatMessage = {
  id: string;
  role: Role;
  content: string;
  images?: PreviewImage[];
  status?: "streaming" | "done" | "error" | "hitl_waiting";
  meta?: {
    event?: string;
    interrupt?: HitlInterrupt;
  };
};

export type KnowledgeStats = {
  enabled?: boolean;
  ready?: boolean;
  reason?: string;
  source_dirs?: string[];
  index_dir?: string;
  vector_db_provider?: string;
  parent_count?: number;
  child_count?: number;
  retriever_ready?: boolean;
  index_loaded?: boolean;
  index_rebuilt?: boolean;
  index_meta_match?: boolean | null;
  index_meta_reason?: string | null;
  rerank_enabled?: boolean;
  parallel_recall?: boolean;
  cache_enabled?: boolean;
};

export type KnowledgeSourceFile = {
  path: string;
  name: string;
  size: number;
  modified_at?: number | null;
  exists: boolean;
  source_dir: string;
};

export type KnowledgeParent = {
  title: string;
  source: string;
  content: string;
  metadata: Record<string, unknown>;
};

export type KnowledgeSearchResponse = {
  ok: boolean;
  query: string;
  answer: string;
  sources: string[];
  parents: KnowledgeParent[];
  debug: Record<string, unknown>;
  insufficient_info: boolean;
  elapsed_ms: number;
};

export type KnowledgeUploadResponse = {
  ok: boolean;
  saved_files: string[];
  rebuild?: KnowledgeRebuildResponse | null;
};

export type KnowledgeRebuildResponse = {
  ok: boolean;
  parent_count: number;
  child_count: number;
  index_rebuilt: boolean;
  elapsed_ms: number;
};

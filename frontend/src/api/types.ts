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

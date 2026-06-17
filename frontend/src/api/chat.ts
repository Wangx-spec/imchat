import { consumeSse } from "./sse";
import type { BackendMessage, Conversation, StreamHandlers } from "./types";

export async function createConversation(): Promise<string> {
  const res = await fetch("/api/conversations", { method: "POST" });
  if (!res.ok) throw new Error("创建会话失败");
  const data = (await res.json()) as { session_id?: string };
  if (!data.session_id) throw new Error("后端未返回会话 ID");
  return data.session_id;
}

export async function listConversations(): Promise<Conversation[]> {
  const res = await fetch("/api/conversations");
  if (!res.ok) return [];
  return (await res.json()) as Conversation[];
}

export async function listMessages(sessionId: string): Promise<BackendMessage[]> {
  const res = await fetch(`/api/conversations/${encodeURIComponent(sessionId)}/messages`);
  if (!res.ok) return [];
  return (await res.json()) as BackendMessage[];
}

export async function streamChat(
  sessionId: string,
  message: string,
  handlers: StreamHandlers,
): Promise<void> {
  const res = await fetch("/api/chat/stream", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, message }),
  });
  await ensureStreamResponse(res);
  await consumeSse(res, handlers);
}

export async function streamMultimodalChat(
  sessionId: string,
  message: string,
  images: File[],
  handlers: StreamHandlers,
): Promise<void> {
  const form = new FormData();
  form.append("session_id", sessionId);
  form.append("message", message);
  for (const image of images) {
    form.append("images", image, image.name);
  }

  const res = await fetch("/api/chat/multimodal/stream", {
    method: "POST",
    body: form,
  });
  await ensureStreamResponse(res);
  await consumeSse(res, handlers);
}

export async function resumeHitl(
  sessionId: string,
  decision: "approve" | "reject" | "revise",
  note: string,
  handlers: StreamHandlers,
): Promise<void> {
  const res = await fetch("/api/chat/hitl/resume", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ session_id: sessionId, decision, note }),
  });
  await ensureStreamResponse(res);
  await consumeSse(res, handlers);
}

async function ensureStreamResponse(res: Response): Promise<void> {
  if (res.ok) return;
  let detail = "请求失败";
  try {
    const data = (await res.json()) as { detail?: string };
    detail = data.detail || detail;
  } catch {
    detail = `${detail}: ${res.status}`;
  }
  throw new Error(detail);
}

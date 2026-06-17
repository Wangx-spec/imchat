import type { SsePayload, StreamHandlers } from "./types";

function parseFrames(buffer: string): { frames: string[]; rest: string } {
  const frames: string[] = [];
  let rest = buffer;
  while (true) {
    const idx = rest.indexOf("\n\n");
    if (idx === -1) break;
    frames.push(rest.slice(0, idx));
    rest = rest.slice(idx + 2);
  }
  return { frames, rest };
}

function parseFrame(frame: string): { event: string; payload: SsePayload | null } {
  const lines = frame.split("\n");
  let event = "message";
  const dataLines: string[] = [];

  for (const line of lines) {
    if (line.startsWith("event:")) {
      event = line.slice(6).trim();
    } else if (line.startsWith("data:")) {
      dataLines.push(line.slice(5).trim());
    }
  }

  if (!dataLines.length) return { event, payload: null };
  try {
    return { event, payload: JSON.parse(dataLines.join("\n")) as SsePayload };
  } catch {
    return { event, payload: null };
  }
}

export async function consumeSse(response: Response, handlers: StreamHandlers): Promise<void> {
  if (!response.body) {
    handlers.onError?.("流式响应不可用");
    return;
  }

  const reader = response.body.getReader();
  const decoder = new TextDecoder();
  let pending = "";

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;

    pending += decoder.decode(value, { stream: true });
    const parsed = parseFrames(pending);
    pending = parsed.rest;

    for (const frame of parsed.frames) {
      const { event, payload } = parseFrame(frame);
      if (!payload) continue;

      if (event === "chunk") {
        handlers.onChunk?.(payload.text || "", Boolean(payload.replace));
      } else if (event === "done") {
        handlers.onDone?.(payload.answer || "");
      } else if (event === "error") {
        handlers.onError?.(payload.detail || "流式响应失败");
      } else if (event === "hitl" && payload.interrupt) {
        handlers.onHitl?.(payload.interrupt);
      }
    }
  }
}

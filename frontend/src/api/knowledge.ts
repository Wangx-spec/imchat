import type {
  KnowledgeRebuildResponse,
  KnowledgeSearchResponse,
  KnowledgeSourceFile,
  KnowledgeStats,
  KnowledgeUploadResponse,
} from "./types";

export async function getKnowledgeStats(): Promise<KnowledgeStats> {
  const res = await fetch("/api/knowledge/stats");
  if (!res.ok) throw new Error(await readError(res, "获取知识库状态失败"));
  return (await res.json()) as KnowledgeStats;
}

export async function listKnowledgeSources(): Promise<KnowledgeSourceFile[]> {
  const res = await fetch("/api/knowledge/sources");
  if (!res.ok) throw new Error(await readError(res, "获取知识库源文件失败"));
  const data = (await res.json()) as { items?: KnowledgeSourceFile[] };
  return data.items || [];
}

export async function searchKnowledge(
  query: string,
  topK: number,
  includeAnswer: boolean,
): Promise<KnowledgeSearchResponse> {
  const res = await fetch("/api/knowledge/search", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ query, top_k: topK, include_answer: includeAnswer }),
  });
  if (!res.ok) throw new Error(await readError(res, "知识库召回调试失败"));
  return (await res.json()) as KnowledgeSearchResponse;
}

export async function uploadKnowledgeFile(
  file: File,
  rebuild: boolean,
  pdfParser: "light" | "marker",
  topic: string,
): Promise<KnowledgeUploadResponse> {
  const form = new FormData();
  form.append("file", file, file.name);
  const params = new URLSearchParams({
    rebuild: rebuild ? "true" : "false",
    pdf_parser: pdfParser,
    topic,
  });
  const res = await fetch(`/api/knowledge/upload?${params.toString()}`, {
    method: "POST",
    body: form,
  });
  if (!res.ok) throw new Error(await readError(res, "上传知识库文件失败"));
  return (await res.json()) as KnowledgeUploadResponse;
}

export async function rebuildKnowledge(): Promise<KnowledgeRebuildResponse> {
  const res = await fetch("/api/knowledge/rebuild", { method: "POST" });
  if (!res.ok) throw new Error(await readError(res, "重建知识库失败"));
  return (await res.json()) as KnowledgeRebuildResponse;
}

async function readError(res: Response, fallback: string): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: string };
    return data.detail || fallback;
  } catch {
    return `${fallback}: ${res.status}`;
  }
}

export async function speechToText(audio: Blob): Promise<string> {
  const form = new FormData();
  form.append("audio", audio, "recording.webm");

  const res = await fetch("/api/speech/stt", {
    method: "POST",
    body: form,
  });
  if (!res.ok) {
    const detail = await readErrorDetail(res);
    throw new Error(detail || "语音识别不可用");
  }
  const data = (await res.json()) as { text?: string };
  return data.text || "";
}

export async function textToSpeech(text: string): Promise<string> {
  const res = await fetch("/api/speech/tts", {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ text }),
  });
  if (!res.ok) {
    const detail = await readErrorDetail(res);
    throw new Error(detail || "语音播放不可用");
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

async function readErrorDetail(res: Response): Promise<string> {
  try {
    const data = (await res.json()) as { detail?: string };
    return data.detail || "";
  } catch {
    return "";
  }
}

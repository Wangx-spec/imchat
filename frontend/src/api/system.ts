import type { HealthStatus } from "./types";

export async function getHealth(): Promise<HealthStatus | null> {
  try {
    const res = await fetch("/health");
    if (!res.ok) return null;
    return (await res.json()) as HealthStatus;
  } catch {
    return null;
  }
}

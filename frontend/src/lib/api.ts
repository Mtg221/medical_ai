import type { JobStatus } from "@/types/api";

export async function apiRequest<T>(
  path: string,
  options: {
    method?: string;
    token?: string | null;
    body?: unknown;
  } = {}
): Promise<T> {
  const headers: Record<string, string> = {};
  if (options.token) {
    headers.Authorization = `Bearer ${options.token}`;
  }
  if (options.body !== undefined) {
    headers["Content-Type"] = "application/json";
  }

  const res = await fetch(path, {
    method: options.method ?? "GET",
    headers,
    body: options.body !== undefined ? JSON.stringify(options.body) : undefined,
  });

  const payload = (await res.json()) as T & { error?: string; detail?: string };
  if (!res.ok) {
    throw new Error(payload.error || payload.detail || `HTTP ${res.status}`);
  }
  return payload;
}

export async function uploadDocumentMultipart(
  file: File,
  token: string,
  metadata: {
    subject_id: string;
    specialty: string;
    document_type: string;
  }
): Promise<JobStatus> {
  const form = new FormData();
  form.append("file", file);
  form.append("subject_id", metadata.subject_id);
  form.append("specialty", metadata.specialty);
  form.append("document_type", metadata.document_type);

  const res = await fetch("/api/v1/documents/upload", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  const data = (await res.json()) as JobStatus & { error?: string; detail?: string };
  if (!res.ok) {
    throw new Error(data.error || data.detail || `Upload failed (HTTP ${res.status})`);
  }
  return data;
}

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

export async function uploadImageMultipart(
  file: File,
  token: string,
  metadata: {
    subject_id: string;
    body_site?: string;
    captured_date_iso?: string;
    force_new?: boolean;
  }
): Promise<Record<string, unknown>> {
  const form = new FormData();
  form.append("file", file);
  form.append("subject_id", metadata.subject_id);
  if (metadata.body_site) form.append("body_site", metadata.body_site);
  if (metadata.captured_date_iso) form.append("captured_date_iso", metadata.captured_date_iso);
  if (metadata.force_new) form.append("force_new", "true");

  const res = await fetch("/api/v1/images/upload", {
    method: "POST",
    headers: { Authorization: `Bearer ${token}` },
    body: form,
  });
  const data = (await res.json()) as Record<string, unknown> & { error?: string; detail?: string };
  if (!res.ok) {
    throw new Error(data.error || data.detail || `Image upload failed (HTTP ${res.status})`);
  }
  return data;
}

export async function fetchAuthenticatedBlobUrl(path: string, token: string): Promise<string> {
  const res = await fetch(path, {
    headers: { Authorization: `Bearer ${token}` },
  });
  if (!res.ok) {
    throw new Error(`Failed to fetch binary asset (HTTP ${res.status})`);
  }
  const blob = await res.blob();
  return URL.createObjectURL(blob);
}

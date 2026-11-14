export async function apiRequest(path, { method = "GET", body, token, onAuthError } = {}) {
  const headers = {};
  if (body !== undefined) headers["Content-Type"] = "application/json";
  if (token) headers["Authorization"] = `Bearer ${token}`;

  const res = await fetch(path, {
    method,
    headers,
    body: body !== undefined ? JSON.stringify(body) : undefined,
  });

  let data = null;
  try {
    data = await res.json();
  } catch {
    data = null;
  }

  if (!res.ok) {
    const msg = (data && data.error) || `HTTP ${res.status}`;
    if (token && (/Invalid access token signature/i.test(msg) || /Token expired/i.test(msg))) {
      if (onAuthError) onAuthError();
    }
    throw new Error(msg);
  }
  return data;
}

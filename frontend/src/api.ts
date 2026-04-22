/**
 * Thin client for the translatorx-v3 REST API. One typed function per
 * endpoint we use. The API key lives in localStorage under "trx.apiKey"
 * and is attached as the X-API-Key header on every request.
 */

const KEY_STORAGE = "trx.apiKey";

export function getApiKey(): string {
  return localStorage.getItem(KEY_STORAGE) || "";
}

export function setApiKey(key: string): void {
  localStorage.setItem(KEY_STORAGE, key);
}

function headers(extra?: Record<string, string>): Record<string, string> {
  const h: Record<string, string> = { ...(extra || {}) };
  const key = getApiKey();
  if (key) h["X-API-Key"] = key;
  return h;
}

async function request<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(path, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...headers(init?.headers as Record<string, string> | undefined),
    },
  });
  if (!res.ok) {
    const body = await res.text().catch(() => "");
    throw new Error(`${res.status} ${res.statusText}: ${body}`);
  }
  const ct = res.headers.get("content-type") || "";
  if (ct.includes("application/json")) return (await res.json()) as T;
  return (await res.text()) as unknown as T;
}

// ---- Types ---------------------------------------------------------------

export type TaskStatus = "queued" | "running" | "done" | "failed" | "cancelled";

export interface VideoState {
  task_id: string;
  course: string;
  video: string;
  status: TaskStatus;
  stages: string[];
  src: string | null;
  tgt: string[];
  done: number;
  total: number | null;
  error: string | null;
  elapsed_s: number | null;
}

export interface CreateVideoRequest {
  video: string;
  src?: string | null;
  tgt: string[];
  source_kind?: "srt" | "whisperx" | "text" | null;
  source_path?: string | null;
  source_content?: string | null;
  stages?: ("translate" | "align" | "tts" | "summary")[];
  engine?: string;
}

// ---- Health --------------------------------------------------------------

export const health = () => request<{ status: string }>("/health");
export const ready = () => request<{ status: string }>("/ready");

// ---- Videos --------------------------------------------------------------

export const submitVideo = (course: string, body: CreateVideoRequest) =>
  request<VideoState>(`/api/courses/${encodeURIComponent(course)}/videos`, {
    method: "POST",
    body: JSON.stringify(body),
  });

export const listVideos = (course: string) =>
  request<{ items: VideoState[] }>(`/api/courses/${encodeURIComponent(course)}/videos`);

export const getVideo = (course: string, taskId: string) =>
  request<VideoState>(`/api/courses/${encodeURIComponent(course)}/videos/${encodeURIComponent(taskId)}`);

export const cancelVideo = (course: string, taskId: string) =>
  request<VideoState>(`/api/courses/${encodeURIComponent(course)}/videos/${encodeURIComponent(taskId)}/cancel`, {
    method: "POST",
  });

export const getResult = (course: string, video: string, format: "srt" | "json") =>
  request<string>(`/api/courses/${encodeURIComponent(course)}/videos/${encodeURIComponent(video)}/result?format=${format}`);

/**
 * Subscribe to SSE progress events for a task. Returns an `EventSource`
 * the caller must `.close()` when done. Because the EventSource API
 * cannot send arbitrary headers, we append the API key as a query
 * parameter — the backend must accept either form.
 */
export function subscribeVideoEvents(
  course: string,
  taskId: string,
  onEvent: (data: unknown) => void,
): EventSource {
  const url = `/api/courses/${encodeURIComponent(course)}/videos/${encodeURIComponent(taskId)}/events`;
  const es = new EventSource(url);
  es.onmessage = (ev) => {
    try {
      onEvent(JSON.parse(ev.data));
    } catch {
      onEvent(ev.data);
    }
  };
  return es;
}

// ---- Usage ---------------------------------------------------------------

export const usageSelf = (userId: string) =>
  request<Record<string, unknown>>(`/api/usage/${encodeURIComponent(userId)}`);

export const usageSummary = () => request<Record<string, unknown>>(`/api/usage/summary`);

export const usageTop = (limit = 20) =>
  request<{ items: Record<string, unknown>[] }>(`/api/usage/top?limit=${limit}`);

// ---- Admin ---------------------------------------------------------------

export const adminListTasks = () => request<{ items: Record<string, unknown>[] }>("/api/admin/tasks");
export const adminGetTask = (taskId: string) =>
  request<Record<string, unknown>>(`/api/admin/tasks/${encodeURIComponent(taskId)}`);
export const adminCancelTask = (taskId: string) =>
  request<Record<string, unknown>>(`/api/admin/tasks/${encodeURIComponent(taskId)}/cancel`, { method: "POST" });

export const adminListUsers = () => request<{ items: Record<string, unknown>[] }>("/api/admin/users");

export interface UserUpsert {
  api_key: string;
  user_id: string;
  tier: string;
}
export const adminUpsertUser = (body: UserUpsert) =>
  request<Record<string, unknown>>("/api/admin/users", { method: "POST", body: JSON.stringify(body) });
export const adminDeleteUser = (apiKey: string) =>
  request<Record<string, unknown>>(`/api/admin/users/${encodeURIComponent(apiKey)}`, { method: "DELETE" });

export const adminListEngines = () => request<{ items: Record<string, unknown>[] }>("/api/admin/engines");
export const adminListWorkers = () => request<Record<string, unknown>>("/api/admin/workers");
export const adminWorkspace = (course: string) =>
  request<Record<string, unknown>>(`/api/admin/workspace/${encodeURIComponent(course)}`);
export const adminWorkspaceVideo = (course: string, video: string) =>
  request<Record<string, unknown>>(
    `/api/admin/workspace/${encodeURIComponent(course)}/${encodeURIComponent(video)}`,
  );
export const adminErrors = (limit = 100) =>
  request<{ items: Record<string, unknown>[] }>(`/api/admin/errors?limit=${limit}`);
export const adminGetTerms = (src: string, tgt: string) =>
  request<Record<string, unknown>>(`/api/admin/terms/${encodeURIComponent(src)}/${encodeURIComponent(tgt)}`);
export const adminPutTerms = (src: string, tgt: string, terms: Record<string, string>) =>
  request<Record<string, unknown>>(
    `/api/admin/terms/${encodeURIComponent(src)}/${encodeURIComponent(tgt)}`,
    { method: "PUT", body: JSON.stringify({ terms }) },
  );
export const adminGetConfig = () => request<Record<string, unknown>>("/api/admin/config");

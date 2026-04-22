import { useState, type ReactNode } from "react";

/** Reusable error banner. */
export function ErrorBanner({ error }: { error: unknown }) {
  if (!error) return null;
  const msg = error instanceof Error ? error.message : String(error);
  return <div className="error-banner">{msg}</div>;
}

/** `useAsync` — minimal hook: call `run(fn)` to trigger, renders loading/error/data. */
export function useAsync<T>() {
  const [data, setData] = useState<T | null>(null);
  const [err, setErr] = useState<unknown>(null);
  const [loading, setLoading] = useState(false);
  const run = async (fn: () => Promise<T>) => {
    setLoading(true);
    setErr(null);
    try {
      setData(await fn());
    } catch (e) {
      setErr(e);
    } finally {
      setLoading(false);
    }
  };
  return { data, err, loading, run, setData };
}

export function JsonBlock({ value }: { value: unknown }) {
  return <pre>{JSON.stringify(value, null, 2)}</pre>;
}

export function PageHeader({ title, right }: { title: string; right?: ReactNode }) {
  return (
    <div className="header">
      <h2>{title}</h2>
      <div>{right}</div>
    </div>
  );
}

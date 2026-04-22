import { useEffect, useState } from "react";
import { adminGetConfig } from "../api";
import { ErrorBanner, JsonBlock, PageHeader } from "./_shared";

export function AdminConfig() {
  const [data, setData] = useState<unknown>(null);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => { adminGetConfig().then(setData).catch(setErr); }, []);
  return (
    <>
      <PageHeader title="Config (admin, redacted)" />
      <ErrorBanner error={err} />
      <div className="card">{data ? <JsonBlock value={data} /> : <p className="muted">Loading...</p>}</div>
    </>
  );
}

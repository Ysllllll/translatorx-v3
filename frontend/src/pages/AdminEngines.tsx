import { useEffect, useState } from "react";
import { adminListEngines } from "../api";
import { ErrorBanner, JsonBlock, PageHeader } from "./_shared";

export function AdminEngines() {
  const [data, setData] = useState<unknown>(null);
  const [err, setErr] = useState<unknown>(null);
  useEffect(() => { adminListEngines().then(setData).catch(setErr); }, []);
  return (
    <>
      <PageHeader title="Engines (admin)" />
      <ErrorBanner error={err} />
      <div className="card">{data ? <JsonBlock value={data} /> : <p className="muted">Loading...</p>}</div>
    </>
  );
}

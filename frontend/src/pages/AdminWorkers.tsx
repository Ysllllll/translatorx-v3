import { useEffect, useState } from "react";
import { adminListWorkers } from "../api";
import { ErrorBanner, JsonBlock, PageHeader } from "./_shared";

export function AdminWorkers() {
  const [data, setData] = useState<unknown>(null);
  const [err, setErr] = useState<unknown>(null);
  const refresh = () => { adminListWorkers().then(setData).catch(setErr); };
  useEffect(() => { refresh(); const id = setInterval(refresh, 5000); return () => clearInterval(id); }, []);
  return (
    <>
      <PageHeader title="Workers (admin)" right={<button className="secondary" onClick={refresh}>Refresh</button>} />
      <ErrorBanner error={err} />
      <div className="card">{data ? <JsonBlock value={data} /> : <p className="muted">Loading...</p>}</div>
    </>
  );
}

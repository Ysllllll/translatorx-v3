import { useEffect, useState } from "react";
import { adminErrors } from "../api";
import { ErrorBanner, JsonBlock, PageHeader } from "./_shared";

export function AdminErrors() {
  const [data, setData] = useState<unknown>(null);
  const [err, setErr] = useState<unknown>(null);
  const refresh = () => { adminErrors(100).then(setData).catch(setErr); };
  useEffect(() => { refresh(); }, []);
  return (
    <>
      <PageHeader title="Errors (admin)" right={<button className="secondary" onClick={refresh}>Refresh</button>} />
      <ErrorBanner error={err} />
      <div className="card">{data ? <JsonBlock value={data} /> : <p className="muted">Loading...</p>}</div>
    </>
  );
}

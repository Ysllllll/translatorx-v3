import { useEffect, useState } from "react";
import { usageSelf, usageSummary, usageTop } from "../api";
import { ErrorBanner, JsonBlock, PageHeader } from "./_shared";

export function Usage({ apiKey }: { apiKey: string }) {
  const [userId, setUserId] = useState("");
  const [self, setSelf] = useState<unknown>(null);
  const [summary, setSummary] = useState<unknown>(null);
  const [top, setTop] = useState<unknown>(null);
  const [err, setErr] = useState<unknown>(null);

  useEffect(() => {
    // Try /summary + /top; they are admin-gated and will 403 for non-admins.
    usageSummary().then(setSummary).catch(() => {});
    usageTop(20).then(setTop).catch(() => {});
  }, [apiKey]);

  const fetchSelf = async () => {
    setErr(null);
    try { setSelf(await usageSelf(userId)); } catch (e) { setErr(e); }
  };

  return (
    <>
      <PageHeader title="Usage" />
      <ErrorBanner error={err} />

      <div className="card">
        <h3>My ledger</h3>
        <div className="row">
          <input type="text" placeholder="user_id" value={userId} onChange={(e) => setUserId(e.target.value)} />
          <button onClick={fetchSelf} disabled={!userId}>Fetch</button>
        </div>
        {self && <JsonBlock value={self} />}
      </div>

      <div className="grid2">
        <div className="card">
          <h3>Summary (admin)</h3>
          {summary ? <JsonBlock value={summary} /> : <p className="muted">Admin-only; empty if not authorised.</p>}
        </div>
        <div className="card">
          <h3>Top spenders (admin)</h3>
          {top ? <JsonBlock value={top} /> : <p className="muted">Admin-only; empty if not authorised.</p>}
        </div>
      </div>
    </>
  );
}

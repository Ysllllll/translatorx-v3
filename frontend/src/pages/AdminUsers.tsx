import { useEffect, useState } from "react";
import { adminDeleteUser, adminListUsers, adminUpsertUser } from "../api";
import { ErrorBanner, PageHeader } from "./_shared";

export function AdminUsers() {
  const [items, setItems] = useState<Record<string, unknown>[]>([]);
  const [err, setErr] = useState<unknown>(null);
  const [apiKey, setApiKey] = useState("");
  const [userId, setUserId] = useState("");
  const [tier, setTier] = useState("free");

  const refresh = async () => {
    try { setItems((await adminListUsers()).items); } catch (e) { setErr(e); }
  };
  useEffect(() => { refresh(); }, []);

  const add = async () => {
    if (!apiKey || !userId) return;
    try {
      await adminUpsertUser({ api_key: apiKey, user_id: userId, tier });
      setApiKey(""); setUserId("");
      await refresh();
    } catch (e) { setErr(e); }
  };

  const remove = async (k: string) => {
    if (!window.confirm(`Delete key ${k}?`)) return;
    try { await adminDeleteUser(k); await refresh(); } catch (e) { setErr(e); }
  };

  return (
    <>
      <PageHeader title="Users (admin)" />
      <ErrorBanner error={err} />
      <div className="card">
        <h3>Add / update API key</h3>
        <div className="row">
          <input placeholder="api_key" value={apiKey} onChange={(e) => setApiKey(e.target.value)} />
          <input placeholder="user_id" value={userId} onChange={(e) => setUserId(e.target.value)} />
          <select value={tier} onChange={(e) => setTier(e.target.value)} style={{ width: 120 }}>
            <option>free</option><option>paid</option><option>admin</option>
          </select>
          <button onClick={add}>Save</button>
        </div>
      </div>
      <div className="card">
        <table>
          <thead><tr><th>API key</th><th>User</th><th>Tier</th><th></th></tr></thead>
          <tbody>
            {items.map((u, i) => (
              <tr key={i}>
                <td><code>{String(u.api_key)}</code></td>
                <td>{String(u.user_id)}</td>
                <td>{String(u.tier)}</td>
                <td><button className="danger" onClick={() => remove(String(u.api_key))}>Delete</button></td>
              </tr>
            ))}
            {items.length === 0 && <tr><td colSpan={4} className="muted">No users configured.</td></tr>}
          </tbody>
        </table>
      </div>
    </>
  );
}

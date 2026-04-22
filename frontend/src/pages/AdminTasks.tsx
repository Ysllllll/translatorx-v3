import { useEffect, useState } from "react";
import { adminCancelTask, adminListTasks } from "../api";
import { ErrorBanner, JsonBlock, PageHeader } from "./_shared";

export function AdminTasks() {
  const [items, setItems] = useState<Record<string, unknown>[]>([]);
  const [err, setErr] = useState<unknown>(null);
  const [sel, setSel] = useState<Record<string, unknown> | null>(null);

  const refresh = async () => {
    try { setItems((await adminListTasks()).items); } catch (e) { setErr(e); }
  };
  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 5000);
    return () => clearInterval(id);
  }, []);

  const cancel = async (taskId: string) => {
    try { await adminCancelTask(taskId); await refresh(); } catch (e) { setErr(e); }
  };

  return (
    <>
      <PageHeader title="All tasks (admin)" right={<button className="secondary" onClick={refresh}>Refresh</button>} />
      <ErrorBanner error={err} />
      <div className="card">
        <table>
          <thead><tr><th>Task</th><th>Course</th><th>Video</th><th>Status</th><th>User</th><th></th></tr></thead>
          <tbody>
            {items.map((t) => {
              const id = String(t.task_id);
              const status = String(t.status);
              return (
                <tr key={id}>
                  <td><code>{id.slice(0, 8)}</code></td>
                  <td>{String(t.course)}</td>
                  <td>{String(t.video)}</td>
                  <td><span className={`status ${status}`}>{status}</span></td>
                  <td>{String(t.user_id ?? "-")}</td>
                  <td>
                    <button className="secondary" onClick={() => setSel(t)}>Details</button>{" "}
                    {(status === "queued" || status === "running") && (
                      <button className="danger" onClick={() => cancel(id)}>Cancel</button>
                    )}
                  </td>
                </tr>
              );
            })}
            {items.length === 0 && <tr><td colSpan={6} className="muted">No tasks.</td></tr>}
          </tbody>
        </table>
      </div>
      {sel && <div className="card"><h3>Task {String(sel.task_id).slice(0, 8)}</h3><JsonBlock value={sel} /></div>}
    </>
  );
}

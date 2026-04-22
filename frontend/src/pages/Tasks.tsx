import { useEffect, useRef, useState } from "react";
import { cancelVideo, getResult, listVideos, subscribeVideoEvents, type VideoState } from "../api";
import { ErrorBanner, PageHeader } from "./_shared";

export function Tasks() {
  const [course, setCourse] = useState("demo-course");
  const [items, setItems] = useState<VideoState[]>([]);
  const [err, setErr] = useState<unknown>(null);
  const [selected, setSelected] = useState<string | null>(null);
  const [events, setEvents] = useState<unknown[]>([]);
  const [resultText, setResultText] = useState<string | null>(null);
  const esRef = useRef<EventSource | null>(null);

  const refresh = async () => {
    try {
      const r = await listVideos(course);
      setItems(r.items);
    } catch (e) {
      setErr(e);
    }
  };

  useEffect(() => {
    refresh();
    const id = setInterval(refresh, 3000);
    return () => clearInterval(id);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [course]);

  const watch = (task: VideoState) => {
    esRef.current?.close();
    setEvents([]);
    setResultText(null);
    setSelected(task.task_id);
    const es = subscribeVideoEvents(course, task.task_id, (data) => {
      setEvents((prev) => [...prev, data].slice(-200));
    });
    es.onerror = () => { /* EventSource auto-reconnects; nothing to do */ };
    esRef.current = es;
  };

  const cancel = async (task: VideoState) => {
    try { await cancelVideo(course, task.task_id); await refresh(); } catch (e) { setErr(e); }
  };

  const fetchResult = async (task: VideoState) => {
    try {
      const s = await getResult(course, task.video, "srt");
      setResultText(s);
    } catch (e) { setErr(e); }
  };

  useEffect(() => () => esRef.current?.close(), []);

  return (
    <>
      <PageHeader
        title="My tasks"
        right={
          <div className="row">
            <label style={{ margin: 0 }}>Course</label>
            <input type="text" value={course} onChange={(e) => setCourse(e.target.value)} />
            <button className="secondary" onClick={refresh}>Refresh</button>
          </div>
        }
      />
      <ErrorBanner error={err} />
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>Task</th><th>Video</th><th>Status</th><th>Progress</th><th>Stages</th><th></th>
            </tr>
          </thead>
          <tbody>
            {items.map((t) => (
              <tr key={t.task_id}>
                <td><code>{t.task_id.slice(0, 8)}</code></td>
                <td>{t.video}</td>
                <td><span className={`status ${t.status}`}>{t.status}</span></td>
                <td>{t.done}{t.total != null ? ` / ${t.total}` : ""}</td>
                <td>{t.stages.join(" → ")}</td>
                <td>
                  <button className="secondary" onClick={() => watch(t)}>Watch</button>{" "}
                  {t.status === "done" && <button className="secondary" onClick={() => fetchResult(t)}>Result</button>}{" "}
                  {(t.status === "queued" || t.status === "running") && (
                    <button className="danger" onClick={() => cancel(t)}>Cancel</button>
                  )}
                </td>
              </tr>
            ))}
            {items.length === 0 && <tr><td colSpan={6} className="muted">No tasks in this course.</td></tr>}
          </tbody>
        </table>
      </div>
      {selected && (
        <div className="card">
          <h3>Events for {selected.slice(0, 8)} <span className="muted">(live)</span></h3>
          <pre>{events.map((e, i) => `${i}: ${JSON.stringify(e)}`).join("\n")}</pre>
        </div>
      )}
      {resultText && (
        <div className="card">
          <h3>Result (SRT)</h3>
          <pre>{resultText}</pre>
        </div>
      )}
    </>
  );
}

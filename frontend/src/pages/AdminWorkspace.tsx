import { useState } from "react";
import { adminWorkspace, adminWorkspaceVideo } from "../api";
import { ErrorBanner, JsonBlock, PageHeader } from "./_shared";

export function AdminWorkspace() {
  const [course, setCourse] = useState("");
  const [video, setVideo] = useState("");
  const [data, setData] = useState<unknown>(null);
  const [err, setErr] = useState<unknown>(null);

  const load = async () => {
    setErr(null);
    try {
      setData(video ? await adminWorkspaceVideo(course, video) : await adminWorkspace(course));
    } catch (e) { setErr(e); }
  };

  return (
    <>
      <PageHeader title="Workspace browser (admin)" />
      <ErrorBanner error={err} />
      <div className="card">
        <div className="row">
          <input placeholder="course" value={course} onChange={(e) => setCourse(e.target.value)} />
          <input placeholder="video (optional)" value={video} onChange={(e) => setVideo(e.target.value)} />
          <button onClick={load} disabled={!course}>Load</button>
        </div>
      </div>
      {data && <div className="card"><JsonBlock value={data} /></div>}
    </>
  );
}

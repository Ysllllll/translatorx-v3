import { useState } from "react";
import { submitVideo } from "../api";
import { ErrorBanner, JsonBlock, PageHeader, useAsync } from "./_shared";
import type { VideoState } from "../api";

export function Submit() {
  const [course, setCourse] = useState("demo-course");
  const [video, setVideo] = useState("lec01");
  const [src, setSrc] = useState("en");
  const [tgt, setTgt] = useState("zh");
  const [engine, setEngine] = useState("default");
  const [stages, setStages] = useState("translate");
  const [sourceKind, setSourceKind] = useState<"srt" | "text">("srt");
  const [sourceContent, setSourceContent] = useState(
    "1\n00:00:00,000 --> 00:00:03,000\nHello world.\n",
  );
  const { data, err, loading, run } = useAsync<VideoState>();

  const onSubmit = () => {
    run(() =>
      submitVideo(course, {
        video,
        src,
        tgt: tgt.split(",").map((s) => s.trim()).filter(Boolean),
        engine,
        source_kind: sourceKind,
        source_content: sourceContent,
        stages: stages.split(",").map((s) => s.trim()).filter(Boolean) as never,
      }),
    );
  };

  return (
    <>
      <PageHeader title="Submit a translation task" />
      <ErrorBanner error={err} />
      <div className="card">
        <div className="grid2">
          <div>
            <label>Course</label>
            <input type="text" value={course} onChange={(e) => setCourse(e.target.value)} />
            <label>Video</label>
            <input type="text" value={video} onChange={(e) => setVideo(e.target.value)} />
            <label>Source lang</label>
            <input type="text" value={src} onChange={(e) => setSrc(e.target.value)} />
            <label>Target langs (comma-separated)</label>
            <input type="text" value={tgt} onChange={(e) => setTgt(e.target.value)} />
          </div>
          <div>
            <label>Engine</label>
            <input type="text" value={engine} onChange={(e) => setEngine(e.target.value)} />
            <label>Stages</label>
            <input type="text" value={stages} onChange={(e) => setStages(e.target.value)} />
            <label>Source kind</label>
            <select value={sourceKind} onChange={(e) => setSourceKind(e.target.value as "srt" | "text")}>
              <option value="srt">srt</option>
              <option value="text">text</option>
            </select>
          </div>
        </div>
        <label>Source content</label>
        <textarea value={sourceContent} onChange={(e) => setSourceContent(e.target.value)} />
        <div className="row end" style={{ marginTop: 12 }}>
          <button onClick={onSubmit} disabled={loading}>
            {loading ? "Submitting..." : "Submit"}
          </button>
        </div>
      </div>
      {data && (
        <div className="card">
          <h3>Accepted</h3>
          <p>
            Task ID: <code>{data.task_id}</code> — open <a href={`#/tasks`}>My tasks</a> to follow progress.
          </p>
          <JsonBlock value={data} />
        </div>
      )}
    </>
  );
}

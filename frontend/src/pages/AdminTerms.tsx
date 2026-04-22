import { useState } from "react";
import { adminGetTerms, adminPutTerms } from "../api";
import { ErrorBanner, PageHeader } from "./_shared";

export function AdminTerms() {
  const [src, setSrc] = useState("en");
  const [tgt, setTgt] = useState("zh");
  const [text, setText] = useState("");
  const [err, setErr] = useState<unknown>(null);
  const [msg, setMsg] = useState<string>("");

  const load = async () => {
    setErr(null); setMsg("");
    try {
      const r = await adminGetTerms(src, tgt);
      const terms = (r.terms ?? {}) as Record<string, string>;
      setText(Object.entries(terms).map(([k, v]) => `${k}\t${v}`).join("\n"));
    } catch (e) { setErr(e); }
  };

  const save = async () => {
    setErr(null); setMsg("");
    const terms: Record<string, string> = {};
    for (const line of text.split("\n")) {
      const [k, v] = line.split("\t");
      if (k && v) terms[k.trim()] = v.trim();
    }
    try { await adminPutTerms(src, tgt, terms); setMsg(`Saved ${Object.keys(terms).length} terms.`); }
    catch (e) { setErr(e); }
  };

  return (
    <>
      <PageHeader title="Terms (admin)" />
      <ErrorBanner error={err} />
      <div className="card">
        <div className="row">
          <input placeholder="src" value={src} onChange={(e) => setSrc(e.target.value)} style={{ width: 80 }} />
          <input placeholder="tgt" value={tgt} onChange={(e) => setTgt(e.target.value)} style={{ width: 80 }} />
          <button className="secondary" onClick={load}>Load</button>
          <button onClick={save}>Save</button>
          {msg && <span className="muted">{msg}</span>}
        </div>
        <label>One term per line, tab-separated: <code>source&lt;TAB&gt;target</code></label>
        <textarea value={text} onChange={(e) => setText(e.target.value)} style={{ minHeight: 300 }} />
      </div>
    </>
  );
}

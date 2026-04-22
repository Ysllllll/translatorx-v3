import { useEffect, useState } from "react";
import { getApiKey, setApiKey } from "./api";
import { Submit } from "./pages/Submit";
import { Tasks } from "./pages/Tasks";
import { Usage } from "./pages/Usage";
import { AdminTasks } from "./pages/AdminTasks";
import { AdminUsers } from "./pages/AdminUsers";
import { AdminEngines } from "./pages/AdminEngines";
import { AdminWorkers } from "./pages/AdminWorkers";
import { AdminWorkspace } from "./pages/AdminWorkspace";
import { AdminTerms } from "./pages/AdminTerms";
import { AdminErrors } from "./pages/AdminErrors";
import { AdminConfig } from "./pages/AdminConfig";

type Route =
  | "submit"
  | "tasks"
  | "usage"
  | "admin-tasks"
  | "admin-users"
  | "admin-engines"
  | "admin-workers"
  | "admin-workspace"
  | "admin-terms"
  | "admin-errors"
  | "admin-config";

const DEFAULT_ROUTE: Route = "submit";

function parseHash(): Route {
  const h = window.location.hash.replace(/^#\/?/, "");
  return (h as Route) || DEFAULT_ROUTE;
}

export function App() {
  const [route, setRoute] = useState<Route>(parseHash());
  const [key, setKey] = useState<string>(getApiKey());
  const [keyDraft, setKeyDraft] = useState<string>(key);

  useEffect(() => {
    const onHash = () => setRoute(parseHash());
    window.addEventListener("hashchange", onHash);
    return () => window.removeEventListener("hashchange", onHash);
  }, []);

  const nav = (r: Route) => (e: React.MouseEvent) => {
    e.preventDefault();
    window.location.hash = `#/${r}`;
    setRoute(r);
  };

  const saveKey = () => {
    setApiKey(keyDraft);
    setKey(keyDraft);
  };

  const content = (() => {
    switch (route) {
      case "submit":          return <Submit />;
      case "tasks":           return <Tasks />;
      case "usage":           return <Usage apiKey={key} />;
      case "admin-tasks":     return <AdminTasks />;
      case "admin-users":     return <AdminUsers />;
      case "admin-engines":   return <AdminEngines />;
      case "admin-workers":   return <AdminWorkers />;
      case "admin-workspace": return <AdminWorkspace />;
      case "admin-terms":     return <AdminTerms />;
      case "admin-errors":    return <AdminErrors />;
      case "admin-config":    return <AdminConfig />;
      default:                return <Submit />;
    }
  })();

  const navItem = (r: Route, label: string) => (
    <a href={`#/${r}`} onClick={nav(r)} className={`nav-item ${route === r ? "active" : ""}`}>
      {label}
    </a>
  );

  return (
    <div className="layout">
      <aside className="sidebar">
        <h1>translatorx</h1>
        <div className="section">User</div>
        {navItem("submit", "Submit")}
        {navItem("tasks", "My tasks")}
        {navItem("usage", "Usage")}
        <div className="section">Admin</div>
        {navItem("admin-tasks", "All tasks")}
        {navItem("admin-users", "Users")}
        {navItem("admin-engines", "Engines")}
        {navItem("admin-workers", "Workers")}
        {navItem("admin-workspace", "Workspace")}
        {navItem("admin-terms", "Terms")}
        {navItem("admin-errors", "Errors")}
        {navItem("admin-config", "Config")}
        <div className="section">API key</div>
        <input
          type="password"
          value={keyDraft}
          onChange={(e) => setKeyDraft(e.target.value)}
          placeholder="X-API-Key"
        />
        <div className="row" style={{ marginTop: 6 }}>
          <button onClick={saveKey}>Save</button>
          {key && <span className="muted" style={{ fontSize: 11 }}>saved</span>}
        </div>
      </aside>
      <main className="main">{content}</main>
    </div>
  );
}

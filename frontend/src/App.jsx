import { NavLink, Route, Routes } from "react-router-dom";
import Overview from "./pages/Overview.jsx";
import Suppliers from "./pages/Suppliers.jsx";
import SupplierView from "./pages/SupplierView.jsx";
import Parts from "./pages/Parts.jsx";
import PartView from "./pages/PartView.jsx";
import Network from "./pages/Network.jsx";
import Simulator from "./pages/Simulator.jsx";
import Mitigation from "./pages/Mitigation.jsx";
import Model from "./pages/Model.jsx";
import { DEMO } from "./api.js";

export default function App() {
  return (
    <div className="app">
      <aside className="sidebar">
        <div className="brand">
          <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden>
            <path d="M6 16 26 7M6 16l20 9M26 7v18" stroke="#3987e5" strokeWidth="1.6" />
            <circle cx="6" cy="16" r="4" fill="#3987e5" />
            <circle cx="26" cy="7" r="3.2" fill="#1a1a19" stroke="#3987e5" strokeWidth="2" />
            <circle cx="26" cy="25" r="3.2" fill="#1a1a19" stroke="#3987e5" strokeWidth="2" />
          </svg>
          <div>
            <strong>Aero</strong>
            <span>Supply</span>
          </div>
        </div>
        <nav>
          <NavLink to="/" end>Overview</NavLink>
          <NavLink to="/suppliers">Suppliers</NavLink>
          <NavLink to="/parts">Parts & stock-outs</NavLink>
          <NavLink to="/network">Network</NavLink>
          <NavLink to="/simulate">Disruption simulator</NavLink>
          <NavLink to="/mitigation">Mitigation</NavLink>
          <NavLink to="/model">Model card</NavLink>
        </nav>
        <footer className="muted small">
          Synthetic aerospace supply chain
          <br />
          111 suppliers · 217 parts · 4 programs
        </footer>
      </aside>
      <main className="content">
        {DEMO && (
          <div className="demo-banner">
            Static demo: real pipeline outputs frozen on 1 Nov 2026 (synthetic data). What-if scenarios and budgets are
            pre-computed on a grid. Full stack (FastAPI, PostgreSQL, MongoDB, Airflow):{" "}
            <a href="https://github.com/stevvvvvv/AeroSupply">GitHub</a>.
          </div>
        )}
        <Routes>
          <Route path="/" element={<Overview />} />
          <Route path="/suppliers" element={<Suppliers />} />
          <Route path="/suppliers/:id" element={<SupplierView />} />
          <Route path="/parts" element={<Parts />} />
          <Route path="/parts/:id" element={<PartView />} />
          <Route path="/network" element={<Network />} />
          <Route path="/simulate" element={<Simulator />} />
          <Route path="/mitigation" element={<Mitigation />} />
          <Route path="/model" element={<Model />} />
        </Routes>
      </main>
    </div>
  );
}

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
      <header className="topbar">
        <div className="topbar-inner">
          <NavLink to="/" className="brand" aria-label="AeroSupply home">
            <svg viewBox="0 0 32 32" width="26" height="26" aria-hidden>
              <path d="M5 16h7M20 9l7-3M20 23l7 3M12 16l8-7M12 16l8 7" stroke="#a78bfa" strokeWidth="2" fill="none" />
              <circle cx="5" cy="16" r="3.5" fill="#fff" />
              <circle cx="12" cy="16" r="3" fill="#a78bfa" />
              <circle cx="20" cy="9" r="3" fill="#fff" />
              <circle cx="20" cy="23" r="3" fill="#fff" />
            </svg>
            <span className="brand-word">Aero<b>Supply</b></span>
          </NavLink>
          <nav>
            <NavLink to="/" end>Overview</NavLink>
            <NavLink to="/suppliers">Suppliers</NavLink>
            <NavLink to="/parts">Parts</NavLink>
            <NavLink to="/network">Network</NavLink>
            <NavLink to="/simulate">Simulator</NavLink>
            <NavLink to="/mitigation">Mitigation</NavLink>
            <NavLink to="/model">Model card</NavLink>
          </nav>
          <span className="asof">Data as of 1 Nov 2026 · synthetic</span>
        </div>
      </header>
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

// Light theme. Categorical slots = light steps of the validated reference data-viz palette
// (fixed order, colour follows the entity). Status colours are reserved and always ship with an icon.
export const C = {
  surface: "#ffffff",
  paper: "#f4f6f9",
  grid: "#e8ebf0",
  axis: "#b9c0cc",
  text: "#0f1b2d",
  text2: "#46536a",
  muted: "#7d889b",
  accent: "#4b38c9",
  series: ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"], // blue, orange, aqua, yellow
  band: "rgba(42,120,214,0.16)",
  neg: "#e34948", // diverging: raises risk
  pos: "#2a78d6", // diverging: lowers risk
  edge: "rgba(15,27,45,0.10)",
  edgeHi: "rgba(75,56,201,0.55)",
  edgeDim: "rgba(15,27,45,0.04)",
};

export const PROGRAM_COLOR = { "NB-200": C.series[0], "WB-350": C.series[1], "RJ-72": C.series[2], "BJ-6": C.series[3] };

export const STATUS = {
  HEALTHY: { color: "#0ca30c", label: "Healthy", icon: "●" },
  WATCH: { color: "#fab219", label: "Watch", icon: "◐" },
  WARNING: { color: "#ec835a", label: "Warning", icon: "▲" },
  CRITICAL: { color: "#d03b3b", label: "Critical", icon: "■" },
};

export const baseLayout = (extra = {}) => ({
  paper_bgcolor: "rgba(0,0,0,0)",
  plot_bgcolor: "rgba(0,0,0,0)",
  font: { family: "'Archivo Variable', system-ui, sans-serif", color: C.text2, size: 12 },
  margin: { l: 52, r: 16, t: 12, b: 40 },
  hoverlabel: { bgcolor: C.text, bordercolor: C.text, font: { color: "#ffffff", size: 12, family: "'Archivo Variable', sans-serif" } },
  legend: { orientation: "h", y: 1.14, x: 0, font: { color: C.text2 } },
  xaxis: { gridcolor: C.grid, zeroline: false, linecolor: C.axis, tickcolor: C.axis, ...(extra.xaxis || {}) },
  yaxis: { gridcolor: C.grid, zeroline: false, linecolor: C.axis, tickcolor: C.axis, ...(extra.yaxis || {}) },
  ...Object.fromEntries(Object.entries(extra).filter(([k]) => !["xaxis", "yaxis"].includes(k))),
});

export const plotConfig = { displaylogo: false, responsive: true, modeBarButtonsToRemove: ["lasso2d", "select2d", "autoScale2d"] };

/* =========================================================================
   GeoPINN demo site — populate metrics and draw the per-timestep L2 chart
   ========================================================================= */

async function loadMetadata() {
  const resp = await fetch("assets/metadata.json");
  return resp.json();
}

function fmt(x, digits = 4) {
  return Number(x).toFixed(digits);
}

function populate(meta) {
  const tr = meta.training;
  const ev = meta.evaluation;
  const md = meta.model;
  const rd = meta.render;

  // Hero
  document.getElementById("stat-l2").textContent = fmt(ev.mean_l2, 4);
  document.getElementById("stat-psnr").textContent =
    fmt(ev.mean_psnr_db, 1) + " dB";
  document.getElementById("stat-params").textContent =
    md.n_params.toLocaleString();
  document.getElementById("stat-train").textContent = "38 s";

  // Metrics section
  document.getElementById("m-l2").textContent = fmt(ev.mean_l2, 4);
  document.getElementById("m-psnr").textContent =
    fmt(ev.mean_psnr_db, 2) + " dB";
  document.getElementById("m-res").textContent =
    tr.final_pde_residual.toExponential(2);
  document.getElementById("m-params").textContent =
    md.n_params.toLocaleString();
  document.getElementById("m-sphere").textContent =
    rd.sphere_resolution.vertices.toLocaleString() + " verts";
  document.getElementById("m-train").textContent = "38 seconds";

  drawL2Chart(ev.timesteps, ev.l2_per_timestep, ev.relative_l2_per_timestep);
}

function drawL2Chart(timesteps, l2, relL2) {
  const canvas = document.getElementById("l2-chart");
  if (!canvas) return;

  const dpr = window.devicePixelRatio || 1;
  const rect = canvas.getBoundingClientRect();
  canvas.width = rect.width * dpr;
  canvas.height = rect.height * dpr;

  const ctx = canvas.getContext("2d");
  ctx.scale(dpr, dpr);
  const W = rect.width;
  const H = rect.height;
  ctx.clearRect(0, 0, W, H);

  const PAD_L = 50, PAD_R = 60, PAD_T = 24, PAD_B = 38;
  const plotW = W - PAD_L - PAD_R;
  const plotH = H - PAD_T - PAD_B;

  const tMin = Math.min(...timesteps);
  const tMax = Math.max(...timesteps);
  const yMaxL2 = Math.max(...l2) * 1.15;
  const yMaxRel = Math.max(...relL2) * 1.15;

  const xScale = (t) => PAD_L + ((t - tMin) / (tMax - tMin)) * plotW;
  const yL2 = (v) => PAD_T + plotH - (v / yMaxL2) * plotH;
  const yRel = (v) => PAD_T + plotH - (v / yMaxRel) * plotH;

  // Grid + axes
  ctx.strokeStyle = "rgba(90, 110, 150, 0.12)";
  ctx.lineWidth = 1;
  ctx.fillStyle = "#5d6578";
  ctx.font = '11px "SF Mono", Menlo, monospace';

  // horizontal gridlines
  const gridN = 4;
  for (let i = 0; i <= gridN; i++) {
    const y = PAD_T + (i / gridN) * plotH;
    ctx.beginPath();
    ctx.moveTo(PAD_L, y);
    ctx.lineTo(W - PAD_R, y);
    ctx.stroke();

    // left y-axis (L2 absolute)
    const vL2 = yMaxL2 * (1 - i / gridN);
    ctx.textAlign = "right";
    ctx.textBaseline = "middle";
    ctx.fillStyle = "#6aa3ff";
    ctx.fillText(vL2.toFixed(4), PAD_L - 8, y);

    // right y-axis (rel L2 %)
    const vRel = yMaxRel * (1 - i / gridN) * 100;
    ctx.textAlign = "left";
    ctx.fillStyle = "#f08958";
    ctx.fillText(vRel.toFixed(1) + "%", W - PAD_R + 8, y);
  }

  // x-axis ticks
  ctx.fillStyle = "#8b93a6";
  ctx.textAlign = "center";
  ctx.textBaseline = "top";
  const tickN = 6;
  for (let i = 0; i <= tickN; i++) {
    const t = tMin + (tMax - tMin) * (i / tickN);
    const x = xScale(t);
    ctx.fillText("t=" + t.toFixed(2), x, PAD_T + plotH + 8);
  }

  // rel L2 line (warm)
  ctx.strokeStyle = "#f08958";
  ctx.lineWidth = 2;
  ctx.beginPath();
  timesteps.forEach((t, i) => {
    const x = xScale(t);
    const y = yRel(relL2[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // rel L2 area fill
  ctx.lineTo(xScale(tMax), PAD_T + plotH);
  ctx.lineTo(xScale(tMin), PAD_T + plotH);
  ctx.closePath();
  ctx.fillStyle = "rgba(240, 137, 88, 0.08)";
  ctx.fill();

  // abs L2 line (cool)
  ctx.strokeStyle = "#6aa3ff";
  ctx.lineWidth = 2;
  ctx.beginPath();
  timesteps.forEach((t, i) => {
    const x = xScale(t);
    const y = yL2(l2[i]);
    if (i === 0) ctx.moveTo(x, y);
    else ctx.lineTo(x, y);
  });
  ctx.stroke();

  // Dots on abs L2
  ctx.fillStyle = "#6aa3ff";
  timesteps.forEach((t, i) => {
    const x = xScale(t);
    const y = yL2(l2[i]);
    ctx.beginPath();
    ctx.arc(x, y, 3, 0, Math.PI * 2);
    ctx.fill();
  });

  // Legend
  const legY = PAD_T + 4;
  ctx.font = '11px "SF Mono", Menlo, monospace';
  ctx.textAlign = "left";
  ctx.textBaseline = "top";
  ctx.fillStyle = "#6aa3ff";
  ctx.fillText("── absolute L2", PAD_L + 8, legY);
  ctx.fillStyle = "#f08958";
  ctx.fillText("── relative L2 (%)", PAD_L + 130, legY);
}

// Boot
loadMetadata().then(populate).catch((e) => {
  console.error("Failed to load metadata", e);
});

// Redraw chart on resize
let resizeT;
window.addEventListener("resize", () => {
  clearTimeout(resizeT);
  resizeT = setTimeout(() => {
    loadMetadata().then(populate);
  }, 150);
});

const state = {
  motionPaused: false,
  data: null,
};

function qs(id) {
  return document.getElementById(id);
}

function formatFixed(value, digits = 2) {
  return Number(value).toFixed(digits);
}

function formatSci(value, digits = 2) {
  return Number(value).toExponential(digits);
}

function assetPath(fileName) {
  return fileName ? `assets/media/${fileName}` : "";
}

function fillImage(id, fileName) {
  const node = qs(id);
  if (!node) return;
  if (!fileName) {
    replaceWithMissing(node, "Image missing");
    return;
  }
  node.src = assetPath(fileName);
}

function fillVideo(id, srcName, posterName) {
  const node = qs(id);
  if (!node) return;
  if (!srcName) {
    replaceWithMissing(node, "Video missing");
    return;
  }
  node.src = assetPath(srcName);
  if (posterName) {
    node.poster = assetPath(posterName);
  }
  node.load();
}

function replaceWithMissing(node, message) {
  const fallback = document.createElement("div");
  fallback.className = "asset-missing";
  fallback.textContent = message;
  node.replaceWith(fallback);
}

function makeChip(label, value) {
  const chip = document.createElement("span");
  chip.className = "chip";
  chip.textContent = `${label}: ${value}`;
  return chip;
}

function makeStripChip(label, value) {
  const chip = document.createElement("span");
  chip.className = "strip-chip";
  chip.textContent = `${label}: ${value}`;
  return chip;
}

function populateHero(data) {
  qs("hero-title").textContent = data.title;
  qs("hero-subtitle").textContent = data.subtitle;

  fillVideo("hero-video", data.assets.hero_video.src, data.assets.hero_video.poster);
  fillVideo("taichi-video", data.assets.hero_video.src, data.assets.hero_video.poster);

  const strip = qs("hero-strip");
  strip.innerHTML = "";
  strip.append(
    makeStripChip("Preset", data.taichi.preset),
    makeStripChip("Sampling grid", `${data.taichi.resolution} x ${data.taichi.resolution}`),
    makeStripChip("Export frames", data.taichi.export_frames),
    makeStripChip("Export fps", data.taichi.export_fps),
    makeStripChip("Renderer", `Taichi GGUI / ${data.taichi.arch}`),
    makeStripChip("Surface", "fixed-topology height field"),
  );
}

function populateComparison(data) {
  fillVideo("reference-motion", data.assets.reference.motion, data.assets.reference.poster);
  fillVideo("learned-motion", data.assets.learned.motion, data.assets.learned.poster);

  fillImage("reference-snapshots", data.assets.reference.snapshots);
  fillImage("reference-slice", data.assets.reference.slice);
  fillImage("learned-snapshots", data.assets.learned.snapshots);
  fillImage("learned-slice", data.assets.learned.slice);

  fillImage("paired-reference-snapshots", data.assets.reference.snapshots);
  fillImage("paired-reference-slice", data.assets.reference.slice);
  fillImage("paired-learned-snapshots", data.assets.learned.snapshots);
  fillImage("paired-learned-slice", data.assets.learned.slice);

  const referenceChips = qs("reference-chips");
  referenceChips.innerHTML = "";
  referenceChips.append(
    makeChip("Reference grid", `${data.reference_metadata.resolution} x ${data.reference_metadata.resolution}`),
    makeChip("Reference frames", data.reference_metadata.frames),
    makeChip(
      "Time window",
      `${formatFixed(data.reference_metadata.t_start, 2)} to ${formatFixed(data.reference_metadata.t_end, 2)} s`,
    ),
    makeChip("Source", `x ${formatFixed(data.reference_metadata.source_x, 2)}, z ${formatFixed(data.reference_metadata.source_z, 2)}`),
    makeChip("Baseline", "non-neural reference field"),
  );

  const learnedChips = qs("learned-chips");
  learnedChips.innerHTML = "";
  learnedChips.append(
    makeChip("PSNR", `${formatFixed(data.learned_metrics.final_metrics.psnr_db, 2)} dB`),
    makeChip("MSE", formatSci(data.learned_metrics.final_metrics.mse, 2)),
    makeChip("MAE", formatFixed(data.learned_metrics.final_metrics.mae, 4)),
    makeChip("Relative L2", formatFixed(data.learned_metrics.final_metrics.relative_l2, 3)),
    makeChip("Validation grid", `${data.learned_metrics.resolution} x ${data.learned_metrics.resolution}`),
    makeChip("Training steps", `${data.learned_metrics.training_args.steps}`),
  );
}

function populateTaichi(data) {
  fillImage("preset-research", data.assets.taichi_presets.research);
  fillImage("preset-pitch", data.assets.taichi_presets.pitch);
  fillImage("preset-dramatic", data.assets.taichi_presets.dramatic);
}

function buildMetricCard({ label, value, context }) {
  const template = qs("metric-card-template");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".metric-label").textContent = label;
  node.querySelector(".metric-value").textContent = value;
  node.querySelector(".metric-context").textContent = context;
  return node;
}

function populateMetrics(data) {
  const metricsGrid = qs("metrics-grid");
  metricsGrid.innerHTML = "";
  const items = [
    {
      label: "PSNR",
      value: `${formatFixed(data.learned_metrics.final_metrics.psnr_db, 2)} dB`,
      context: "Learned WavePINN fit against the reference field.",
    },
    {
      label: "MSE",
      value: formatSci(data.learned_metrics.final_metrics.mse, 2),
      context: "Final supervised WavePINN checkpoint error.",
    },
    {
      label: "MAE",
      value: formatFixed(data.learned_metrics.final_metrics.mae, 4),
      context: "Average absolute field error.",
    },
    {
      label: "Relative L2",
      value: formatFixed(data.learned_metrics.final_metrics.relative_l2, 3),
      context: "Structured fit quality rather than a random field.",
    },
    {
      label: "Taichi grid",
      value: `${data.taichi.resolution} x ${data.taichi.resolution}`,
      context: "Sampling resolution used for the final dramatic export.",
    },
    {
      label: "Export clip",
      value: `${data.taichi.export_frames} @ ${data.taichi.export_fps} fps`,
      context: "Deterministic prerecorded Taichi sequence.",
    },
  ];
  items.forEach((item) => metricsGrid.append(buildMetricCard(item)));

  const referenceNote = qs("reference-metric-note");
  referenceNote.textContent =
    `Reference target: ${data.reference_metadata.resolution} x ${data.reference_metadata.resolution} over ` +
    `${data.reference_metadata.frames} frames from ${formatFixed(data.reference_metadata.t_start, 2)} to ` +
    `${formatFixed(data.reference_metadata.t_end, 2)} s, source at x=${formatFixed(data.reference_metadata.source_x, 2)}, ` +
    `z=${formatFixed(data.reference_metadata.source_z, 2)}.`;
}

function buildPathItem(label, value) {
  const template = qs("path-item-template");
  const node = template.content.firstElementChild.cloneNode(true);
  node.querySelector(".path-label").textContent = label;
  node.querySelector(".path-code").textContent = value;
  return node;
}

function populatePaths(data) {
  const list = qs("path-list");
  list.innerHTML = "";
  const order = [
    ["Final Taichi clip", data.paths.taichi_video],
    ["Final Taichi metadata", data.paths.taichi_metadata],
    ["Trained checkpoint", data.paths.checkpoint],
    ["Learned snapshots", data.paths.learned_snapshots],
    ["Learned x-t slice", data.paths.learned_slice],
    ["Learned metrics", data.paths.learned_metrics],
    ["Learned wavefield sequence", data.paths.learned_sequence],
    ["Reference snapshots", data.paths.reference_snapshots],
    ["Reference x-t slice", data.paths.reference_slice],
    ["Reference metadata", data.paths.reference_metadata],
    ["Reference wavefield sequence", data.paths.reference_sequence],
  ];
  order
    .filter(([, value]) => Boolean(value))
    .forEach(([label, value]) => list.append(buildPathItem(label, value)));
}

function syncMotionButton() {
  const button = qs("toggle-motion");
  button.textContent = state.motionPaused ? "Play motion" : "Pause motion";
}

function toggleComparisonMotion() {
  state.motionPaused = !state.motionPaused;
  document.querySelectorAll(".autoplay-video").forEach((video) => {
    if (state.motionPaused) {
      video.pause();
    } else {
      video.play().catch(() => {});
    }
  });
  syncMotionButton();
}

async function loadData() {
  const response = await fetch("assets/data/site-data.json");
  if (!response.ok) {
    throw new Error(`Failed to load site data: ${response.status}`);
  }
  return response.json();
}

function showLoadError(error) {
  const heroStrip = qs("hero-strip");
  heroStrip.innerHTML = "";
  heroStrip.append(makeStripChip("Error", "site-data.json could not be loaded"));
  console.error(error);
}

async function init() {
  qs("toggle-motion").addEventListener("click", toggleComparisonMotion);
  syncMotionButton();

  try {
    const data = await loadData();
    state.data = data;
    populateHero(data);
    populateComparison(data);
    populateTaichi(data);
    populateMetrics(data);
    populatePaths(data);
  } catch (error) {
    showLoadError(error);
  }
}

init();

// Small runtime: hydrate hero chips from metadata.json, toggle motion on all
// autoplay videos, and stamp the footer date.

const METADATA_URL = "assets/metadata.json";

function formatChamfer(v) {
  if (v == null) return "—";
  return v.toExponential(2).replace("e", "·10^");
}

function formatHausdorff(v) {
  if (v == null) return "—";
  return v.toFixed(3);
}

async function hydrateChips() {
  try {
    const response = await fetch(METADATA_URL, { cache: "no-cache" });
    if (!response.ok) throw new Error(`metadata ${response.status}`);
    const meta = await response.json();

    const chamfer = meta.evaluation?.average_chamfer;
    const hausdorff = meta.evaluation?.average_hausdorff;
    const params = meta.model?.parameters;
    const ckptMb = meta.model?.checkpoint_size_mb;

    const chamferEl = document.getElementById("chip-chamfer");
    const hausdorffEl = document.getElementById("chip-hausdorff");
    const paramsEl = document.getElementById("chip-params");
    const ckptEl = document.getElementById("chip-ckpt");

    if (chamferEl && chamfer != null) {
      chamferEl.textContent = chamfer.toExponential(2);
    }
    if (hausdorffEl && hausdorff != null) {
      hausdorffEl.textContent = hausdorff.toFixed(3);
    }
    if (paramsEl && params != null) {
      paramsEl.textContent = params.toLocaleString();
    }
    if (ckptEl && ckptMb != null) {
      ckptEl.textContent = `${ckptMb.toFixed(1)} MB`;
    }
  } catch (err) {
    console.warn("Could not hydrate hero chips:", err);
    // Leave the placeholder text; site still renders.
  }
}

function wireMotionToggle() {
  const button = document.getElementById("toggle-motion");
  if (!button) return;

  const videos = Array.from(document.querySelectorAll("video"));
  let paused = false;

  const apply = () => {
    for (const v of videos) {
      if (paused) {
        v.pause();
      } else {
        // `play` returns a promise; swallow rejection for muted autoplay quirks
        v.play().catch(() => {});
      }
    }
    button.textContent = paused ? "Resume motion" : "Pause motion";
  };

  button.addEventListener("click", () => {
    paused = !paused;
    apply();
  });
}

function stampFooter() {
  const el = document.getElementById("footer-date");
  if (!el) return;
  const now = new Date();
  el.textContent = now.toLocaleDateString(undefined, {
    year: "numeric",
    month: "short",
    day: "numeric",
  });
}

function init() {
  hydrateChips();
  wireMotionToggle();
  stampFooter();
}

if (document.readyState === "loading") {
  document.addEventListener("DOMContentLoaded", init, { once: true });
} else {
  init();
}

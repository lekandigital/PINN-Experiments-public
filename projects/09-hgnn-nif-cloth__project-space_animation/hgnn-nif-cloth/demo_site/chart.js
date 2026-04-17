(function () {
  var canvas = document.getElementById("rmse-chart");
  if (!canvas) return;
  var ctx = canvas.getContext("2d");
  var W = canvas.width, H = canvas.height;
  var PAD_L = 64, PAD_R = 20, PAD_T = 20, PAD_B = 48;
  var pw = W - PAD_L - PAD_R, ph = H - PAD_T - PAD_B;

  fetch("assets/rollout_curve.json")
    .then(function (r) { return r.json(); })
    .then(draw);

  function draw(data) {
    var rmse = data.per_frame_rmse;
    var thresh = data.rmse_threshold;
    var trust = data.trustworthy_frame;
    var N = rmse.length;
    var maxY = Math.min(Math.max.apply(null, rmse) * 1.08, 8);

    function x(i) { return PAD_L + (i / (N - 1)) * pw; }
    function y(v) { return PAD_T + ph - (v / maxY) * ph; }

    // background
    ctx.fillStyle = "#10101a";
    ctx.fillRect(0, 0, W, H);

    // grid lines
    ctx.strokeStyle = "#1e1e30";
    ctx.lineWidth = 1;
    for (var v = 0; v <= maxY; v += 1) {
      ctx.beginPath(); ctx.moveTo(PAD_L, y(v)); ctx.lineTo(W - PAD_R, y(v)); ctx.stroke();
      ctx.fillStyle = "#505068"; ctx.font = "11px monospace";
      ctx.fillText(v.toFixed(0), PAD_L - 28, y(v) + 4);
    }
    for (var i = 0; i <= N; i += 100) {
      ctx.beginPath(); ctx.moveTo(x(i), PAD_T); ctx.lineTo(x(i), H - PAD_B); ctx.stroke();
      ctx.fillStyle = "#505068"; ctx.font = "11px monospace";
      ctx.fillText(i.toString(), x(i) - 8, H - PAD_B + 18);
    }

    // threshold line
    ctx.strokeStyle = "#b04040"; ctx.lineWidth = 1;
    ctx.setLineDash([6, 4]);
    ctx.beginPath(); ctx.moveTo(PAD_L, y(thresh)); ctx.lineTo(W - PAD_R, y(thresh)); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#b04040"; ctx.font = "11px monospace";
    ctx.fillText("threshold = " + thresh, PAD_L + 8, y(thresh) - 6);

    // trust vertical
    ctx.strokeStyle = "#4ea8de"; ctx.lineWidth = 1;
    ctx.setLineDash([4, 4]);
    ctx.beginPath(); ctx.moveTo(x(trust), PAD_T); ctx.lineTo(x(trust), H - PAD_B); ctx.stroke();
    ctx.setLineDash([]);
    ctx.fillStyle = "#4ea8de"; ctx.font = "11px monospace";
    ctx.fillText("trust = " + trust, x(trust) + 4, PAD_T + 16);

    // RMSE curve
    ctx.strokeStyle = "#e8a852"; ctx.lineWidth = 2;
    ctx.beginPath();
    for (var i = 0; i < N; i++) {
      var px = x(i), py = y(rmse[i]);
      if (i === 0) ctx.moveTo(px, py); else ctx.lineTo(px, py);
    }
    ctx.stroke();

    // axes labels
    ctx.fillStyle = "#7080a0"; ctx.font = "12px sans-serif";
    ctx.fillText("frame", W / 2 - 16, H - 4);
    ctx.save(); ctx.translate(14, H / 2); ctx.rotate(-Math.PI / 2);
    ctx.fillText("RMSE (world units)", -50, 0);
    ctx.restore();
  }
})();

// Calculator: the same formulas as src/capplan/capacity.py, run in the browser
// on the measurements in data.js (written by `capplan report`).
(function () {
  "use strict";
  const D = window.CAPPLAN_DATA;
  const HOURS_PER_MONTH = 730;
  const $ = (id) => document.getElementById(id);
  const prices = {}; // user-edited $/GPU-hour, by GPU name

  // ---- model (mirrors capacity.py) ----
  function passes(r, slo) {
    return r.ttft_p95_ms != null && r.itl_p95_ms != null &&
      r.ttft_p95_ms <= slo.ttft && r.itl_p95_ms <= 1000 / slo.toks &&
      (r.error_rate || 0) <= 0.01;
  }
  function crossing(c0, v0, c1, v1, lim) {
    return v1 === v0 ? c1 : c0 + (lim - v0) * (c1 - c0) / (v1 - v0);
  }
  function maxConcurrency(rows, slo) {
    const pts = rows.slice().sort((a, b) => a.concurrency - b.concurrency);
    if (!pts.length || !passes(pts[0], slo)) return 0;
    for (let i = 1; i < pts.length; i++) {
      const lo = pts[i - 1], hi = pts[i];
      if (passes(hi, slo)) continue;
      if ((hi.error_rate || 0) > 0.01) return lo.concurrency;
      const lims = [hi.concurrency];
      if (hi.ttft_p95_ms > slo.ttft) lims.push(crossing(lo.concurrency, lo.ttft_p95_ms, hi.concurrency, hi.ttft_p95_ms, slo.ttft));
      const itlLim = 1000 / slo.toks;
      if (hi.itl_p95_ms > itlLim) lims.push(crossing(lo.concurrency, lo.itl_p95_ms, hi.concurrency, hi.itl_p95_ms, itlLim));
      return Math.max(lo.concurrency, Math.min(...lims));
    }
    return pts[pts.length - 1].concurrency;
  }
  function interpE2E(rows, c) {
    const pts = rows.filter((r) => r.e2e_p50_s != null).map((r) => [r.concurrency, r.e2e_p50_s]).sort((a, b) => a[0] - b[0]);
    if (c <= pts[0][0]) return pts[0][1];
    for (let i = 1; i < pts.length; i++) {
      const [c0, v0] = pts[i - 1], [c1, v1] = pts[i];
      if (c >= c0 && c <= c1) return v0 + (v1 - v0) * (c - c0) / (c1 - c0);
    }
    return pts[pts.length - 1][1];
  }
  function vramConcurrency(gpu, wl) {
    const free = gpu.vram_gb * 0.9 - wl.weights_gb - 1.0;
    return free <= 0 ? 0 : Math.floor(free * 1e9 / (wl.kv_bytes_token * wl.context_length));
  }
  function estimateGpu(gpu, wl, slo) {
    const r = gpu.bandwidth_gbs / D.measured_gpu.bandwidth_gbs;
    const limits = { vram: vramConcurrency(gpu, wl), cap: 512 };
    const pairs = [["ttft_p95_ms", slo.ttft], ["itl_p95_ms", 1000 / slo.toks]];
    for (const [key, lim] of pairs) {
      const [a, b] = wl.fits[key];
      if (a / r > lim) limits[key] = 0;
      else if (b <= 0) limits[key] = 512;
      else limits[key] = (lim * r - a) / b;
    }
    let bound = "vram";
    for (const k of Object.keys(limits)) if (limits[k] < limits[bound]) bound = k;
    const c = Math.max(0, Math.floor(limits[bound]));
    const [a, b] = wl.fits.e2e_p50_s;
    return { c, bound, e2e: (a + b * Math.max(c, 1)) / r };
  }
  function plan(users, c, e2e, think, price, headroom) {
    const perGpu = Math.floor(c) * ((e2e + think) / e2e) * (1 - headroom);
    if (perGpu <= 0) return { perGpu: 0, gpus: null, cost: null };
    const gpus = Math.ceil(users / perGpu);
    return { perGpu, gpus, cost: price != null ? gpus * price * HOURS_PER_MONTH : null };
  }

  // ---- UI ----
  const BOUND_LABEL = {
    measured: "measured",
    "not reached": "SLO met at every level tested",
    ttft_p95_ms: "time to first token",
    itl_p95_ms: "streaming speed",
    vram: "GPU memory",
    cap: "model cap (512)",
  };
  const fmtInt = (v) => v == null ? "–" : Math.round(v).toLocaleString("en-US");
  const fmtMoney = (v) => v == null ? "–" : "$" + Math.round(v).toLocaleString("en-US");

  function readInputs() {
    const num = (id, fallback) => {
      const v = parseFloat($(id).value);
      return Number.isFinite(v) ? v : fallback;
    };
    const d = D.defaults;
    return {
      wl: D.workloads[$("workload").value],
      users: Math.max(1, num("users", d.users)),
      think: Math.max(0, num("think", d.think_s)),
      slo: { ttft: Math.max(1, num("ttft", d.ttft_ms)), toks: Math.max(0.1, num("toks", d.tok_s)) },
      headroom: Math.min(0.9, Math.max(0, num("headroom", d.headroom * 100) / 100)),
    };
  }

  function computeRows(inp) {
    const { wl, slo } = inp;
    const out = [];
    const cMeas = maxConcurrency(wl.rows, slo);
    const e2eMeas = interpE2E(wl.rows, Math.max(cMeas, 1));
    const top = Math.max(...wl.rows.map((r) => r.concurrency));
    out.push({ name: D.measured_gpu.name, measured: true, c: Math.floor(cMeas), bound: cMeas >= top ? "not reached" : "measured", price: null,
      ...plan(inp.users, cMeas, e2eMeas, inp.think, null, inp.headroom) });
    for (const g of D.gpus) {
      const est = estimateGpu(g, wl, slo);
      const price = g.name in prices ? prices[g.name] : g.price_per_hour;
      out.push({ name: g.name, measured: false, c: est.c, bound: est.bound, price, source: g.source, instance: g.instance,
        ...plan(inp.users, est.c, est.e2e, inp.think, price, inp.headroom) });
    }
    return out;
  }

  function render() {
    const inp = readInputs();
    const rows = computeRows(inp);
    const priced = rows.filter((r) => r.cost != null);
    const best = priced.length ? priced.reduce((a, b) => (b.cost < a.cost ? b : a)) : null;

    const tbody = $("gpus").querySelector("tbody");
    tbody.textContent = "";
    for (const r of rows) {
      const tr = document.createElement("tr");
      if (best && r === best) tr.className = "best";
      const nameTd = document.createElement("td");
      nameTd.textContent = r.name;
      const tag = document.createElement("span");
      tag.className = "tag" + (best && r === best ? " tag-best" : "");
      tag.textContent = r.measured ? "measured" : (best && r === best ? "estimate · lowest cost" : "estimate");
      nameTd.appendChild(tag);
      tr.appendChild(nameTd);
      const cell = (text, cls) => {
        const td = document.createElement("td");
        if (cls) td.className = cls;
        td.textContent = text;
        tr.appendChild(td);
        return td;
      };
      cell((r.bound === "not reached" ? "≥ " : "") + fmtInt(r.c), "num");
      cell(BOUND_LABEL[r.bound] || r.bound);
      cell(fmtInt(r.perGpu), "num");
      cell(r.gpus == null ? "–" : fmtInt(r.gpus), "num");
      const priceTd = cell("", "num");
      if (r.measured) {
        priceTd.textContent = "–";
      } else {
        const input = document.createElement("input");
        input.type = "number";
        input.min = "0";
        input.step = "0.01";
        input.value = r.price.toFixed(2);
        input.setAttribute("aria-label", `Price per GPU-hour for ${r.name}`);
        input.addEventListener("change", () => {
          const v = parseFloat(input.value);
          if (Number.isFinite(v) && v >= 0) prices[r.name] = v;
          render();
        });
        priceTd.appendChild(input);
      }
      cell(fmtMoney(r.cost), "num");
      tbody.appendChild(tr);
    }

    if (best) {
      $("headline").textContent = `${fmtInt(inp.users)} active users → ${best.gpus} × ${best.name}, about ${fmtMoney(best.cost)} a month`;
      $("subline").textContent = `Each ${best.name} holds about ${fmtInt(best.c)} requests in flight within the SLO, which serves about ${fmtInt(best.perGpu)} users after ${Math.round(inp.headroom * 100)}% headroom. Cloud figures are bandwidth-scaled estimates.`;
    } else {
      $("headline").textContent = "No GPU in the list meets this SLO";
      $("subline").textContent = "Loosen the time-to-first-token or streaming-speed limit.";
    }

    const mt = $("measured").querySelector("tbody");
    mt.textContent = "";
    for (const r of inp.wl.rows) {
      const tr = document.createElement("tr");
      const vals = [r.concurrency, fmtInt(r.output_tok_per_s), fmtInt(r.ttft_p50_ms), fmtInt(r.ttft_p95_ms),
        r.itl_p95_ms == null ? "–" : r.itl_p95_ms.toFixed(1), r.e2e_p50_s == null ? "–" : r.e2e_p50_s.toFixed(1),
        ((r.error_rate || 0) * 100).toFixed(0) + "%"];
      for (const v of vals) {
        const td = document.createElement("td");
        td.className = "num";
        td.textContent = v;
        tr.appendChild(td);
      }
      mt.appendChild(tr);
    }
    $("measured-note").textContent = `${inp.wl.label}. Model ${inp.wl.model}, Ollama with ${inp.wl.slots_note} (${inp.wl.context_length} tokens each), ${inp.wl.kv_cache_type} KV cache, on an ${D.measured_gpu.name} (${D.measured_gpu.vram_gb} GB, ${D.measured_gpu.bandwidth_gbs} GB/s).`;
  }

  function init() {
    const sel = $("workload");
    for (const [key, wl] of Object.entries(D.workloads)) {
      const o = document.createElement("option");
      o.value = key;
      o.textContent = wl.label;
      sel.appendChild(o);
    }
    const d = D.defaults;
    $("users").value = d.users;
    $("think").value = d.think_s;
    $("ttft").value = d.ttft_ms;
    $("toks").value = d.tok_s;
    $("headroom").value = Math.round(d.headroom * 100);
    for (const id of ["workload", "users", "think", "ttft", "toks", "headroom"]) {
      $(id).addEventListener("input", render);
    }
    $("price-note").textContent = `Prices: ${D.price_note}, read ${D.price_date}. Edit them in the table. Per-GPU price includes the CPU and RAM of the instance.`;
    render();
  }

  if (!D || !D.workloads || !Object.keys(D.workloads).length) {
    $("headline").textContent = "No results yet: run `capplan report` to write data.js.";
    return;
  }
  init();
})();

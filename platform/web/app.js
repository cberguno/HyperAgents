/* HyperAgents Control Platform — M1 dashboard (vanilla JS foundation). */
"use strict";

const API = "/api";
const state = { slots: [], experiments: {}, selected: null, sse: null, shellHistory: [], models: [] };

const $ = (sel) => document.querySelector(sel);
const fmt = {
  money: (usd) => (usd == null ? "—" : `$${Number(usd).toFixed(usd < 1 ? 4 : 2)}`),
  tokens: (t) => (t == null ? "—" : t >= 1e6 ? (t / 1e6).toFixed(1) + "M" : t >= 1e3 ? (t / 1e3).toFixed(1) + "k" : String(t)),
  time: (ts) => (ts ? new Date(ts * 1000).toLocaleTimeString() : "—"),
};

async function jfetch(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) {
    let detail = r.statusText;
    try { detail = (await r.json()).detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return r.json();
}

/* ---------------- slots grid ---------------- */

async function refreshSlots() {
  const data = await jfetch(`${API}/slots`);
  state.slots = data.slots;
  $("#slots-count").textContent = `${state.slots.length} slots`;
  renderSlots();
}

function renderSlots() {
  const grid = $("#slots");
  grid.innerHTML = "";
  const makeSlotCard = (el) => {
    el.tabIndex = 0;
    el.setAttribute("role", "button");
    el.addEventListener("keydown", (ev) => {
      if (ev.key === "Enter" || ev.key === " ") el.click();
    });
    grid.appendChild(el);
  };
  for (const slot of state.slots) {
    const el = document.createElement("div");
    if (slot.experiment) {
      const e = slot.experiment;
      el.className = "slot";
      el.setAttribute("aria-label", `Open experiment ${e.name} (${e.id})`);
      el.innerHTML = `
        <h4>${escapeHtml(e.name)}</h4>
        <div class="meta">slot ${slot.position} · <span class="status-chip st-${e.status}">${e.status}</span> · gen ${e.current_gen ?? "—"}/${e.max_generation}</div>
        <div class="stats">
          <div>tokens<b>${fmt.tokens(e.total_tokens)}</b></div>
          <div>cost<b>${fmt.money(e.cost_usd)}</b></div>
          <div>meta<b class="subtle small">${escapeHtml(shortModel(e.meta_model))}</b></div>
        </div>`;
      el.onclick = () => openDetail(e.id);
    } else {
      el.className = "slot empty";
      el.textContent = `Empty slot ${slot.position}`;
      el.setAttribute("aria-label", `Create experiment in slot ${slot.position}`);
      el.onclick = () => openCreate(slot.position);
    }
    makeSlotCard(el);
  }
  const plus = document.createElement("div");
  plus.className = "slot empty";
  plus.textContent = "+ New slot";
  plus.setAttribute("aria-label", "Add a new experiment slot");
  plus.onclick = async () => { await jfetch(`${API}/slots/expand`, { method: "POST" }); refreshSlots(); };
  makeSlotCard(plus);
}

function shortModel(m) { return m ? m.split("/").pop().slice(0, 18) : "—"; }
function escapeHtml(s) { const d = document.createElement("span"); d.textContent = s ?? ""; return d.innerHTML; }

/* ---------------- create dialog ---------------- */

async function openCreate(slotPos) {
  $("#create-slot-label").textContent = slotPos != null ? `(slot ${slotPos})` : "(new slot)";
  await Promise.all([fillModelSelect("#sel-meta", "anthropic"), fillModelSelect("#sel-task", "openai")]);
  const dlg = $("#create-dialog");
  dlg.dataset.slot = slotPos ?? "";
  dlg.showModal();
}

async function fillModelSelect(sel, preferProvider) {
  const selEl = $(sel);
  if (!state.models.length) {
    const data = await jfetch(`${API}/models`);
    state.models = data.models;
  }
  selEl.innerHTML = state.models
    .filter((m) => m.provider !== "stub")
    .map((m) => `<option value="${m.id}">${m.label} — $${m.input_usd_per_mtok}/${m.output_usd_per_mtok} per Mtok</option>`)
    .join("");
  const preferred = state.models.find((m) => m.provider === preferProvider);
  if (preferred) selEl.value = preferred.id;
}

$("#create-form").addEventListener("submit", async (ev) => {
  ev.preventDefault();
  const f = new FormData(ev.target);
  const cfg = {
    name: f.get("name"),
    domains: ["paper_review"],
    meta_model: f.get("meta_model"),
    task_model: f.get("task_model"),
    max_generation: Number(f.get("max_generation")),
    parent_selection: f.get("parent_selection"),
    loop: f.get("loop"),
    budget: {
      max_cost_usd: numOrNull(f.get("max_cost_usd")),
      max_tokens: numOrNull(f.get("max_tokens")),
      max_wallclock_hours: numOrNull(f.get("max_wallclock_hours")),
    },
  };
  const slot = $("#create-dialog").dataset.slot;
  const body = { ...cfg };
  if (slot !== "") body.slot = Number(slot);
  try {
    await jfetch(`${API}/experiments`, { method: "POST", body: JSON.stringify(body) });
    $("#create-dialog").close();
    refreshSlots();
  } catch (e) { alert(`Could not create experiment: ${e.message}`); }
});

function numOrNull(v) { return v === "" || v == null ? null : Number(v); }

/* ---------------- detail view ---------------- */

function openDetail(expId) {
  state.selected = expId;
  show("#detail-panel");
  hide("#slots-panel"); hide("#compare-panel");
  $("#btn-back").onclick = () => { closeDetail(); };
  refreshDetail();
  openSSE(expId);
}

function closeDetail() {
  if (state.sse) { state.sse.close(); state.sse = null; }
  state.selected = null;
  hide("#detail-panel"); show("#slots-panel"); show("#compare-panel");
  refreshSlots();
}

function show(sel) { $(sel).classList.remove("hidden"); }
function hide(sel) { $(sel).classList.add("hidden"); }

async function refreshDetail() {
  if (!state.selected) return;
  const e = await jfetch(`${API}/experiments/${state.selected}`);
  $("#detail-title").textContent = `${e.name} · ${e.id}`;
  renderControls(e);
  renderMetrics(e);
}

function renderControls(e) {
  const c = $("#detail-controls");
  c.innerHTML = "";
  const mk = (label, path, cls, enabled, title) => {
    const b = document.createElement("button");
    b.className = "btn " + (cls || "");
    b.textContent = label;
    b.disabled = !enabled;
    if (title) b.title = title;
    b.onclick = async () => {
      try { await jfetch(`${API}/experiments/${e.id}${path}`, { method: "POST" }); setTimeout(refreshDetail, 600); }
      catch (err) { alert(`${label} failed: ${err.message}`); }
    };
    c.appendChild(b);
  };
  const idle = ["created", "stopped", "paused", "paused_budget", "failed"].includes(e.status);
  mk("Start", "/start", "primary", idle, "Launch the (unmodified) upstream generate_loop.py");
  if (e.workspace_status && e.workspace_status.initial_evals_ready === false && idle) {
    mk("Seed initial evals", "/seed", "", e.config.loop !== "stub",
      "Runs upstream setup steps (real LLM calls — needs approval)");
  }
  mk("Pause", "/pause", "", e.status === "running", "SIGINT at the next safe boundary; resume with --resume_from");
  mk("Stop", "/stop", "danger", e.status === "running", "Graceful stop; history preserved");
  mk("Resume", "/resume", "", ["paused", "paused_budget", "stopped"].includes(e.status),
    "Continues from the last completed generation (upstream --resume_from)");
}

function renderMetrics(e) {
  const u = e.usage || {};
  const b = e.budget_state || {};
  $("#metrics").innerHTML = `
    <div class="metric"><div class="label">Status</div><div class="value"><span class="status-chip st-${e.status}">${e.status}</span></div></div>
    <div class="metric"><div class="label">Current generation</div><div class="value">${e.current_gen ?? "—"}</div></div>
    <div class="metric"><div class="label">Tokens (proxy)</div><div class="value">${fmt.tokens(u.total_tokens)}</div></div>
    <div class="metric"><div class="label">Cost (est.)</div><div class="value">${fmt.money(u.cost_usd)}</div></div>
    <div class="metric"><div class="label">Requests</div><div class="value">${u.requests ?? 0}</div></div>
    <div class="metric"><div class="label">Budget</div><div class="value">${b.exceeded ? "EXCEEDED" : (JSON.stringify(b.limits || {}).length > 2 ? "within limits" : "unlimited")}</div></div>`;
}

function openSSE(expId) {
  if (state.sse) state.sse.close();
  const log = $("#live-log");
  log.textContent = "";
  let gensRendered = "";
  state.sse = new EventSource(`${API}/experiments/${expId}/events`);
  state.sse.onmessage = (ev) => {
    let m; try { m = JSON.parse(ev.data); } catch (_) { return; }
    if (m.type === "log") {
      log.textContent += `[${fmt.time(m.ts)}] ${m.data.line}\n`;
      log.scrollTop = log.scrollHeight;
    } else if (m.type === "generations") {
      gensRendered = renderGenerations(m.data.generations);
    } else if (["started", "stopping", "stopped", "paused", "paused_budget", "finished", "failed",
               "budget_exceeded", "budget_blocked", "workspace_prepared", "resuming", "seeding_started"].includes(m.type)) {
      log.textContent += `== ${m.type} ${JSON.stringify(m.data)}\n`;
      log.scrollTop = log.scrollHeight;
      refreshDetail();
    }
  };
}

function renderGenerations(gens) {
  const el = $("#generations");
  el.innerHTML = `<table><thead><tr><th>gen</th><th>parent</th><th>paper_review train</th><th>val</th></tr></thead><tbody>${
    gens.map((g) => {
      const s = (g.scores || {}).paper_review || {};
      return `<tr><td>${g.gen}</td><td>${g.parent ?? "—"}</td><td>${fmt.score(s.train)}</td><td>${fmt.score(s.val)}</td></tr>`;
    }).join("")
  }</tbody></table>`;
}

fmt.score = (v) => (v == null ? "—" : (100 * v).toFixed(1) + "%");

/* ---------------- comparison ---------------- */

async function refreshCompareSelects() {
  const data = await jfetch(`${API}/experiments`);
  const opts = data.experiments.map((e) => `<option value="${e.id}">${escapeHtml(e.name)} (${e.id})</option>`).join("");
  $("#cmp-a").innerHTML = opts; $("#cmp-b").innerHTML = opts;
}

$("#btn-compare").onclick = async () => {
  const a = $("#cmp-a").value, b = $("#cmp-b").value;
  if (!a || !b) return;
  const data = await jfetch(`${API}/comparison?ids=${a},${b}`);
  $("#compare-result").innerHTML = `<div class="card"><table>
    <thead><tr><th></th>${data.comparison.map((c) => `<th>${escapeHtml(c.name)}</th>`).join("")}</tr></thead>
    <tbody>
      ${row("Status", data.comparison, (c) => `<span class="status-chip st-${c.status}">${c.status}</span>`)}
      ${row("Domains", data.comparison, (c) => (c.domains || []).join(", "))}
      ${row("Meta model", data.comparison, (c) => escapeHtml(c.meta_model || ""))}
      ${row("Task model", data.comparison, (c) => escapeHtml(c.task_model || ""))}
      ${row("Tokens", data.comparison, (c) => fmt.tokens(c.usage?.total_tokens))}
      ${row("Cost", data.comparison, (c) => fmt.money(c.usage?.cost_usd))}
      ${row("Requests", data.comparison, (c) => c.usage?.requests ?? 0)}
      ${row("Wallclock", data.comparison, (c) => fmt.time(c.finished_at || Date.now() / 1000 - (c.started_at || Date.now() / 1000)) )}
    </tbody></table></div>`;
};

function row(label, items, cell) {
  return `<tr><td class="subtle">${label}</td>${items.map((c) => `<td>${cell(c)}</td>`).join("")}</tr>`;
}

/* ---------------- shell ---------------- */

async function sendShell(text) {
  const log = $("#shell-log");
  if (text) log.appendChild(msgEl(text, "user"));
  const input = $("#shell-input"); input.value = ""; input.disabled = true;
  try {
    const r = await jfetch(`${API}/shell/chat`, {
      method: "POST",
      body: JSON.stringify({ message: text, history: state.shellHistory }),
    });
    for (const tc of r.tool_calls || []) {
      log.appendChild(msgEl(`→ ${tc.tool_name}(${JSON.stringify(tc.tool_input)})`, "tool"));
    }
    log.appendChild(msgEl(r.reply, "assistant"));
    state.shellHistory = r.history || state.shellHistory;
  } catch (e) {
    log.appendChild(msgEl(`shell error: ${e.message}`, "tool"));
  }
  input.disabled = false; input.focus();
  log.scrollTop = log.scrollHeight;
}

function msgEl(text, cls) {
  const d = document.createElement("div");
  d.className = "msg " + cls;
  d.textContent = text;
  return d;
}

$("#btn-shell").onclick = () => { const v = $("#shell-input").value.trim(); if (v) sendShell(v); };
$("#shell-input").addEventListener("keydown", (ev) => { if (ev.key === "Enter") { const v = ev.target.value.trim(); if (v) sendShell(v); } });

/* ---------------- boot ---------------- */

(async function boot() {
  try {
    const h = await jfetch(`${API}/health`);
    $("#upstream-badge").textContent = `upstream: ${h.upstream}`;
    $("#upstream-badge").style.color = h.upstream === "stub" ? "var(--warn)" : "var(--accent2)";
    $("#shell-model").textContent = "shell model: configured via HACP_SHELL_MODEL";
    await refreshSlots();
    await refreshCompareSelects();
  } catch (e) {
    $("#upstream-badge").textContent = `API error: ${e.message}`;
  }
})();

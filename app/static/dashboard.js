const $ = (selector) => document.querySelector(selector);
const state = { armed: false, zones: [], drawing: false, draft: [] };

async function api(path, options = {}) {
  const response = await fetch(path, { headers: { "Content-Type": "application/json" }, ...options });
  if (!response.ok) throw new Error((await response.text()) || `Request failed: ${response.status}`);
  return response.json();
}

function showToast(message) {
  $("#toast").textContent = message;
  window.setTimeout(() => { if ($("#toast").textContent === message) $("#toast").textContent = ""; }, 4000);
}

function updateArmButton() {
  const button = $("#armButton");
  button.textContent = state.armed ? "Disarm system" : "Arm system";
  button.classList.toggle("armed", state.armed);
}

async function refreshStatus() {
  try {
    const status = await api("/api/status");
    state.armed = status.armed;
    updateArmButton();
    const badge = $("#connectionBadge");
    badge.textContent = status.state === "running" ? "Camera online" : status.state.replaceAll("-", " ");
    badge.className = `badge ${status.state === "running" ? "ok" : status.state.includes("error") ? "error" : "neutral"}`;
    $("#cameraMessage").style.display = status.state === "running" ? "none" : "grid";
    $("#cameraMessage").textContent = status.error || "Waiting for camera…";
    $("#fps").textContent = status.fps || "--";
    $("#smsMode").textContent = status.sms_mode;
    if (status.assessment) {
      $("#riskScore").textContent = `${status.assessment.score}/100`;
      $("#riskSummary").textContent = status.assessment.summary;
    }
  } catch { $("#connectionBadge").textContent = "Dashboard offline"; }
}

async function refreshEvents() {
  const [{ events }, stats] = await Promise.all([api("/api/events?limit=20"), api("/api/stats")]);
  $("#eventsToday").textContent = stats.today;
  $("#highRisk").textContent = `${stats.high_risk} high risk total`;
  const list = $("#eventList");
  if (!events.length) { list.innerHTML = '<p class="empty-state">No events recorded yet.</p>'; return; }
  list.innerHTML = events.map((event) => `
    <article class="event-row">
      ${event.snapshot ? `<a href="${event.snapshot}" target="_blank"><img src="${event.snapshot}" alt="Event snapshot"></a>` : "<span></span>"}
      <div><strong>${escapeHtml(event.summary)}</strong><small>${new Date(event.created_at).toLocaleString()}</small></div>
      <span class="risk-pill ${event.level}">${event.level} · ${event.score}</span>
      <small>SMS: ${escapeHtml(event.sms_status)}</small>
    </article>`).join("");
}

function escapeHtml(value) {
  const node = document.createElement("div");
  node.textContent = String(value);
  return node.innerHTML;
}

async function loadRules() {
  const rules = await api("/api/rules");
  $("#scheduleEnabled").checked = rules.schedule_enabled;
  $("#activeFrom").value = rules.active_from;
  $("#activeUntil").value = rules.active_until;
  $("#watchLabels").value = rules.watch_labels.join(", ");
}

async function loadZones() {
  state.zones = (await api("/api/zones")).zones;
  renderZones();
}

function renderZones() {
  $("#zoneList").innerHTML = state.zones.length ? state.zones.map((zone, index) => `
    <div class="zone-chip"><span>${escapeHtml(zone.name)} · ${zone.points.length} points</span><button data-zone="${index}">Remove</button></div>`).join("") : '<p class="hint">No zones configured.</p>';
  $("#zoneList").querySelectorAll("button[data-zone]").forEach((button) => button.addEventListener("click", async () => {
    state.zones.splice(Number(button.dataset.zone), 1);
    await saveZones();
  }));
  drawCanvas();
}

async function saveZones() {
  await api("/api/zones", { method: "PUT", body: JSON.stringify({ zones: state.zones }) });
  renderZones();
  showToast("Zones saved");
}

function canvasPoint(event) {
  const canvas = $("#zoneCanvas");
  const rect = canvas.getBoundingClientRect();
  return [(event.clientX - rect.left) / rect.width, (event.clientY - rect.top) / rect.height];
}

function drawCanvas() {
  const canvas = $("#zoneCanvas");
  const rect = canvas.getBoundingClientRect();
  canvas.width = Math.max(1, Math.round(rect.width * devicePixelRatio));
  canvas.height = Math.max(1, Math.round(rect.height * devicePixelRatio));
  const context = canvas.getContext("2d");
  context.scale(devicePixelRatio, devicePixelRatio);
  const polygons = [...state.zones.map((zone) => zone.points), state.draft];
  polygons.forEach((points, index) => {
    if (!points.length) return;
    context.beginPath();
    points.forEach(([x, y], pointIndex) => pointIndex ? context.lineTo(x * rect.width, y * rect.height) : context.moveTo(x * rect.width, y * rect.height));
    if (index < state.zones.length) context.closePath();
    context.strokeStyle = index < state.zones.length ? "#59d7e8" : "#f0bf64";
    context.lineWidth = 2;
    context.stroke();
    points.forEach(([x, y]) => { context.beginPath(); context.arc(x * rect.width, y * rect.height, 4, 0, Math.PI * 2); context.fillStyle = context.strokeStyle; context.fill(); });
  });
}

$("#armButton").addEventListener("click", async () => {
  const result = await api("/api/monitor/arm", { method: "POST", body: JSON.stringify({ armed: !state.armed }) });
  state.armed = result.armed;
  updateArmButton();
});
$("#rulesForm").addEventListener("submit", async (event) => {
  event.preventDefault();
  await api("/api/rules", { method: "PUT", body: JSON.stringify({ schedule_enabled: $("#scheduleEnabled").checked, active_from: $("#activeFrom").value, active_until: $("#activeUntil").value, watch_labels: $("#watchLabels").value.split(",").map((value) => value.trim()).filter(Boolean) }) });
  showToast("Detection rules saved");
});
$("#drawZoneButton").addEventListener("click", () => {
  state.drawing = !state.drawing;
  state.draft = [];
  $("#cameraWrap").classList.toggle("drawing", state.drawing);
  $("#drawZoneButton").textContent = state.drawing ? "Cancel drawing" : "Draw zone";
  drawCanvas();
});
$("#zoneCanvas").addEventListener("click", (event) => { if (state.drawing) { state.draft.push(canvasPoint(event)); drawCanvas(); } });
$("#zoneCanvas").addEventListener("dblclick", async (event) => {
  event.preventDefault();
  if (!state.drawing || state.draft.length < 3) return showToast("Add at least three points");
  const name = window.prompt("Zone name", `Zone ${state.zones.length + 1}`);
  if (!name) return;
  state.zones.push({ name, points: state.draft, enabled: true });
  state.draft = []; state.drawing = false;
  $("#cameraWrap").classList.remove("drawing"); $("#drawZoneButton").textContent = "Draw zone";
  await saveZones();
});
$("#clearZones").addEventListener("click", async () => { state.zones = []; await saveZones(); });
$("#testSms").addEventListener("click", async () => { const result = await api("/api/alerts/test", { method: "POST" }); showToast(`SMS result: ${result.status}`); });
$("#refreshEvents").addEventListener("click", refreshEvents);
window.addEventListener("resize", drawCanvas);

Promise.all([loadRules(), loadZones(), refreshStatus(), refreshEvents()]).catch((error) => showToast(error.message));
window.setInterval(refreshStatus, 2000);
window.setInterval(refreshEvents, 7000);


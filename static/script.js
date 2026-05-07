/* ============================================================
   script.js — Stock Predictor Frontend Logic
   ============================================================ */

let priceChart = null;
let refreshTimer = null;
let acIndex = -1;
let acItems = [];

// ── DOM refs ──
const symbolInput  = document.getElementById('symbol-input');
const acList       = document.getElementById('autocomplete');
const predictBtn   = document.getElementById('predict-btn');
const loadingEl    = document.getElementById('loading');
const errorBox     = document.getElementById('error-box');
const resultsEl    = document.getElementById('results');
const refreshSel   = document.getElementById('refresh-select');

// ── Autocomplete ──
let acDebounce = null;
symbolInput.addEventListener('input', () => {
  clearTimeout(acDebounce);
  const q = symbolInput.value.trim();
  if (q.length < 1) { hideAc(); return; }
  acDebounce = setTimeout(() => fetchAc(q), 220);
});

async function fetchAc(q) {
  try {
    const res  = await fetch(`/api/search?q=${encodeURIComponent(q)}`);
    const data = await res.json();
    renderAc(data);
  } catch { hideAc(); }
}

function renderAc(items) {
  acItems = items;
  acIndex = -1;
  if (!items.length) { hideAc(); return; }
  acList.innerHTML = items.map((it, i) =>
    `<div class="ac-item" data-i="${i}" data-sym="${it.symbol}">
       <span class="ac-sym">${it.symbol}</span>
       <span class="ac-name">${it.name}</span>
     </div>`
  ).join('');
  acList.style.display = 'block';
  acList.querySelectorAll('.ac-item').forEach(el => {
    el.addEventListener('mousedown', e => {
      e.preventDefault();
      selectAc(parseInt(el.dataset.i));
    });
  });
}

function selectAc(i) {
  symbolInput.value = acItems[i].symbol;
  hideAc();
}

function hideAc() { acList.style.display = 'none'; acIndex = -1; }

symbolInput.addEventListener('keydown', e => {
  const visible = acList.style.display === 'block';
  if (e.key === 'ArrowDown')   { e.preventDefault(); if (visible) moveAc(1); }
  if (e.key === 'ArrowUp')     { e.preventDefault(); if (visible) moveAc(-1); }
  if (e.key === 'Enter')       { if (visible && acIndex >= 0) { e.preventDefault(); selectAc(acIndex); } else if (!visible) runPredict(); }
  if (e.key === 'Escape')      { hideAc(); }
});

function moveAc(dir) {
  const els = acList.querySelectorAll('.ac-item');
  if (!els.length) return;
  els[acIndex]?.classList.remove('selected');
  acIndex = (acIndex + dir + els.length) % els.length;
  els[acIndex].classList.add('selected');
  symbolInput.value = acItems[acIndex].symbol;
}

document.addEventListener('click', e => {
  if (!acList.contains(e.target) && e.target !== symbolInput) hideAc();
});

// ── Predict ──
predictBtn.addEventListener('click', runPredict);

async function runPredict() {
  const symbol = symbolInput.value.trim().toUpperCase();
  if (!symbol) { showError('Please enter a stock symbol.'); return; }
  hideAc();
  setLoading(true);
  hideError();
  resultsEl.style.display = 'none';

  try {
    const res  = await fetch('/api/predict', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ symbol }),
    });
    const data = await res.json();
    if (!res.ok || data.error) throw new Error(data.error || 'Prediction failed');
    renderResults(data);
  } catch (err) {
    showError(err.message);
  } finally {
    setLoading(false);
  }
}

// ── Render results (without accuracy) ──
function renderResults(d) {
  // KPIs
  set('kpi-current',   `$${fmt(d.current_price)}`);
  set('kpi-predicted', `$${fmt(d.predicted_price)}`);

  const chgEl = document.getElementById('kpi-change');
  const sign  = d.change_percent >= 0 ? '+' : '';
  chgEl.textContent = `${sign}${d.change_percent.toFixed(2)}%`;
  chgEl.className   = `kpi-value ${d.change_percent >= 0 ? 'change-positive' : 'change-negative'}`;

  // Suggestion badge
  const badge = document.getElementById('kpi-suggestion');
  badge.textContent = d.suggestion;
  badge.className   = `suggestion-badge ${d.suggestion}`;

  // Data source
  const pill = document.getElementById('data-source');
  pill.innerHTML = `<span class="source-dot"></span>${d.data_source_used}`;

  // Symbol
  set('result-symbol', d.symbol);

  // Chart
  renderChart(d);

  resultsEl.style.display = 'block';
}

function renderChart(d) {
  const ctx = document.getElementById('price-chart').getContext('2d');
  if (priceChart) priceChart.destroy();

  const labels = [...d.dates, 'Predicted'];
  const histData = d.historical_prices.map((v, i) => ({ x: d.dates[i], y: v }));
  const predData = new Array(d.historical_prices.length).fill(null);
  predData[predData.length - 1] = d.historical_prices[d.historical_prices.length - 1];

  const gradient = ctx.createLinearGradient(0, 0, 0, 320);
  gradient.addColorStop(0, 'rgba(0,212,255,0.25)');
  gradient.addColorStop(1, 'rgba(0,212,255,0)');

  priceChart = new Chart(ctx, {
    type: 'line',
    data: {
      labels,
      datasets: [
        {
          label: 'Historical Price',
          data: [...d.historical_prices, null],
          borderColor: '#00d4ff',
          backgroundColor: gradient,
          borderWidth: 2,
          pointRadius: 0,
          pointHoverRadius: 4,
          tension: 0.4,
          fill: true,
        },
        {
          label: 'Predicted Price',
          data: [...new Array(d.historical_prices.length - 1).fill(null),
                 d.historical_prices[d.historical_prices.length - 1],
                 d.predicted_price],
          borderColor: '#7c3aed',
          backgroundColor: 'rgba(124,58,237,0.2)',
          borderWidth: 2.5,
          borderDash: [6, 4],
          pointRadius: [0,0,0,0,0,0,0,8],
          pointHoverRadius: 10,
          pointBackgroundColor: '#7c3aed',
          pointBorderColor: '#fff',
          pointBorderWidth: 2,
          tension: 0,
          fill: false,
        }
      ]
    },
    options: {
      responsive: true,
      maintainAspectRatio: true,
      interaction: { intersect: false, mode: 'index' },
      plugins: {
        legend: { display: false },
        tooltip: {
          backgroundColor: '#121a28',
          borderColor: '#1e2d45',
          borderWidth: 1,
          titleColor: '#6b84a3',
          bodyColor: '#e2ecf8',
          titleFont: { family: 'DM Mono, monospace', size: 11 },
          bodyFont:  { family: 'DM Mono, monospace', size: 13 },
          padding: 12,
          callbacks: {
            label: ctx => ctx.raw != null ? ` $${fmt(ctx.raw)}` : null,
          }
        }
      },
      scales: {
        x: {
          grid: { color: 'rgba(30,45,69,0.6)', drawBorder: false },
          ticks: { color: '#6b84a3', font: { family: 'DM Mono, monospace', size: 10 }, maxTicksLimit: 8 },
        },
        y: {
          grid: { color: 'rgba(30,45,69,0.6)', drawBorder: false },
          ticks: { color: '#6b84a3', font: { family: 'DM Mono, monospace', size: 10 },
                   callback: v => `$${v >= 1000 ? (v/1000).toFixed(1)+'k' : v.toFixed(2)}` },
        }
      }
    }
  });
}

// ── Auto-refresh ──
refreshSel.addEventListener('change', setupRefresh);

function setupRefresh() {
  clearInterval(refreshTimer);
  const val = parseInt(refreshSel.value);
  if (!val) return;
  refreshTimer = setInterval(() => {
    const sym = symbolInput.value.trim().toUpperCase();
    if (sym) runPredict();
  }, val * 60 * 1000);
}

// ── Helpers ──
function set(id, text) { document.getElementById(id).textContent = text; }
function fmt(n) {
  if (n == null) return '—';
  if (n >= 1000) return n.toLocaleString('en-US', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return Number(n).toFixed(2);
}

function setLoading(on) {
  loadingEl.style.display  = on ? 'flex' : 'none';
  predictBtn.disabled      = on;
  predictBtn.innerHTML     = on
    ? `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><circle cx="12" cy="12" r="10"/></svg> Analyzing…`
    : `<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2.5"><polygon points="5 3 19 12 5 21 5 3"/></svg> Predict`;
}

function showError(msg) {
  errorBox.textContent    = `⚠️  ${msg}`;
  errorBox.style.display  = 'block';
}

function hideError() { errorBox.style.display = 'none'; }
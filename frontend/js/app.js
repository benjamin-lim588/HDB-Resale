import {getFilterOptions, getDashboardData} from './data.js';
import {updateCharts, emptyChart} from './charts.js';

const $ = id => document.getElementById(id);
const number = new Intl.NumberFormat('en-SG', {maximumFractionDigits: 0});
let options, state, timer, revision = 0;
const definitions = [['towns', 'Town'], ['flatTypes', 'Flat Type'], ['storeys', 'Storey Range']];

function status(message, error = false) {
  $('status').hidden = !message;
  $('status').textContent = message;
  $('status').classList.toggle('error', error);
}
function monthLabel(month) {
  return new Date(`${month}-01T00:00:00Z`).toLocaleDateString('en-SG', {month: 'long', year: 'numeric', timeZone: 'UTC'});
}
function defaultState() {
  return {towns: [...options.towns], flatTypes: [...options.flatTypes], storeys: [...options.storeys], years: [options.years[0], options.years.at(-1)]};
}
function syncControls() {
  for (const [key] of definitions) {
    const selected = new Set(state[key]);
    document.querySelectorAll(`input[data-filter="${key}"]`).forEach(input => { input.checked = selected.has(input.value); });
    $(`summary-${key}`).textContent = state[key].length === options[key].length ? `All ${key === 'flatTypes' ? 'flat types' : key === 'storeys' ? 'storeys' : 'towns'}` : `${state[key].length} selected`;
  }
  $('start-year').value = state.years[0];
  $('end-year').value = state.years[1];
  $('start-label').textContent = state.years[0];
  $('end-label').textContent = state.years[1];
}
function makeFilters() {
  for (const [key, label] of definitions) {
    const group = document.createElement('div');
    group.className = 'filter-group';
    const heading = document.createElement('span'); heading.className = 'filter-label'; heading.id = `label-${key}`; heading.textContent = label;
    const details = document.createElement('details');
    const summary = document.createElement('summary'); summary.id = `summary-${key}`; summary.setAttribute('aria-labelledby', `label-${key} summary-${key}`);
    const menu = document.createElement('div'); menu.className = 'filter-menu';
    const actions = document.createElement('div'); actions.className = 'selection-actions';
    for (const [text, values] of [['Select all', options[key]], ['Clear', []]]) {
      const button = document.createElement('button'); button.type = 'button'; button.textContent = text;
      button.addEventListener('click', () => { state[key] = [...values]; changed(); }); actions.append(button);
    }
    const list = document.createElement('div'); list.className = 'checkbox-list';
    for (const value of options[key]) {
      const labelElement = document.createElement('label');
      const input = document.createElement('input'); input.type = 'checkbox'; input.value = value; input.dataset.filter = key;
      input.addEventListener('change', () => { state[key] = [...list.querySelectorAll('input:checked')].map(i => i.value); changed(); });
      labelElement.append(input, document.createTextNode(value)); list.append(labelElement);
    }
    menu.append(actions, list); details.append(summary, menu); group.append(heading, details); $('category-filters').append(group);
  }
  for (const [id, index] of [['start-year', 0], ['end-year', 1]]) {
    const input = $(id); input.min = options.years[0]; input.max = options.years.at(-1); input.step = 1; input.disabled = options.years.length === 1;
    input.addEventListener('input', () => {
      state.years[index] = Number(input.value);
      if (state.years[0] > state.years[1]) state.years[1 - index] = state.years[index];
      changed();
    });
  }
  $('reset').disabled = false;
  $('reset').addEventListener('click', () => { state = defaultState(); changed(); });
  $('filters').addEventListener('submit', event => event.preventDefault());
  document.addEventListener('keydown', event => { if (event.key === 'Escape') document.querySelectorAll('details[open]').forEach(d => { d.open = false; }); });
}
function changed() {
  syncControls();
  revision++;
  clearTimeout(timer);
  timer = setTimeout(render, 100);
}
async function render() {
  const request = ++revision;
  document.querySelector('main').setAttribute('aria-busy', 'true');
  try {
    const result = await getDashboardData(state);
    if (request !== revision) return;
    const k = result.kpis;
    $('kpi-price').textContent = k.medianPrice == null ? '—' : k.medianPrice >= 1_000_000 ? `$${(k.medianPrice / 1_000_000).toFixed(2)}M` : `$${Math.round(k.medianPrice / 1000)}K`;
    $('kpi-price').title = k.medianPrice == null ? '' : `S$${number.format(k.medianPrice)}`;
    $('kpi-psm').replaceChildren(document.createTextNode(k.medianPsm == null ? '—' : `$${number.format(k.medianPsm)}`));
    if (k.medianPsm != null) { const unit = document.createElement('small'); unit.textContent = ' / sqm'; $('kpi-psm').append(unit); }
    $('kpi-count').textContent = number.format(k.count);
    $('kpi-towns').textContent = number.format(k.towns);
    status(k.count ? '' : 'No transactions match your filters. Select at least one town, flat type and storey range, or widen the year range.');
    $('partial-note').hidden = !result.excludedCount;
    if (result.excludedCount) $('partial-note').textContent = `${monthLabel(result.partialMonth)} is still in progress. Its ${number.format(result.excludedCount)} selected transactions are excluded from time-series charts; KPIs and town comparison retain them.`;
    await updateCharts(result);
  } catch (error) { showError(error); }
  finally { if (request === revision) document.querySelector('main').setAttribute('aria-busy', 'false'); }
}
function showError(error) {
  status(error instanceof SyntaxError ? 'The local dataset could not be read. Run python frontend/prepare_data.py again, then reload.' : error.message, true);
  for (const id of ['price-chart', 'psm-chart', 'volume-chart', 'town-chart']) emptyChart(id, 'Market charts are unavailable. See the message above.');
}
async function init() {
  try {
    options = await getFilterOptions();
    state = defaultState();
    $('coverage').textContent = `${monthLabel(options.firstMonth)} – ${monthLabel(options.latestMonth)} · ${number.format(options.count)} transactions`;
    makeFilters(); syncControls(); await render();
  } catch (error) { showError(error); }
}
init();

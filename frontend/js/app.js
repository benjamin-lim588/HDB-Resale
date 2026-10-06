import {getFilterOptions, getDashboardData} from './data.js';
import {updateCharts, emptyChart} from './charts.js';
import {MAX_COMPARE_TOWNS, defaultFilters, isAllSelected, toggleTown, selectTown, activeChips, removeChip} from './filters.js';

const $ = id => document.getElementById(id);
const number = new Intl.NumberFormat('en-SG', {maximumFractionDigits: 0});
let options, state, timer, revision = 0;
let primaryMetric = 'medianPrice';
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
  return defaultFilters(options);
}
function syncControls() {
  for (const [key] of definitions) {
    const selected = new Set(state[key]);
    const all = isAllSelected(state[key], options[key]);
    document.querySelectorAll(`input[data-filter="${key}"]`).forEach(input => {
      input.checked = key === 'towns' && all ? false : selected.has(input.value);
      input.disabled = key === 'towns' && !all && !input.checked && state.towns.length >= MAX_COMPARE_TOWNS;
    });
    $(`summary-${key}`).textContent = all ? `All ${key === 'flatTypes' ? 'flat types' : key === 'storeys' ? 'storeys' : 'towns'}` : `${state[key].length} selected`;
    if (key === 'towns') $('all-towns').setAttribute('aria-pressed', String(all));
  }
  $('start-year').value = state.years[0];
  $('end-year').value = state.years[1];
  $('start-label').textContent = state.years[0];
  $('end-label').textContent = state.years[1];
  const chips = activeChips(state, options);
  $('active-filters').replaceChildren();
  if (!chips.length) {
    const overview = document.createElement('span'); overview.className = 'filter-overview'; overview.textContent = 'All transactions · no active filters'; $('active-filters').append(overview);
  }
  for (const chip of chips) {
    const button = document.createElement('button'); button.type = 'button'; button.className = 'filter-chip';
    button.setAttribute('aria-label', `Remove ${chip.label} filter`);
    const close = document.createElement('span'); close.textContent = '×'; close.setAttribute('aria-hidden', 'true');
    button.append(document.createTextNode(chip.label), close);
    button.addEventListener('click', () => { removeChip(state, chip, options); changed(); });
    $('active-filters').append(button);
  }
  $('reset-all').disabled = !chips.length;
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
      const button = document.createElement('button'); button.type = 'button'; button.textContent = key === 'towns' && text === 'Select all' ? 'All towns overview' : text;
      if (key === 'towns' && text === 'Select all') { button.id = 'all-towns'; button.setAttribute('aria-pressed', 'true'); }
      button.addEventListener('click', () => { state[key] = [...values]; changed(); }); actions.append(button);
    }
    const list = document.createElement('div'); list.className = 'checkbox-list';
    for (const value of options[key]) {
      const labelElement = document.createElement('label');
      const input = document.createElement('input'); input.type = 'checkbox'; input.value = value; input.dataset.filter = key;
      input.addEventListener('change', () => {
        if (key === 'towns') {
          if (!toggleTown(state, value, options)) {
            $('comparison-limit').textContent = 'You can compare up to 3 towns. Remove a town before adding another.';
            syncControls(); return;
          }
        } else state[key] = [...list.querySelectorAll('input:checked')].map(i => i.value);
        changed();
      });
      labelElement.append(input, document.createTextNode(value)); list.append(labelElement);
    }
    menu.append(actions, list); details.append(summary, menu); group.append(heading, details);
    if (key === 'towns') {
      const help = document.createElement('p'); help.className = 'compare-help'; help.id = 'town-help';
      help.textContent = 'Choose up to 3 towns for separate trend lines, or use All towns overview.';
      const limit = document.createElement('p'); limit.className = 'comparison-limit'; limit.id = 'comparison-limit'; limit.setAttribute('role', 'status'); limit.setAttribute('aria-live', 'polite');
      summary.setAttribute('aria-describedby', 'town-help');
      group.append(help, limit);
    }
    $('category-filters').append(group);
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
  $('reset-all').addEventListener('click', () => { state = defaultState(); changed(); });
  document.querySelectorAll('[data-metric]').forEach(button => {
    button.disabled = false;
    button.addEventListener('click', () => {
      if (primaryMetric === button.dataset.metric) return;
      primaryMetric = button.dataset.metric;
      syncMetric();
      revision++; clearTimeout(timer); timer = setTimeout(render, 0);
    });
  });
  $('filters').addEventListener('submit', event => event.preventDefault());
  document.addEventListener('keydown', event => { if (event.key === 'Escape') document.querySelectorAll('details[open]').forEach(d => { d.open = false; }); });
}
function changed() {
  $('comparison-limit').textContent = !isAllSelected(state.towns, options.towns) && state.towns.length >= MAX_COMPARE_TOWNS ? '3 towns selected. Remove a town to compare another.' : '';
  syncControls();
  revision++;
  clearTimeout(timer);
  timer = setTimeout(render, 100);
}
function syncMetric() {
  const psm = primaryMetric === 'medianPsm';
  document.querySelectorAll('[data-metric]').forEach(button => button.setAttribute('aria-pressed', String(button.dataset.metric === primaryMetric)));
  $('primary-title').textContent = psm ? 'Median price per SQM trend' : 'Median resale price trend';
  $('primary-description').textContent = psm ? 'Resale price per square metre, month by month' : 'The typical resale price, month by month';
  $('primary-unit').textContent = psm ? 'SGD / sqm' : 'SGD';
  $('price-chart').setAttribute('aria-label', psm ? 'Monthly median price per square metre' : 'Monthly median resale price');
}
function townClicked(town) {
  if (selectTown(state, town, options)) changed();
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
    await updateCharts(result, {primaryMetric, onTownClick: townClicked});
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

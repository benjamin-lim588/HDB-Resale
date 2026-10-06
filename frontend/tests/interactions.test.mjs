import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {setTimeout as pause} from 'node:timers/promises';
import {defaultFilters, toggleTown, selectTown, activeChips, removeChip} from '../js/filters.js';

const raw = JSON.parse(await readFile(new URL('../data/transactions.json', import.meta.url), 'utf8'));
const options = {towns: raw.towns, flatTypes: raw.flatTypes, storeys: raw.storeys, years: [2017, 2026]};

test('selection rules: comparison limit, cross-filtering and chip/reset semantics', () => {
  const filters = defaultFilters(options);
  filters.flatTypes = ['4 ROOM']; filters.storeys = ['10 TO 12']; filters.years = [2021, 2026];
  assert.ok(toggleTown(filters, 'TAMPINES', options));
  assert.deepEqual(filters.towns, ['TAMPINES']);
  assert.ok(toggleTown(filters, 'BISHAN', options));
  assert.ok(toggleTown(filters, 'QUEENSTOWN', options));
  assert.equal(toggleTown(filters, 'ANG MO KIO', options), false);
  assert.equal(filters.towns.length, 3);
  assert.ok(selectTown(filters, 'ANG MO KIO', options));
  assert.deepEqual(filters.flatTypes, ['4 ROOM']);
  assert.deepEqual(filters.storeys, ['10 TO 12']);
  assert.deepEqual(filters.years, [2021, 2026]);
  assert.equal(selectTown(filters, 'UNKNOWN', options), false);
  const chips = activeChips(filters, options);
  assert.deepEqual(chips.map(c => c.label), ['ANG MO KIO', '4 ROOM', '10 TO 12', '2021–2026']);
  for (const chip of chips) removeChip(filters, chip, options);
  assert.deepEqual(filters, defaultFilters(options));
  filters.towns = [];
  removeChip(filters, activeChips(filters, options)[0], options);
  assert.deepEqual(filters.towns, options.towns);
});

// A small DOM/Plotly test double exercises the real app event wiring without
// introducing runtime dependencies. This is not a visual/browser-console test.
class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.attributes = {}; this.dataset = {};
    this.listeners = new Map(); this.style = {}; this.checked = false; this.disabled = false;
    this.classes = new Set();
    this.classList = {add: c => this.classes.add(c), remove: c => this.classes.delete(c), toggle: (c, on) => on ? this.classes.add(c) : this.classes.delete(c)};
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return (this.text ?? '') + this.children.map(c => c.textContent).join(''); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.text = ''; this.children = children; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(event, listener) { const values = this.listeners.get(event) ?? new Set(); values.add(listener); this.listeners.set(event, values); }
  on(event, listener) { this.addEventListener(event, listener); }
  removeListener(event, listener) { this.listeners.get(event)?.delete(listener); }
  fire(event, payload = {}) { for (const listener of this.listeners.get(event) ?? []) listener({preventDefault() {}, ...payload}); }
  click() { if (!this.disabled) this.fire('click'); }
  change() { if (!this.disabled) { this.checked = !this.checked; this.fire('change'); } }
  querySelectorAll(selector) { return descendants(this).filter(el => matches(el, selector)); }
}
function descendants(element) { return element.children.flatMap(child => [child, ...descendants(child)]); }
function matches(el, selector) {
  if (selector === '[data-metric]') return el.dataset.metric != null;
  if (selector === 'input:checked') return el.tag === 'input' && el.checked;
  const category = selector.match(/^input\[data-filter="(.*)"\]$/);
  if (category) return el.tag === 'input' && el.dataset.filter === category[1];
  if (selector === 'details[open]') return el.tag === 'details' && el.open;
  return el.tag === selector;
}

test('app events update KPIs, all charts, chips, comparison, metric and empty states without refetching', async () => {
  const html = await readFile(new URL('../index.html', import.meta.url), 'utf8');
  const root = new Element('body');
  const main = new Element('main'); root.append(main);
  for (const [, id] of html.matchAll(/id="([^"]+)"/g)) { const element = new Element(); element.id = id; main.append(element); }
  for (const metric of ['medianPrice', 'medianPsm']) { const button = new Element('button'); button.dataset.metric = metric; main.append(button); }
  const find = id => descendants(root).find(el => el.id === id);
  globalThis.document = {
    getElementById: find, createElement: tag => new Element(tag),
    createTextNode: text => { const element = new Element('#text'); element.textContent = text; return element; },
    querySelectorAll: selector => descendants(root).filter(el => matches(el, selector)),
    querySelector: selector => descendants(root).find(el => matches(el, selector)), addEventListener() {},
  };
  let fetches = 0, updates = 0;
  globalThis.fetch = async () => { fetches++; return {ok: true, json: async () => structuredClone(raw)}; };
  globalThis.window = {Plotly: {
    react: async (element, traces, layout) => { element.data = traces; element.layout = layout; updates++; },
    purge: element => { delete element.data; element.listeners.delete('plotly_click'); },
  }};
  const waitFor = async condition => {
    for (let i = 0; i < 250; i++) { if (condition()) return; await pause(10); }
    assert.fail('The dashboard event did not finish updating');
  };
  const act = async callback => {
    callback();
    await waitFor(() => main.attributes['aria-busy'] === 'false');
    await pause(160); // allow the existing filter debounce to start
    await waitFor(() => main.attributes['aria-busy'] === 'false');
    assert.equal(find('status').classes.has('error'), false, find('status').textContent);
  };
  await import('../js/app.js');
  await waitFor(() => updates === 4 && main.attributes['aria-busy'] === 'false');
  assert.equal(find('kpi-count').textContent, '241,920');
  const checkbox = (key, value) => document.querySelectorAll(`input[data-filter="${key}"]`).find(el => el.value === value);
  const chips = () => find('active-filters').children.filter(el => el.className === 'filter-chip');

  await act(() => find('town-chart').fire('plotly_click', {points: [{y: 'TAMPINES'}]}));
  assert.equal(find('kpi-towns').textContent, '1');
  assert.equal(find('price-chart').data[0].name, 'TAMPINES');
  assert.ok(checkbox('towns', 'TAMPINES').checked);
  assert.ok(chips().some(el => el.textContent === 'TAMPINES×'));
  const townIndex = find('town-chart').data[0].y.indexOf('TAMPINES');
  assert.equal(find('town-chart').data[0].marker.color[townIndex], '#176b58');

  await act(() => checkbox('towns', 'BISHAN').change());
  await act(() => checkbox('towns', 'QUEENSTOWN').change());
  assert.equal(find('price-chart').data.length, 3);
  assert.equal(find('psm-chart').data.length, 3);
  assert.equal(find('price-chart').layout.showlegend, true);
  assert.equal(checkbox('towns', 'ANG MO KIO').disabled, true);
  assert.equal(find('town-chart').listeners.get('plotly_click').size, 1);

  await act(() => chips().find(el => el.textContent === 'BISHAN×').click());
  assert.equal(find('price-chart').data.length, 2);
  assert.equal(checkbox('towns', 'ANG MO KIO').disabled, false);
  const beforeMetric = find('kpi-count').textContent;
  const townNames = find('price-chart').data.map(t => t.name);
  await act(() => document.querySelectorAll('[data-metric]')[1].click());
  assert.equal(find('primary-title').textContent, 'Median price per SQM trend');
  assert.equal(find('primary-unit').textContent, 'SGD / sqm');
  assert.equal(find('kpi-count').textContent, beforeMetric);
  assert.deepEqual(find('price-chart').data.map(t => t.name), townNames);
  assert.deepEqual(find('price-chart').data[0].y, find('psm-chart').data[0].y);
  assert.ok(find('price-chart').data[0].hovertemplate.includes('customdata[3]'));
  assert.equal(find('town-chart').data[0].customdata[0].length, 2);

  await act(() => find('reset-all').click());
  assert.equal(find('kpi-count').textContent, '241,920');
  assert.equal(chips().length, 0);
  assert.equal(find('price-chart').data.length, 1);
  assert.equal(find('primary-unit').textContent, 'SGD / sqm', 'reset filters keeps the metric preference');
  await act(() => document.querySelectorAll('[data-metric]')[0].click());
  assert.equal(find('primary-unit').textContent, 'SGD');

  // Category chips and year chips call the same selection rules as sidebar edits.
  const clearButton = key => {
    const group = find('category-filters').children.find(el => descendants(el).some(child => child.dataset.filter === key));
    return descendants(group).find(el => el.tag === 'button' && el.textContent === 'Clear');
  };
  const clearTypes = clearButton('flatTypes');
  await act(() => clearTypes.click());
  assert.equal(find('kpi-count').textContent, '0');
  assert.ok(!find('status').hidden);
  assert.ok(find('price-chart').textContent.includes('No transactions'));
  assert.equal(find('town-chart').listeners.get('plotly_click'), undefined);
  await act(() => chips().find(el => el.textContent === 'No flat types×').click());
  assert.equal(find('kpi-count').textContent, '241,920');
  assert.equal(find('town-chart').listeners.get('plotly_click').size, 1);

  await act(() => clearTypes.click());
  await act(() => checkbox('flatTypes', '4 ROOM').change());
  await act(() => clearButton('storeys').click());
  await act(() => checkbox('storeys', '10 TO 12').change());
  const filteredCount = find('kpi-count').textContent;
  await act(() => find('town-chart').fire('plotly_click', {points: [{y: 'TAMPINES'}]}));
  assert.notEqual(find('kpi-count').textContent, filteredCount);
  assert.deepEqual(chips().map(el => el.textContent), ['TAMPINES×', '4 ROOM×', '10 TO 12×']);
  assert.ok(checkbox('flatTypes', '4 ROOM').checked);
  assert.ok(checkbox('storeys', '10 TO 12').checked);
  await act(() => chips().find(el => el.textContent === '4 ROOM×').click());
  assert.ok(document.querySelectorAll('input[data-filter="flatTypes"]').every(el => el.checked));
  await act(() => chips().find(el => el.textContent === '10 TO 12×').click());
  assert.ok(document.querySelectorAll('input[data-filter="storeys"]').every(el => el.checked));
  await act(() => chips().find(el => el.textContent === 'TAMPINES×').click());
  assert.equal(find('kpi-count').textContent, '241,920');

  await act(() => { find('start-year').value = 2021; find('start-year').fire('input'); });
  assert.ok(chips().some(el => el.textContent === '2021–2026×'));
  await act(() => chips().find(el => el.textContent === '2021–2026×').click());
  assert.equal(Number(find('start-year').value), 2017);
  await act(() => { find('end-year').value = 2018; find('end-year').fire('input'); });
  await act(() => { find('start-year').value = 2024; find('start-year').fire('input'); });
  assert.equal(Number(find('end-year').value), 2024, 'moving from-year beyond to-year keeps a valid range');
  await act(() => clearButton('towns').click());
  assert.equal(find('kpi-count').textContent, '0');
  assert.ok(chips().some(el => el.textContent === 'No towns×'));
  await act(() => find('reset').click());
  assert.equal(find('kpi-count').textContent, '241,920');
  assert.equal(fetches, 1);
  assert.equal(checkbox('flatTypes', '4 ROOM').disabled, false);
});

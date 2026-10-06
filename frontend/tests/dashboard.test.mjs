import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {performance} from 'node:perf_hooks';

const NativeDate = Date;
globalThis.Date = class extends NativeDate {
  constructor(...args) { super(...(args.length ? args : ['2026-10-05T12:00:00Z'])); }
};
const raw = JSON.parse(await readFile(new URL('../data/transactions.json', import.meta.url), 'utf8'));
let fetches = 0;
globalThis.fetch = async () => { fetches++; return {ok: true, json: async () => structuredClone(raw)}; };
const data = await import('../js/data.js');
const options = await data.getFilterOptions();
const filters = {towns: options.towns, flatTypes: options.flatTypes, storeys: options.storeys, years: [2017, 2026]};

test('default KPIs equal independently calculated CSV medians; partial month omitted only from trends', async () => {
  const result = await data.getDashboardData(filters);
  assert.deepEqual(result.kpis, {medianPrice: 505000, medianPsm: 5326.086956521739, count: 241920, towns: 26});
  assert.equal(result.partialMonth, '2026-10');
  assert.equal(result.excludedCount, 212);
  assert.equal(result.monthly.at(-1).month, '2026-09-01');
  assert.equal(result.monthly.reduce((sum, m) => sum + m.count, 0), 241708);
  assert.equal(result.towns[0].town, 'BUKIT TIMAH');
  assert.equal(result.towns[0].medianPrice, 792500);
  assert.ok(result.towns.every((t, i, a) => !i || a[i - 1].medianPrice >= t.medianPrice));
});

test('all four filters combine correctly and retain exact medians', async () => {
  const result = await data.getDashboardData({...filters, towns: ['ANG MO KIO'], flatTypes: ['3 ROOM'], storeys: ['01 TO 03'], years: [2020, 2022]});
  assert.deepEqual(result.kpis, {medianPrice: 310000, medianPsm: 4391.067897165458, count: 330, towns: 1});
  assert.equal(result.monthly.reduce((s, m) => s + m.count, 0), 330);
  assert.ok(result.monthly.every(m => m.month >= '2020-01-01' && m.month <= '2022-12-01'));
});

test('per-town monthly medians are exact, not medians of group medians', async () => {
  const selected = ['TAMPINES', 'BISHAN', 'QUEENSTOWN'];
  const result = await data.getDashboardData({...filters, towns: selected, years: [2021, 2022]});
  assert.equal(result.townSeries.length, 3);
  const median = values => {
    values.sort((a, b) => a - b);
    return values.length ? (values[Math.floor((values.length - 1) / 2)] + values[Math.floor(values.length / 2)]) / 2 : null;
  };
  for (const series of result.townSeries) {
    const townIndex = raw.towns.indexOf(series.town);
    const monthIndex = raw.months.indexOf('2021-01');
    const ids = raw.columns[0].flatMap((m, i) => m === monthIndex && raw.columns[1][i] === townIndex ? [i] : []);
    assert.deepEqual(series.monthly[0], {month: '2021-01-01', medianPrice: median(ids.map(i => raw.columns[4][i])), medianPsm: median(ids.map(i => raw.columns[5][i])), count: ids.length});
  }
  const comparison = result.towns.find(t => t.town === 'TAMPINES');
  assert.ok(comparison.medianPsm > 0);
  assert.equal(result.monthly.reduce((s, m) => s + m.count, 0), result.townSeries.reduce((s, t) => s + t.monthly.reduce((total, m) => total + m.count, 0), 0));
});

test('each categorical filter and the year range affect matching records', async () => {
  for (const changed of [{towns: ['ANG MO KIO']}, {flatTypes: ['3 ROOM']}, {storeys: ['01 TO 03']}, {years: [2026, 2026]}]) {
    const result = await data.getDashboardData({...filters, ...changed});
    assert.ok(result.kpis.count > 0 && result.kpis.count < raw.count);
  }
  const a = await data.getTownComparison({...filters, towns: ['ANG MO KIO']});
  const b = await data.getTownComparison(filters);
  assert.deepEqual(a, b, 'town comparison intentionally honors all other filters across all towns');
});

test('empty selections return valid empty results and local data is fetched once', async () => {
  for (const key of ['towns', 'flatTypes', 'storeys']) {
    const result = await data.getDashboardData({...filters, [key]: []});
    assert.equal(result.kpis.count, 0);
    assert.equal(result.kpis.medianPrice, null);
    assert.ok(result.monthly.every(m => m.count === 0));
  }
  assert.equal(fetches, 1);
});

test('all four chart update calls and empty-state recovery', async () => {
  const elements = new Map();
  const calls = [];
  const get = id => {
    if (!elements.has(id)) elements.set(id, {id, classList: {add() {}, remove() {}}, style: {}, textContent: ''});
    return elements.get(id);
  };
  globalThis.document = {getElementById: get};
  globalThis.window = {Plotly: {react: async (element, traces, layout) => { element.data = traces; calls.push({element, traces, layout}); }, purge: element => { delete element.data; }}};
  const {updateCharts} = await import('../js/charts.js');
  await updateCharts(await data.getDashboardData(filters));
  assert.equal(calls.length, 4);
  assert.deepEqual(calls.map(c => c.traces[0].type), ['scatter', 'scatter', 'bar', 'bar']);
  assert.equal(calls[3].layout.yaxis.autorange, 'reversed');
  await updateCharts(await data.getDashboardData({...filters, flatTypes: []}));
  assert.equal(calls.length, 4);
  assert.ok([...elements.values()].every(el => !el.data && el.textContent.includes('No')));
  await updateCharts(await data.getDashboardData(filters));
  assert.equal(calls.length, 8);
});

test('missing and corrupt generated data fail with readable messages', async () => {
  globalThis.fetch = async () => ({ok: false});
  const missing = await import('../js/data.js?missing');
  await assert.rejects(missing.getFilterOptions(), /Local market data is missing/);
  globalThis.fetch = async () => ({ok: true, json: async () => ({version: 1, count: 3})});
  const invalid = await import('../js/data.js?invalid');
  await assert.rejects(invalid.getFilterOptions(), /prepared dataset is invalid/);
});

test('report query timing on the full dataset', async () => {
  const start = performance.now();
  await data.getDashboardData({...filters, years: [2018, 2025]});
  console.log(`Full-dataset filter query: ${(performance.now() - start).toFixed(1)} ms`);
});

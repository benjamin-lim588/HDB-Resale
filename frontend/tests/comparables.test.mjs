import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {MATCHING, selectComparables, summarizeComparables, validateProperty, median} from '../js/comparable-matching.js';
import {readPropertyValues} from '../js/fair-value-app.js';
import {validateComparableData} from '../js/comparables-data.js';

const asOf = '2026-10-06';
const property = {town: 'TAMPINES', street: 'TAMPINES ST 11', block: '101', flatType: '4 ROOM',
  floorArea: 100, storey: 8, remainingLeaseMonths: 780, askingPrice: 600000, flatModel: ''};
const sale = (id, changes = {}) => ({id, town: property.town, street: property.street, block: property.block,
  flatType: property.flatType, flatModel: 'Model A', date: '2026-09-01', floorArea: 100,
  storeyRange: '07 TO 09', storeyMid: 8, remainingLeaseMonths: 781, resalePrice: 550000, pricePerSqm: 5500, ...changes});

test('same-block sales are preferred and sufficient block sales stop widening at 5–10', () => {
  const candidates = [...Array.from({length: 12}, (_, i) => sale(i, {floorArea: 110})),
    ...Array.from({length: 12}, (_, i) => sale(i + 20, {block: '102'}))];
  const selected = selectComparables(property, candidates, asOf);
  assert.equal(selected.length, 10);
  assert.ok(selected.every(c => c.tier === 0));
});

test('progressive fallback fills from same street before considering same town', () => {
  const candidates = [sale(1), ...Array.from({length: 5}, (_, i) => sale(i + 2, {block: '102'})), sale(20, {street: 'TAMPINES ST 22'})];
  const selected = selectComparables(property, candidates, asOf);
  assert.deepEqual(selected.map(c => c.tier), [0, 1, 1, 1, 1, 1]);
  assert.equal(selected[0].block, '101');
});

test('same-town fallback retains narrower-tier matches, with no repeated candidate IDs', () => {
  const candidates = [sale(1), sale(2, {block: '102'}), ...Array.from({length: 10}, (_, i) => sale(i + 3, {street: 'TAMPINES ST 22'}))];
  const selected = selectComparables(property, candidates, asOf);
  assert.equal(selected.length, 10);
  assert.deepEqual(selected.slice(0, 3).map(c => c.tier), [0, 1, 2]);
  assert.equal(new Set(selected.map(c => c.id)).size, selected.length);
});

test('flat type and town must match; same block number on another street is town-tier', () => {
  const selected = selectComparables(property, [sale(1, {flatType: '5 ROOM'}), sale(2, {town: 'BISHAN'}), sale(3, {street: 'OTHER ST'})], asOf);
  assert.deepEqual(selected.map(c => c.id), [3]);
  assert.equal(selected[0].tier, 2);
});

test('similarity ranks smaller area, storey, lease and recency differences first', () => {
  for (const changes of [{floorArea: 110}, {storeyMid: 11}, {remainingLeaseMonths: 850}, {date: '2025-09-01', remainingLeaseMonths: 793}]) {
    const selected = selectComparables(property, [sale(1, changes), sale(2)], asOf);
    assert.equal(selected[0].id, 2);
    assert.ok(selected[0].score < selected[1].score);
  }
  const selected = selectComparables({...property, flatModel: 'Model A'}, [sale(1, {flatModel: 'Model B'}), sale(2)], asOf);
  assert.equal(selected[0].id, 2);
});

test('asking price never changes selected transactions or ranking scores', () => {
  const candidates = [sale(1), sale(2, {block: '102', resalePrice: 900000, pricePerSqm: 9000})];
  const cheap = selectComparables({...property, askingPrice: 100000}, candidates, asOf);
  const expensive = selectComparables({...property, askingPrice: 9000000}, candidates, asOf);
  assert.deepEqual(cheap, expensive);
  assert.deepEqual(cheap, selectComparables({...property, askingPrice: NaN}, candidates, asOf));
});

test('hard similarity limits, recency and future exclusion are enforced', () => {
  const candidates = [sale(1, {floorArea: 121}), sale(2, {storeyMid: 15}), sale(3, {remainingLeaseMonths: 902}),
    sale(4, {date: '2023-09-01'}), sale(5, {date: '2026-11-01'}), sale(6, {date: '2026-10-07'}),
    sale(7, {date: '2026-02-30'}), sale(8, {floorArea: NaN}), sale(9)];
  assert.deepEqual(selectComparables(property, candidates, asOf).map(c => c.id), [9]);
  const boundary = sale(10, {date: '2023-10-01', floorArea: 120, storeyMid: 14, remainingLeaseMonths: 936});
  assert.equal(selectComparables(property, [boundary], asOf).length, 1);
  assert.throws(() => selectComparables(property, [], '2026-02-30'), /Invalid analysis date/);
});

test('lease matching uses elapsed months while preserving original source lease', () => {
  const c = selectComparables(property, [sale(1, {date: '2025-10-01', remainingLeaseMonths: 792})], asOf)[0];
  assert.equal(c.remainingLeaseMonths, 792);
  assert.equal(c.estimatedLeaseNow, 780);
  assert.equal(c.leaseDifference, 0);
});

test('median price per sqm, area-scaled value, range and premium are calculated exactly', () => {
  const comps = [sale(1, {pricePerSqm: 5000, resalePrice: 500000}), sale(2, {pricePerSqm: 6000, resalePrice: 660000}), sale(3, {pricePerSqm: 20000, resalePrice: 1400000})].map(c => ({...c, tier: 0, ageMonths: 1}));
  const result = summarizeComparables({...property, floorArea: 90, askingPrice: 594000}, comps);
  assert.equal(result.medianPricePerSqm, 6000);
  assert.equal(result.impliedValue, 540000);
  assert.equal(result.medianTransactionPrice, 660000);
  assert.deepEqual(result.range, {lower: 450000, upper: 1800000});
  assert.equal(result.difference, 54000);
  assert.ok(Math.abs(result.differencePercent - 10) < 1e-10);
  assert.equal(result.assessment, 'Within comparable range');
  const discount = summarizeComparables({...property, floorArea: 90, askingPrice: 400000}, comps);
  assert.ok(discount.differencePercent < 0);
  assert.equal(discount.assessment, 'Below comparable market');
  assert.equal(summarizeComparables({...property, floorArea: 90, askingPrice: 1900000}, comps).assessment, 'Above comparable market');
  assert.equal(median([1, 9, 3, 7]), 5);
  assert.equal(median([]), null);
});

test('quality is deterministic and small or mostly town-level samples remain Low', () => {
  const sample = Array.from({length: 5}, (_, id) => ({...sale(id), tier: 1, ageMonths: 1}));
  assert.equal(summarizeComparables(property, sample).confidence, 'High');
  assert.equal(summarizeComparables(property, sample.map(c => ({...c, ageMonths: 18}))).confidence, 'Medium');
  assert.equal(summarizeComparables(property, sample.map(c => ({...c, tier: 2}))).confidence, 'Low');
  assert.equal(summarizeComparables(property, sample.slice(0, 4)).confidence, 'Low');
});

test('no-comparable result has no invented price benchmark', () => {
  assert.deepEqual(selectComparables(property, [sale(1, {flatType: '3 ROOM'})], asOf), []);
  const result = summarizeComparables(property, []);
  for (const key of ['impliedValue', 'medianPricePerSqm', 'medianTransactionPrice', 'range', 'differencePercent']) assert.equal(result[key], null);
  assert.equal(result.count, 0);
  assert.equal(result.confidence, 'Unavailable');
});

test('form parsing and validation reject blanks, invalid lease components and nonfinite inputs', () => {
  const values = {town: 'TAMPINES', street: 'TAMPINES ST 11', block: '101', flatType: '4 ROOM', floorArea: '100',
    storeyRange: '07 TO 09', leaseYears: '65', leaseMonths: '0', askingPrice: '600000'};
  assert.deepEqual(readPropertyValues(values), property);
  assert.deepEqual(validateProperty(property), {});
  for (const changes of [{leaseMonths: '12'}, {leaseYears: '65.5'}, {leaseYears: ''}, {leaseMonths: ''}, {leaseYears: '99', leaseMonths: '1'}]) {
    assert.ok(validateProperty(readPropertyValues({...values, ...changes})).remainingLeaseMonths);
  }
  for (const changes of [{town: ''}, {street: ''}, {block: ''}, {flatType: ''}, {floorArea: ''}, {floorArea: 'Infinity'},
    {askingPrice: '-1'}, {storeyRange: ''}]) {
    assert.ok(Object.keys(validateProperty(readPropertyValues({...values, ...changes}))).length > 0);
  }
});

test('prepared data provider constrains streets/blocks, fetches once and does not fabricate an ML value', async () => {
  const raw = JSON.parse(await readFile(new URL('../data/comparables.json', import.meta.url), 'utf8'));
  let fetches = 0;
  globalThis.fetch = async () => { fetches++; return {ok: true, json: async () => structuredClone(raw)}; };
  const provider = await import('../js/comparables-data.js?provider');
  const options = await provider.getComparableOptions();
  assert.ok(options.locations.TAMPINES['TAMPINES ST 11'].includes('101'));
  const i = raw.columns[0].findIndex(d => raw.dates[d] === '2026-09-01');
  assert.ok(i >= 0);
  const target = {town: raw.towns[raw.columns[1][i]], street: raw.streets[raw.columns[2][i]], block: raw.blocks[raw.columns[3][i]],
    flatType: raw.flatTypes[raw.columns[4][i]], flatModel: '', floorArea: raw.columns[7][i],
    remainingLeaseMonths: raw.columns[8][i] - 1, storey: raw.columns[9][i], askingPrice: 600000};
  const result = await provider.getComparableAnalysis(target, {asOf});
  assert.ok(result.count >= 1 && result.count <= MATCHING.maxComparables);
  assert.ok(result.comparables.every(c => c.flatType === target.flatType && c.town === target.town));
  assert.equal(result.impliedValue, median(result.comparables.map(c => c.pricePerSqm)) * target.floorArea);
  assert.equal(result.modelFairValue, null);
  assert.equal(await provider.getModelFairValue(target), null);
  assert.deepEqual((await provider.getComparableAnalysis({...target, askingPrice: 900000}, {asOf})).comparables, result.comparables);
  assert.equal(fetches, 1);
  await assert.rejects(provider.getComparableAnalysis({...target, askingPrice: NaN}), /Asking price/);
});

test('missing and corrupt data show actionable errors', async () => {
  globalThis.fetch = async () => ({ok: false});
  const missing = await import('../js/comparables-data.js?missing');
  await assert.rejects(missing.getComparableOptions(), /Local comparable data is missing/);
  globalThis.fetch = async () => ({ok: true, json: async () => ({version: 1, count: 4})});
  const invalid = await import('../js/comparables-data.js?invalid');
  await assert.rejects(invalid.getComparableOptions(), /comparable dataset is invalid/);
  assert.throws(() => validateComparableData({version: 1, count: 1}), /invalid/);
});

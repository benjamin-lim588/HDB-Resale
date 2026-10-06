import {selectComparables, summarizeComparables, validateProperty, singaporeToday} from './comparable-matching.js';

const DICTIONARIES = ['dates', 'towns', 'streets', 'blocks', 'flatTypes', 'flatModels', 'storeys'];
const FIELDS = ['transaction_date', 'town', 'street_name', 'block', 'flat_type', 'flat_model', 'storey_range',
  'floor_area_sqm', 'remaining_lease_months', 'storey_mid', 'resale_price', 'price_per_sqm'];
let pending;

export function validateComparableData(raw) {
  const fail = () => { throw new Error('The comparable dataset is invalid. Run python frontend/prepare_comparables_data.py, then reload.'); };
  if (raw.version !== 1 || !Number.isInteger(raw.count) || raw.count < 1 || JSON.stringify(raw.fields) !== JSON.stringify(FIELDS)) fail();
  for (const key of DICTIONARIES) {
    if (!Array.isArray(raw[key]) || !raw[key].length || raw[key].some(v => typeof v !== 'string' || !v.trim()) || new Set(raw[key]).size !== raw[key].length) fail();
  }
  if (raw.dates.some((d, i) => !/^\d{4}-\d{2}-\d{2}$/.test(d) || !Number.isFinite(Date.parse(d)) || new Date(d).toISOString().slice(0, 10) !== d || (i && d <= raw.dates[i - 1]))) fail();
  if (!Array.isArray(raw.columns) || raw.columns.length !== FIELDS.length) fail();
  for (const [c, column] of raw.columns.entries()) {
    if (!Array.isArray(column) || column.length !== raw.count) fail();
    for (const v of column) {
      if (c < DICTIONARIES.length ? !Number.isInteger(v) || v < 0 || v >= raw[DICTIONARIES[c]].length : !Number.isFinite(v) || v <= 0) fail();
    }
  }
  for (let i = 0; i < raw.count; i++) {
    if (raw.columns[8][i] > 1188 || raw.columns[9][i] > 60 || Math.abs(raw.columns[10][i] / raw.columns[7][i] / raw.columns[11][i] - 1) > 1e-6) fail();
  }
  const columns = raw.columns.map((values, c) => c < 7 ? Uint32Array.from(values) : Float64Array.from(values));
  const segments = new Map();
  const locations = new Map();
  for (let i = 0; i < raw.count; i++) {
    const key = `${columns[1][i]}:${columns[4][i]}`;
    if (!segments.has(key)) segments.set(key, []);
    segments.get(key).push(i);
    const town = raw.towns[columns[1][i]], street = raw.streets[columns[2][i]], block = raw.blocks[columns[3][i]];
    if (!locations.has(town)) locations.set(town, new Map());
    if (!locations.get(town).has(street)) locations.get(town).set(street, new Set());
    locations.get(town).get(street).add(block);
  }
  return {...raw, columns, segments, locations};
}

async function load() {
  if (!pending) pending = (async () => {
    const response = await fetch(new URL('../data/comparables.json', import.meta.url));
    if (!response.ok) throw new Error('Local comparable data is missing. Run python frontend/prepare_comparables_data.py from the repository root, then reload.');
    return validateComparableData(await response.json());
  })().catch(error => { pending = null; throw error; });
  return pending;
}

export async function getComparableOptions() {
  const db = await load();
  return {towns: [...db.towns], flatTypes: [...db.flatTypes], flatModels: [...db.flatModels], storeys: [...db.storeys],
    locations: Object.fromEntries([...db.locations].map(([town, streets]) => [town,
      Object.fromEntries([...streets].sort(([a], [b]) => a.localeCompare(b)).map(([street, blocks]) => [street, [...blocks].sort((a, b) => a.localeCompare(b, 'en', {numeric: true}))]))])),
    firstDate: db.dates[0], latestDate: db.dates.at(-1), count: db.count, duplicatesExcluded: db.duplicatesExcluded};
}

function decode(db, i) {
  const c = db.columns;
  return {id: i, date: db.dates[c[0][i]], town: db.towns[c[1][i]], street: db.streets[c[2][i]],
    block: db.blocks[c[3][i]], flatType: db.flatTypes[c[4][i]], flatModel: db.flatModels[c[5][i]],
    storeyRange: db.storeys[c[6][i]], floorArea: c[7][i], remainingLeaseMonths: c[8][i],
    storeyMid: c[9][i], resalePrice: c[10][i], pricePerSqm: c[11][i]};
}

// UI-facing provider boundary. A future authorized API can replace the local
// provider here; matching logic and presentation stay separate.
export async function getComparableAnalysis(property, {asOf = singaporeToday()} = {}) {
  const errors = validateProperty(property);
  if (Object.keys(errors).length) throw new Error(Object.values(errors).join(' '));
  const db = await load();
  const town = db.towns.findIndex(t => t.toUpperCase() === property.town.trim().toUpperCase());
  const type = db.flatTypes.findIndex(t => t.toUpperCase() === property.flatType.trim().toUpperCase());
  const indexes = db.segments.get(`${town}:${type}`) ?? [];
  const candidates = indexes.map(i => decode(db, i));
  const comparables = selectComparables(property, candidates, asOf);
  return {...summarizeComparables(property, comparables), asOf, latestDate: db.dates.at(-1),
    modelFairValue: await getModelFairValue(property)};
}

export async function getModelFairValue(_property) { return null; }

// The UI consumes this query interface, never CSV rows or storage details.
// Replace the local provider with API calls here when a backend is available.
let database;
let pendingLoad;
let lastKey;
let lastResult;

function validate(raw) {
  const fail = () => { throw new Error('The prepared dataset is invalid. Run python frontend/prepare_data.py again.'); };
  if (raw.version !== 1 || !Number.isInteger(raw.count) || raw.count < 1) fail();
  for (const key of ['months', 'towns', 'flatTypes', 'storeys']) {
    if (!Array.isArray(raw[key]) || !raw[key].length || raw[key].some(v => typeof v !== 'string' || !v.trim())) fail();
  }
  if (raw.months.some((m, i) => !/^\d{4}-(0[1-9]|1[0-2])$/.test(m) || (i && m <= raw.months[i - 1]))) fail();
  if (!Array.isArray(raw.columns) || raw.columns.length !== 6) fail();
  const limits = [raw.months.length, raw.towns.length, raw.flatTypes.length, raw.storeys.length];
  raw.columns.forEach((column, c) => {
    if (!Array.isArray(column) || column.length !== raw.count) fail();
    for (const value of column) {
      if (c < 4 ? !Number.isInteger(value) || value < 0 || value >= limits[c] : !Number.isFinite(value) || value <= 0) fail();
    }
  });
  for (const [key, c] of [['priceOrder', 4], ['psmOrder', 5]]) {
    const seen = new Uint8Array(raw.count);
    if (!Array.isArray(raw[key]) || raw[key].length !== raw.count) fail();
    let previous = -Infinity;
    for (const i of raw[key]) {
      if (!Number.isInteger(i) || i < 0 || i >= raw.count || seen[i] || raw.columns[c][i] < previous) fail();
      seen[i] = 1;
      previous = raw.columns[c][i];
    }
  }
  return {...raw, columns: raw.columns.map((c, i) => i < 4 ? Uint16Array.from(c) : Float64Array.from(c)),
    priceOrder: Uint32Array.from(raw.priceOrder), psmOrder: Uint32Array.from(raw.psmOrder)};
}

async function load() {
  if (database) return database;
  if (!pendingLoad) pendingLoad = (async () => {
    const response = await fetch(new URL('../data/transactions.json', import.meta.url));
    if (!response.ok) throw new Error('Local market data is missing. Run python frontend/prepare_data.py from the repository root, then reload this page.');
    database = validate(await response.json());
    return database;
  })().catch(error => { pendingLoad = null; throw error; });
  return pendingLoad;
}

export async function getFilterOptions() {
  const db = await load();
  return {towns: [...db.towns], flatTypes: [...db.flatTypes], storeys: [...db.storeys],
    years: [...new Set(db.months.map(m => Number(m.slice(0, 4))))],
    firstMonth: db.months[0], latestMonth: db.months.at(-1), count: db.count};
}

function currentMonth() {
  const parts = new Intl.DateTimeFormat('en', {timeZone: 'Asia/Singapore', year: 'numeric', month: '2-digit'}).formatToParts(new Date());
  return `${parts.find(p => p.type === 'year').value}-${parts.find(p => p.type === 'month').value}`;
}
function median(sorted) {
  const n = sorted.length;
  return n ? (sorted[Math.floor((n - 1) / 2)] + sorted[Math.floor(n / 2)]) / 2 : null;
}
function selectedMask(options, selected) {
  const set = new Set(selected);
  return Uint8Array.from(options, value => set.has(value) ? 1 : 0);
}

export async function getDashboardData(filters) {
  const db = await load();
  const partialMonth = db.months.at(-1) >= currentMonth() ? db.months.at(-1) : null;
  const key = JSON.stringify([filters, partialMonth]);
  if (key === lastKey) return lastResult;
  const towns = selectedMask(db.towns, filters.towns);
  const types = selectedMask(db.flatTypes, filters.flatTypes);
  const storeys = selectedMask(db.storeys, filters.storeys);
  const [month, town, type, storey, price, psm] = db.columns;
  const monthOK = db.months.map(m => Number(m.slice(0, 4)) >= filters.years[0] && Number(m.slice(0, 4)) <= filters.years[1]);
  const matched = new Uint8Array(db.count);
  const comparison = new Uint8Array(db.count);
  const represented = new Set();
  const buckets = db.months.map(() => ({prices: [], psm: []}));
  const townPrices = db.towns.map(() => []);
  const townPsms = db.towns.map(() => []);
  const comparing = filters.towns.length > 0 && filters.towns.length <= 3;
  const comparisonBuckets = new Map(comparing ? filters.towns.map(name => [
    db.towns.indexOf(name), db.months.map(() => ({prices: [], psm: []})),
  ]) : []);
  let count = 0, excludedCount = 0;
  for (let i = 0; i < db.count; i++) {
    if (!monthOK[month[i]] || !types[type[i]] || !storeys[storey[i]]) continue;
    comparison[i] = 1;
    if (!towns[town[i]]) continue;
    matched[i] = 1;
    count++;
    represented.add(town[i]);
    if (db.months[month[i]] === partialMonth) excludedCount++;
  }
  const prices = [], psms = [];
  // Pre-sorted indexes preserve the full price distributions and exact medians.
  // Each interaction uses linear scans, no CSV parsing or per-group sorting.
  for (const i of db.priceOrder) {
    if (comparison[i]) townPrices[town[i]].push(price[i]);
    if (!matched[i]) continue;
    prices.push(price[i]);
    if (db.months[month[i]] !== partialMonth) {
      buckets[month[i]].prices.push(price[i]);
      comparisonBuckets.get(town[i])?.[month[i]].prices.push(price[i]);
    }
  }
  for (const i of db.psmOrder) {
    if (comparison[i]) townPsms[town[i]].push(psm[i]);
    if (!matched[i]) continue;
    psms.push(psm[i]);
    if (db.months[month[i]] !== partialMonth) {
      buckets[month[i]].psm.push(psm[i]);
      comparisonBuckets.get(town[i])?.[month[i]].psm.push(psm[i]);
    }
  }
  const activeMonths = db.months.filter((m, i) => monthOK[i] && m !== partialMonth);
  const monthly = [];
  if (activeMonths.length) {
    const [startYear, startMonth] = activeMonths[0].split('-').map(Number);
    const end = activeMonths.at(-1);
    const date = new Date(Date.UTC(startYear, startMonth - 1, 1));
    while (date.toISOString().slice(0, 7) <= end) {
      const m = date.toISOString().slice(0, 7);
      const i = db.months.indexOf(m);
      const bucket = i >= 0 ? buckets[i] : {prices: [], psm: []};
      monthly.push({month: `${m}-01`, medianPrice: median(bucket.prices), medianPsm: median(bucket.psm), count: bucket.prices.length});
      date.setUTCMonth(date.getUTCMonth() + 1);
    }
  }
  lastKey = key;
  lastResult = {
    kpis: {medianPrice: median(prices), medianPsm: median(psms), count, towns: represented.size},
    monthly, partialMonth, excludedCount,
    selectedTowns: [...filters.towns],
    isTownOverview: filters.towns.length === db.towns.length,
    townSeries: comparing ? filters.towns.map(name => {
      const perTown = comparisonBuckets.get(db.towns.indexOf(name));
      return {town: name, monthly: monthly.map(m => {
        const i = db.months.indexOf(m.month.slice(0, 7));
        const bucket = perTown?.[i] ?? {prices: [], psm: []};
        return {month: m.month, medianPrice: median(bucket.prices), medianPsm: median(bucket.psm), count: bucket.prices.length};
      })};
    }) : [],
    towns: db.towns.map((name, i) => ({town: name, medianPrice: median(townPrices[i]), medianPsm: median(townPsms[i]), count: townPrices[i].length}))
      .filter(t => t.count).sort((a, b) => b.medianPrice - a.medianPrice || a.town.localeCompare(b.town)),
  };
  return lastResult;
}

export async function getMonthlyPriceTrend(filters) { return (await getDashboardData(filters)).monthly.map(m => ({month: m.month, value: m.medianPrice})); }
export async function getMonthlyPsmTrend(filters) { return (await getDashboardData(filters)).monthly.map(m => ({month: m.month, value: m.medianPsm})); }
export async function getTransactionVolume(filters) { return (await getDashboardData(filters)).monthly.map(m => ({month: m.month, value: m.count})); }
export async function getTownComparison(filters) { return (await getDashboardData(filters)).towns; }

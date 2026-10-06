// All selection, ranking and confidence rules live here. No asking price enters
// selectComparables(): the value benchmark is established before price comparison.
export const MATCHING = Object.freeze({
  minComparables: 5, maxComparables: 10,
  maxAgeMonths: 36, recentMonths: 12,
  maxAreaFraction: 0.20, maxStoreyDifference: 6, maxLeaseDifferenceMonths: 120,
  weights: Object.freeze({area: 0.25, storey: 0.15, lease: 0.20, recency: 0.35, model: 0.05}),
  highRecentNearby: 5, mediumNearby: 3,
});
export const TIERS = ['Same block', 'Same street', 'Same town'];

export function singaporeToday() {
  const parts = new Intl.DateTimeFormat('en', {timeZone: 'Asia/Singapore',
    year: 'numeric', month: '2-digit', day: '2-digit'}).formatToParts(new Date());
  return ['year', 'month', 'day'].map(type => parts.find(p => p.type === type).value).join('-');
}

function validDate(value) {
  if (typeof value !== 'string' || !/^\d{4}-\d{2}-\d{2}$/.test(value)) return false;
  const date = new Date(`${value}T00:00:00Z`);
  return Number.isFinite(date.getTime()) && date.toISOString().slice(0, 10) === value;
}
function monthIndex(value) { return Number(value.slice(0, 4)) * 12 + Number(value.slice(5, 7)) - 1; }
const normalize = value => typeof value === 'string' ? value.trim().toUpperCase() : '';

export function validateProperty(property) {
  const errors = {};
  for (const field of ['town', 'street', 'block', 'flatType']) {
    if (!normalize(property[field])) errors[field] = 'Choose a value.';
  }
  for (const [field, minimum, maximum, label] of [
    ['floorArea', 10, 500, 'Floor area'], ['storey', 1, 60, 'Storey'],
    ['remainingLeaseMonths', 1, 1188, 'Remaining lease'], ['askingPrice', 1, 100000000, 'Asking price'],
  ]) {
    if (typeof property[field] !== 'number' || !Number.isFinite(property[field]) || property[field] < minimum || property[field] > maximum) {
      errors[field] = `${label} must be between ${minimum} and ${maximum.toLocaleString('en-SG')}.`;
    }
  }
  if (Number.isFinite(property.remainingLeaseMonths) && !Number.isInteger(property.remainingLeaseMonths)) errors.remainingLeaseMonths = 'Remaining lease must be in whole months.';
  return errors;
}

export function selectComparables(property, transactions, asOf = singaporeToday()) {
  if (!validDate(asOf)) throw new Error('Invalid analysis date.');
  const p = {town: normalize(property.town), street: normalize(property.street), block: normalize(property.block),
    flatType: normalize(property.flatType), flatModel: normalize(property.flatModel),
    floorArea: property.floorArea, storey: property.storey, remainingLeaseMonths: property.remainingLeaseMonths};
  const groups = [[], [], []];
  for (const transaction of transactions) {
    if (normalize(transaction.town) !== p.town || normalize(transaction.flatType) !== p.flatType) continue;
    if (!validDate(transaction.date) || transaction.date > asOf) continue;
    if (![transaction.floorArea, transaction.storeyMid, transaction.remainingLeaseMonths, transaction.resalePrice, transaction.pricePerSqm].every(v => Number.isFinite(v) && v > 0)) continue;
    const ageMonths = monthIndex(asOf) - monthIndex(transaction.date);
    if (ageMonths > MATCHING.maxAgeMonths) continue;
    const areaFraction = Math.abs(transaction.floorArea - p.floorArea) / p.floorArea;
    const storeyDifference = Math.abs(transaction.storeyMid - p.storey);
    // Compare with the lease remaining today, reducing the sale-time lease by
    // elapsed calendar months. Displayed source lease always remains unchanged.
    const estimatedLeaseNow = Math.max(0, transaction.remainingLeaseMonths - ageMonths);
    const leaseDifference = Math.abs(estimatedLeaseNow - p.remainingLeaseMonths);
    if (areaFraction > MATCHING.maxAreaFraction || storeyDifference > MATCHING.maxStoreyDifference || leaseDifference > MATCHING.maxLeaseDifferenceMonths) continue;
    const sameStreet = normalize(transaction.street) === p.street;
    const tier = sameStreet && normalize(transaction.block) === p.block ? 0 : sameStreet ? 1 : 2;
    const w = MATCHING.weights;
    const score = w.area * areaFraction / MATCHING.maxAreaFraction +
      w.storey * storeyDifference / MATCHING.maxStoreyDifference +
      w.lease * leaseDifference / MATCHING.maxLeaseDifferenceMonths +
      w.recency * ageMonths / MATCHING.maxAgeMonths +
      w.model * (p.flatModel && normalize(transaction.flatModel) !== p.flatModel ? 1 : 0);
    groups[tier].push({...transaction, tier, tierLabel: TIERS[tier], score, ageMonths,
      areaDifference: transaction.floorArea - p.floorArea, storeyDifference, leaseDifference, estimatedLeaseNow});
  }
  const selected = [];
  for (const group of groups) {
    group.sort((a, b) => a.score - b.score || b.date.localeCompare(a.date) ||
      a.street.localeCompare(b.street) || a.block.localeCompare(b.block) || a.id - b.id);
    selected.push(...group.slice(0, MATCHING.maxComparables - selected.length));
    if (selected.length >= MATCHING.minComparables) break;
  }
  return selected;
}

export function median(values) {
  const sorted = [...values].sort((a, b) => a - b);
  const n = sorted.length;
  return n ? (sorted[Math.floor((n - 1) / 2)] + sorted[Math.floor(n / 2)]) / 2 : null;
}

export function summarizeComparables(property, comparables) {
  if (!comparables.length) return {comparables, count: 0, impliedValue: null, medianPricePerSqm: null,
    medianTransactionPrice: null, range: null, difference: null, differencePercent: null,
    assessment: null, confidence: 'Unavailable', qualityReason: 'No eligible comparable transactions.'};
  const psm = comparables.map(c => c.pricePerSqm);
  const medianPricePerSqm = median(psm);
  const impliedValue = medianPricePerSqm * property.floorArea;
  // Range is min/max selected price per sqm scaled to target area, not a
  // statistical confidence interval and not raw-price min/max for other sizes.
  const range = {lower: Math.min(...psm) * property.floorArea, upper: Math.max(...psm) * property.floorArea};
  const nearby = comparables.filter(c => c.tier < 2);
  const recentNearby = nearby.filter(c => c.ageMonths <= MATCHING.recentMonths);
  const confidence = comparables.length >= MATCHING.minComparables && recentNearby.length >= MATCHING.highRecentNearby ? 'High' :
    comparables.length >= MATCHING.minComparables && nearby.length >= MATCHING.mediumNearby ? 'Medium' : 'Low';
  return {comparables, count: comparables.length, impliedValue, medianPricePerSqm,
    medianTransactionPrice: median(comparables.map(c => c.resalePrice)), range,
    difference: property.askingPrice - impliedValue,
    differencePercent: (property.askingPrice / impliedValue - 1) * 100,
    assessment: property.askingPrice > range.upper ? 'Above comparable market' : property.askingPrice < range.lower ? 'Below comparable market' : 'Within comparable range',
    confidence, qualityReason: `${comparables.length} selected; ${nearby.length} same-block/street sales, including ${recentNearby.length} from the last ${MATCHING.recentMonths} calendar months.`};
}

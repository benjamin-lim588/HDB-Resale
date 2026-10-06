import {getComparableOptions, getComparableAnalysis} from './comparables-data.js';
import {MATCHING, validateProperty} from './comparable-matching.js';

const $ = id => document.getElementById(id);
const money = value => value == null ? '—' : `S$${new Intl.NumberFormat('en-SG', {maximumFractionDigits: 0}).format(value)}`;
const number = value => new Intl.NumberFormat('en-SG', {maximumFractionDigits: 1}).format(value);
const lease = months => `${Math.floor(months / 12)}y ${months % 12}m`;
let options, revision = 0;
const controls = {town: ['property-town'], street: ['property-street'], block: ['property-block'],
  flatType: ['property-type'], floorArea: ['property-area'], storey: ['property-storey'],
  remainingLeaseMonths: ['lease-years', 'lease-months'], askingPrice: ['property-asking']};

function status(message, error = false) {
  $('fair-status').hidden = !message;
  $('fair-status').textContent = message;
  $('fair-status').classList.toggle('error', error);
}
function setOptions(id, values, placeholder) {
  const select = $(id);
  select.replaceChildren();
  for (const value of ['', ...values]) {
    const option = document.createElement('option');
    option.value = value; option.textContent = value || placeholder;
    select.append(option);
  }
  select.value = '';
}
function streetsChanged() {
  const streets = options.locations[$('property-town').value] ?? {};
  const blocks = streets[$('property-street').value] ?? [];
  setOptions('property-block', blocks, blocks.length ? 'Choose block' : 'Choose street first');
  $('property-block').disabled = !blocks.length;
}
function townChanged() {
  const streets = Object.keys(options.locations[$('property-town').value] ?? {});
  setOptions('property-street', streets, streets.length ? 'Choose street' : 'Choose town first');
  $('property-street').disabled = !streets.length;
  streetsChanged();
}

// Kept separate from DOM access so conversion/validation can be tested directly.
export function readPropertyValues(values) {
  const storeyMatch = String(values.storeyRange ?? '').match(/^(\d+) TO (\d+)$/);
  const years = String(values.leaseYears ?? '').trim(), months = String(values.leaseMonths ?? '').trim();
  const leaseValid = /^\d+$/.test(years) && /^\d+$/.test(months) && Number(months) <= 11;
  return {town: values.town, street: values.street, block: values.block, flatType: values.flatType,
    flatModel: values.flatModel || '', floorArea: Number(values.floorArea),
    storey: storeyMatch ? (Number(storeyMatch[1]) + Number(storeyMatch[2])) / 2 : NaN,
    remainingLeaseMonths: leaseValid ? Number(years) * 12 + Number(months) : NaN,
    askingPrice: Number(values.askingPrice)};
}

function readForm() {
  return readPropertyValues({town: $('property-town').value, street: $('property-street').value,
    block: $('property-block').value, flatType: $('property-type').value,
    flatModel: $('property-model').value, floorArea: $('property-area').value,
    storeyRange: $('property-storey').value, leaseYears: $('lease-years').value,
    leaseMonths: $('lease-months').value, askingPrice: $('property-asking').value});
}
function showErrors(errors) {
  for (const [key, ids] of Object.entries(controls)) {
    $(`error-${key}`).textContent = errors[key] ?? '';
    for (const id of ids) {
      $(id).setAttribute('aria-invalid', String(Boolean(errors[key])));
      $(id).setAttribute('aria-describedby', `error-${key}`);
    }
  }
}
function clearResult() {
  revision++;
  $('fair-results').hidden = true;
  $('check-value').disabled = false;
  $('check-value').textContent = 'Check comparables';
  $('property-form').setAttribute('aria-busy', 'false');
  if (window.Plotly && $('comparable-price-chart').data) window.Plotly.purge($('comparable-price-chart'));
}
function appendCell(row, text, secondary) {
  const td = document.createElement('td'); td.textContent = text;
  if (secondary) { const small = document.createElement('small'); small.textContent = secondary; td.append(small); }
  row.append(td);
  return td;
}
function renderTable(comparables) {
  $('comparable-rows').replaceChildren();
  for (const c of comparables) {
    const row = document.createElement('tr');
    appendCell(row, `Block ${c.block}`, c.street);
    appendCell(row, c.date.slice(0, 7));
    appendCell(row, c.flatType, c.flatModel);
    appendCell(row, number(c.floorArea));
    appendCell(row, c.storeyRange);
    appendCell(row, lease(c.remainingLeaseMonths), `${lease(c.estimatedLeaseNow)} estimated today`);
    appendCell(row, money(c.resalePrice));
    appendCell(row, money(c.pricePerSqm));
    const cell = appendCell(row, '');
    const tier = document.createElement('span'); tier.className = `tier-label tier-${c.tier}`; tier.textContent = c.tierLabel;
    cell.append(tier); $('comparable-rows').append(row);
  }
}
export async function renderComparableChart(analysis, property) {
  const element = $('comparable-price-chart');
  if (!window.Plotly) { element.textContent = 'Chart library unavailable. Restore frontend/vendor/plotly.min.js and reload.'; return; }
  const comps = analysis.comparables;
  if (!comps.length) { if (element.data) window.Plotly.purge(element); element.textContent = ''; return; }
  if (!element.data) element.textContent = '';
  const labels = comps.map((c, i) => `${i + 1}. Block ${c.block} · ${c.date.slice(0, 7)}`);
  const positions = comps.map((_, i) => i);
  const extent = [-0.5, comps.length - 0.5];
  element.style.height = `${Math.max(310, comps.length * 30 + 110)}px`;
  await window.Plotly.react(element, [
    {type: 'scatter', mode: 'markers', name: 'Comparable sale price', x: comps.map(c => c.resalePrice), y: positions,
      marker: {color: comps.map(c => ['#176b58', '#5b7cbb', '#bd8542'][c.tier]), size: 9},
      customdata: comps.map(c => [c.block, c.street, c.date.slice(0, 7), c.floorArea, c.pricePerSqm, c.tierLabel]),
      hovertemplate: 'Block %{customdata[0]} · %{customdata[1]}<br>%{customdata[2]}<br>S$%{x:,.0f}<br>%{customdata[3]} sqm · S$%{customdata[4]:,.0f} / sqm<br>%{customdata[5]}<extra></extra>'},
    {type: 'scatter', mode: 'lines', name: 'Comparable-implied value', x: [analysis.impliedValue, analysis.impliedValue], y: extent,
      line: {color: '#176b58', width: 2, dash: 'dash'}, hovertemplate: 'Comparable-implied value: S$%{x:,.0f}<extra></extra>'},
    {type: 'scatter', mode: 'lines', name: 'Asking price', x: [property.askingPrice, property.askingPrice], y: extent,
      line: {color: '#bd8542', width: 2}, hovertemplate: 'Asking price: S$%{x:,.0f}<extra></extra>'},
  ], {paper_bgcolor: '#fff', plot_bgcolor: '#fff', font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: '#74817a', size: 10},
    margin: {l: 145, r: 15, t: 45, b: 55}, hovermode: 'closest',
    legend: {orientation: 'h', x: 0, y: 1.16, font: {size: 10}},
    xaxis: {title: {text: 'Price (SGD)'}, gridcolor: '#edf1ee', zeroline: false, tickformat: ',.0f'},
    yaxis: {range: [extent[1], extent[0]], tickvals: positions, ticktext: labels, showgrid: false, zeroline: false}},
  {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d']});
}
async function renderResult(analysis, property) {
  $('fair-results').hidden = false;
  $('result-property').textContent = `Block ${property.block} · ${property.street} · ${property.flatType}`;
  $('result-asking').textContent = money(property.askingPrice);
  $('result-implied').textContent = money(analysis.impliedValue);
  $('result-psm').textContent = analysis.medianPricePerSqm == null ? 'No comparable benchmark' : `${money(analysis.medianPricePerSqm)} / sqm × ${number(property.floorArea)} sqm`;
  $('result-premium').textContent = analysis.differencePercent == null ? '—' : `${analysis.differencePercent >= 0 ? '+' : ''}${analysis.differencePercent.toFixed(1)}%`;
  $('result-assessment').textContent = analysis.assessment == null ? 'No comparable benchmark' : `${analysis.assessment} · ${money(Math.abs(analysis.difference))} ${analysis.difference >= 0 ? 'above' : 'below'} implied value`;
  $('result-model').textContent = analysis.modelFairValue == null ? 'Unavailable' : money(analysis.modelFairValue);
  $('result-range').textContent = analysis.range ? `${money(analysis.range.lower)}–${money(analysis.range.upper)}` : '—';
  $('result-median-price').textContent = money(analysis.medianTransactionPrice);
  $('result-count').textContent = analysis.count;
  $('result-tier').textContent = analysis.count ? [...new Set(analysis.comparables.map(c => c.tierLabel))].join(' · ') : 'No eligible transactions';
  $('result-confidence').textContent = analysis.confidence;
  $('result-quality').textContent = analysis.qualityReason;
  $('result-note').textContent = analysis.count ? `As of ${analysis.asOf}; dataset through ${analysis.latestDate.slice(0, 7)}. ${analysis.count < MATCHING.minComparables ? 'Fewer than five comparable transactions were found; this benchmark has low comparable quality. ' : ''}Historical prices are not adjusted for market changes. This is a comparable-sales benchmark, not an official appraisal.` :
    `No comparables match the same flat type within ${MATCHING.maxAgeMonths} calendar months, ±${MATCHING.maxAreaFraction * 100}% area, ±${MATCHING.maxStoreyDifference} storeys and ±${MATCHING.maxLeaseDifferenceMonths / 12} lease years, across this block, street or town. Check the property details. No benchmark has been calculated.`;
  $('comparison-chart-card').hidden = !analysis.count;
  renderTable(analysis.comparables);
  $('table-note').textContent = analysis.count ? 'Ranked by location tier, then similarity. Transaction dates in the source represent registration months.' : 'No eligible comparable transactions.';
  await renderComparableChart(analysis, property);
}

function methodology() {
  const rules = [
    `Same town and flat type are required. Look back ${MATCHING.maxAgeMonths} calendar months from today; future transactions are excluded.`,
    `Search same block (including its street), then same street, then same town. Widen only while fewer than ${MATCHING.minComparables} comparables are available; return at most ${MATCHING.maxComparables}.`,
    `Require area within ±${MATCHING.maxAreaFraction * 100}%, storey midpoint within ±${MATCHING.maxStoreyDifference} floors, and estimated remaining lease today within ±${MATCHING.maxLeaseDifferenceMonths / 12} years. Sale-time lease is reduced by elapsed calendar months for matching.`,
    `Within each tier, lower weighted differences rank first: area ${MATCHING.weights.area * 100}%, storey ${MATCHING.weights.storey * 100}%, lease ${MATCHING.weights.lease * 100}%, recency ${MATCHING.weights.recency * 100}%, optional flat-model mismatch ${MATCHING.weights.model * 100}%. Differences are normalized by the matching limits.`,
    `Quality is High with at least ${MATCHING.highRecentNearby} block/street sales from the last ${MATCHING.recentMonths} months; Medium with at least ${MATCHING.minComparables} sales including ${MATCHING.mediumNearby} block/street sales; otherwise Low. This describes comparable evidence, not prediction accuracy.`,
    'Comparable-implied value = median selected price per sqm × your floor area. Range = selected min/max price per sqm × your floor area. Premium/discount = (asking price / comparable-implied value − 1) × 100%. Asking price is excluded from matching.',
    `${number(options.duplicatesExcluded ?? 0)} identical exported rows excluded to avoid double-counting indistinguishable records. The source CSV is unchanged.`,
  ];
  const list = document.createElement('ul');
  for (const rule of rules) { const li = document.createElement('li'); li.textContent = rule; list.append(li); }
  $('matching-methodology').replaceChildren(list);
}

async function submit(event) {
  event.preventDefault();
  const property = readForm();
  const errors = validateProperty(property);
  if (options && !options.locations[property.town]?.[property.street]?.includes(property.block)) {
    errors.block = 'Choose a block on the selected street and town.';
  }
  showErrors(errors);
  if (Object.keys(errors).length) {
    clearResult(); status('Check the highlighted property details.', true);
    $(controls[Object.keys(errors)[0]][0]).focus(); return;
  }
  const request = ++revision;
  $('check-value').disabled = true; $('check-value').textContent = 'Finding comparables…';
  $('property-form').setAttribute('aria-busy', 'true');
  $('fair-results').hidden = true; status('Finding similar historical sales…');
  try {
    const result = await getComparableAnalysis(property);
    if (request !== revision) return;
    await renderResult(result, property);
    if (request === revision) status('');
  } catch (error) {
    if (request === revision) { $('fair-results').hidden = true; status(error.message, true); }
  } finally {
    if (request === revision) {
      $('check-value').disabled = false; $('check-value').textContent = 'Check comparables';
      $('property-form').setAttribute('aria-busy', 'false');
    }
  }
}

async function init() {
  try {
    options = await getComparableOptions();
    setOptions('property-town', options.towns, 'Choose town');
    setOptions('property-type', options.flatTypes, 'Choose flat type');
    setOptions('property-model', options.flatModels, 'No preference');
    setOptions('property-storey', options.storeys, 'Choose storey range');
    $('property-fields').disabled = false;
    $('comparable-coverage').textContent = `${options.firstDate.slice(0, 7)} – ${options.latestDate.slice(0, 7)} · ${number(options.count)} historical transactions`;
    methodology(); status('');
    $('property-town').addEventListener('change', townChanged);
    $('property-street').addEventListener('change', streetsChanged);
    $('property-form').addEventListener('submit', submit);
    for (const event of ['input', 'change']) $('property-form').addEventListener(event, () => { clearResult(); showErrors({}); status(''); });
    $('property-form').addEventListener('reset', () => {
      clearResult(); showErrors({}); status('');
      // Native form reset runs after this event. Reset dependent controls now,
      // using explicit empty options so stale streets/blocks cannot survive.
      setOptions('property-street', [], 'Choose town first'); $('property-street').disabled = true;
      setOptions('property-block', [], 'Choose street first'); $('property-block').disabled = true;
    });
  } catch (error) { status(error instanceof SyntaxError ? 'Comparable data could not be read. Run python frontend/prepare_comparables_data.py, then reload.' : error.message, true); }
}
// Tests can import the pure form conversion without starting a page.
if (typeof document !== 'undefined' && document.getElementById('property-form')) init();

import test from 'node:test';
import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {setTimeout as pause} from 'node:timers/promises';

// Event/DOM tests of the actual page controller; not a visual browser test.
class Element {
  constructor(tag = 'div') {
    this.tag = tag; this.children = []; this.attributes = {}; this.style = {};
    this.listeners = new Map(); this.value = ''; this.disabled = false; this.hidden = false;
    this.classes = new Set();
    this.classList = {add: c => this.classes.add(c), remove: c => this.classes.delete(c), toggle: (c, on) => on ? this.classes.add(c) : this.classes.delete(c)};
  }
  set textContent(value) { this.text = String(value); this.children = []; }
  get textContent() { return (this.text ?? '') + this.children.map(c => c.textContent).join(''); }
  append(...children) { this.children.push(...children); }
  replaceChildren(...children) { this.text = ''; this.children = children; }
  setAttribute(key, value) { this.attributes[key] = String(value); }
  addEventListener(event, listener) { const listeners = this.listeners.get(event) ?? []; listeners.push(listener); this.listeners.set(event, listeners); }
  fire(event) { for (const listener of this.listeners.get(event) ?? []) listener({preventDefault() {}}); }
  focus() { this.focused = true; }
}

test('Fair Value form cascades locations, validates inputs, renders values/table/chart and clears stale results', async () => {
  const NativeDate = Date;
  globalThis.Date = class extends NativeDate {
    constructor(...args) { super(...(args.length ? args : ['2026-10-06T12:00:00Z'])); }
  };
  const html = await readFile(new URL('../fair-value.html', import.meta.url), 'utf8');
  const raw = JSON.parse(await readFile(new URL('../data/comparables.json', import.meta.url), 'utf8'));
  const elements = new Map([...html.matchAll(/id="([^"]+)"/g)].map(([, id]) => [id, new Element()]));
  const $ = id => elements.get(id);
  globalThis.document = {getElementById: $, createElement: tag => new Element(tag)};
  let fetches = 0, charts = 0;
  globalThis.fetch = async () => { fetches++; return {ok: true, json: async () => structuredClone(raw)}; };
  globalThis.window = {Plotly: {react: async (element, traces, layout) => { charts++; element.data = traces; element.layout = layout; },
    purge: element => { delete element.data; }}};
  const until = async condition => {
    for (let i = 0; i < 200; i++) { if (condition()) return; await pause(10); }
    assert.fail(`Page did not complete: ${$('fair-status').textContent}`);
  };
  await import('../js/fair-value-app.js');
  await until(() => $('property-town').children.length > 1 && $('property-fields').disabled === false);
  assert.equal($('fair-status').hidden, true);
  assert.ok($('matching-methodology').textContent.includes('Asking price is excluded'));

  $('property-form').fire('submit');
  assert.ok($('error-town').textContent);
  assert.ok($('property-town').focused);
  assert.equal($('fair-results').hidden, true);

  const i = raw.columns[0].findIndex(d => raw.dates[d] === '2026-09-01');
  const town = raw.towns[raw.columns[1][i]], street = raw.streets[raw.columns[2][i]], block = raw.blocks[raw.columns[3][i]];
  const originalLease = raw.columns[8][i] - 1;
  $('property-town').value = town; $('property-town').fire('change'); $('property-form').fire('change');
  assert.ok($('property-street').children.some(o => o.value === street));
  assert.equal($('property-block').disabled, true);
  $('property-street').value = street; $('property-street').fire('change'); $('property-form').fire('change');
  assert.ok($('property-block').children.some(o => o.value === block));
  assert.equal($('property-block').disabled, false);
  $('property-block').value = block;
  $('property-type').value = raw.flatTypes[raw.columns[4][i]];
  $('property-area').value = String(raw.columns[7][i]);
  $('property-storey').value = raw.storeys[raw.columns[6][i]];
  $('lease-years').value = String(Math.floor(originalLease / 12));
  $('lease-months').value = String(originalLease % 12);
  $('property-asking').value = '600000';
  const submit = async () => {
    $('property-form').fire('submit');
    await until(() => $('property-form').attributes['aria-busy'] === 'false' && $('fair-results').hidden === false);
    assert.equal($('fair-status').classes.has('error'), false, $('fair-status').textContent);
  };
  await submit();
  assert.equal($('result-model').textContent, 'Unavailable');
  assert.equal($('result-asking').textContent, 'S$600,000');
  assert.ok(Number($('result-count').textContent) >= 1);
  assert.equal($('comparable-rows').children.length, Number($('result-count').textContent));
  assert.equal($('comparable-price-chart').data.length, 3);
  assert.equal($('comparable-price-chart').data[2].x[0], 600000);
  assert.ok($('comparable-rows').children[0].children[5].textContent.includes('estimated today'));
  assert.equal(charts, 1);
  const selected = structuredClone($('comparable-price-chart').data[0]);
  const implied = $('result-implied').textContent;

  $('property-asking').value = '900000'; $('property-form').fire('input');
  assert.equal($('fair-results').hidden, true, 'editing invalidates old results');
  await submit();
  assert.equal($('result-implied').textContent, implied);
  assert.deepEqual($('comparable-price-chart').data[0], selected);
  assert.equal($('comparable-price-chart').data[2].x[0], 900000);
  assert.equal(fetches, 1);

  $('lease-months').value = '12'; $('property-form').fire('input'); $('property-form').fire('submit');
  assert.ok($('error-remainingLeaseMonths').textContent);
  assert.equal($('fair-results').hidden, true);
  assert.equal($('lease-months').attributes['aria-invalid'], 'true');
  $('lease-years').value = '0'; $('lease-months').value = '1'; $('property-form').fire('input');
  await submit();
  assert.equal($('result-count').textContent, '0');
  assert.equal($('result-implied').textContent, '—');
  assert.equal($('comparison-chart-card').hidden, true);
  assert.equal($('comparable-rows').children.length, 0);
  assert.ok($('result-note').textContent.includes('No comparables match'));

  $('property-town').value = town === 'TAMPINES' ? 'BISHAN' : 'TAMPINES';
  $('property-town').fire('change'); $('property-form').fire('change');
  assert.equal($('property-street').value, '');
  assert.equal($('property-block').value, '');
  assert.equal($('property-block').disabled, true);
  assert.equal($('fair-results').hidden, true);
  $('property-form').fire('reset');
  assert.equal($('property-street').disabled, true);
  assert.equal($('property-block').disabled, true);
  assert.equal($('error-remainingLeaseMonths').textContent, '');
  assert.equal($('property-form').attributes['aria-busy'], 'false');
});

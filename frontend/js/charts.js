const config = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d']};
const townClickHandlers = new WeakMap();
const townColors = ['#176b58', '#5b7cbb', '#bd8542'];
const monthlyHover = '%{x|%B %Y}<br>%{customdata[0]}<br>Median resale price: S$%{customdata[1]:,.0f}<br>Median price per sqm: S$%{customdata[2]:,.0f}<br>Transactions: %{customdata[3]:,d}<extra></extra>';

function prepareElement(id) {
  const element = document.getElementById(id);
  element.classList.remove('chart-empty');
  if (!element.data) element.textContent = '';
  return element;
}

function bindTownClick(element, onTownClick) {
  const previous = townClickHandlers.get(element);
  if (previous) element.removeListener('plotly_click', previous);
  townClickHandlers.delete(element);
  if (typeof onTownClick !== 'function') return;
  const handler = event => {
    const town = event.points?.[0]?.y;
    if (typeof town === 'string') onTownClick(town);
  };
  element.on('plotly_click', handler);
  townClickHandlers.set(element, handler);
}
function layout(yTitle) {
  return {paper_bgcolor: '#fff', plot_bgcolor: '#fff', font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: '#74817a', size: 10},
    margin: {l: 64, r: 15, t: 18, b: 48}, hovermode: 'x unified', showlegend: false,
    xaxis: {type: 'date', tickformat: '%b %Y', nticks: 6, showgrid: false, zeroline: false, title: {text: 'Transaction month', font: {size: 10}}},
    yaxis: {title: {text: yTitle, font: {size: 10}}, gridcolor: '#edf1ee', zeroline: false, tickformat: ',.0f'},
  };
}
export function emptyChart(id, message) {
  const element = document.getElementById(id);
  if (id === 'town-chart') bindTownClick(element, null);
  if (window.Plotly && element.data) window.Plotly.purge(element);
  element.classList.add('chart-empty');
  element.textContent = message;
}
export async function updateCharts(result, {primaryMetric = 'medianPrice', onTownClick} = {}) {
  if (!window.Plotly) throw new Error('The chart library is missing. Restore frontend/vendor/plotly.min.js and reload.');
  const tasks = [];
  const series = result.townSeries?.length ? result.townSeries : [{town: 'Selected towns', monthly: result.monthly}];
  for (const [id, field, title, color, bar] of [
    ['price-chart', primaryMetric, primaryMetric === 'medianPsm' ? 'Median price / sqm (S$)' : 'Median price (S$)', '#176b58', false],
    ['psm-chart', 'medianPsm', 'Median price / sqm (S$)', '#5b7cbb', false],
    ['volume-chart', 'count', 'Transactions', '#73a991', true],
  ]) {
    if (!result.monthly.some(m => m.count > 0)) { emptyChart(id, result.kpis.count ? 'No complete months in this selection.' : 'No transactions match your filters.'); continue; }
    const element = prepareElement(id);
    const chartSeries = bar ? [{town: result.townSeries?.length === 1 ? result.townSeries[0].town : 'Selected towns', monthly: result.monthly}] : series;
    const traces = chartSeries.map((s, index) => {
      const traceColor = bar || !result.townSeries?.length ? color : townColors[index];
      return {name: s.town, x: s.monthly.map(m => m.month), y: s.monthly.map(m => m[field]),
        customdata: s.monthly.map(m => [s.town, m.medianPrice, m.medianPsm, m.count]),
        type: bar ? 'bar' : 'scatter', mode: 'lines',
        line: {color: traceColor, width: 2.5}, marker: {color: traceColor}, connectgaps: false,
        hovertemplate: monthlyHover};
    });
    const chartLayout = layout(title);
    chartLayout.showlegend = !bar && result.townSeries?.length > 1;
    chartLayout.legend = {orientation: 'h', x: 0, y: 1.16, font: {size: 10}};
    if (chartLayout.showlegend) chartLayout.margin.t = 40;
    if (bar) chartLayout.yaxis.rangemode = 'tozero';
    tasks.push(window.Plotly.react(element, traces, chartLayout, config));
  }
  if (!result.towns.length) emptyChart('town-chart', 'No towns match your flat-type, storey and year filters.');
  else {
    const element = prepareElement('town-chart');
    element.style.height = `${Math.max(300, result.towns.length * 27 + 65)}px`;
    const chartLayout = layout('');
    chartLayout.margin = {l: 122, r: 20, t: 10, b: 50};
    chartLayout.hovermode = 'closest';
    chartLayout.xaxis = {title: {text: 'Median resale price (S$)', font: {size: 10}}, tickformat: ',.0f', gridcolor: '#edf1ee', zeroline: false};
    chartLayout.yaxis = {type: 'category', autorange: 'reversed', categoryorder: 'array', categoryarray: result.towns.map(t => t.town), showgrid: false, tickfont: {size: 9}};
    const selected = new Set(result.selectedTowns);
    tasks.push(window.Plotly.react(element, [{type: 'bar', orientation: 'h', x: result.towns.map(t => t.medianPrice), y: result.towns.map(t => t.town),
      customdata: result.towns.map(t => [t.medianPsm, t.count]),
      marker: {color: result.towns.map((t, i) => result.isTownOverview ? (i < 3 ? '#176b58' : '#93b9a5') : selected.has(t.town) ? townColors[result.selectedTowns.indexOf(t.town) % 3] : '#c7d9cd')},
      hovertemplate: '%{y}<br>Median resale price: S$%{x:,.0f}<br>Median price per sqm: S$%{customdata[0]:,.0f}<br>Transactions: %{customdata[1]:,d}<extra></extra>'}], chartLayout, config)
      .then(() => bindTownClick(element, onTownClick)));
  }
  await Promise.all(tasks);
}

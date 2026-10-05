const config = {responsive: true, displaylogo: false, modeBarButtonsToRemove: ['select2d', 'lasso2d']};
function layout(yTitle) {
  return {paper_bgcolor: '#fff', plot_bgcolor: '#fff', font: {family: 'Inter, Segoe UI, Arial, sans-serif', color: '#74817a', size: 10},
    margin: {l: 64, r: 15, t: 18, b: 48}, hovermode: 'x unified', showlegend: false,
    xaxis: {type: 'date', tickformat: '%b %Y', nticks: 6, showgrid: false, zeroline: false, title: {text: 'Transaction month', font: {size: 10}}},
    yaxis: {title: {text: yTitle, font: {size: 10}}, gridcolor: '#edf1ee', zeroline: false, tickformat: ',.0f'},
  };
}
export function emptyChart(id, message) {
  const element = document.getElementById(id);
  if (window.Plotly && element.data) window.Plotly.purge(element);
  element.classList.add('chart-empty');
  element.textContent = message;
}
export async function updateCharts(result) {
  if (!window.Plotly) throw new Error('The chart library is missing. Restore frontend/vendor/plotly.min.js and reload.');
  const tasks = [];
  const x = result.monthly.map(m => m.month);
  const hasSeries = result.monthly.some(m => m.count > 0);
  for (const [id, field, title, color, bar] of [
    ['price-chart', 'medianPrice', 'Median price (S$)', '#176b58', false],
    ['psm-chart', 'medianPsm', 'Median price / sqm (S$)', '#5b7cbb', false],
    ['volume-chart', 'count', 'Transactions', '#73a991', true],
  ]) {
    if (!hasSeries) { emptyChart(id, result.kpis.count ? 'No complete months in this selection.' : 'No transactions match your filters.'); continue; }
    const element = document.getElementById(id);
    element.classList.remove('chart-empty');
    if (!element.data) element.textContent = '';
    const trace = {x, y: result.monthly.map(m => m[field]), type: bar ? 'bar' : 'scatter', mode: 'lines',
      line: {color, width: 2.5}, marker: {color}, connectgaps: false,
      hovertemplate: bar ? '%{x|%b %Y}<br>%{y:,d} transactions<extra></extra>' : '%{x|%b %Y}<br>S$%{y:,.0f}<extra></extra>'};
    const chartLayout = layout(title);
    if (bar) chartLayout.yaxis.rangemode = 'tozero';
    tasks.push(window.Plotly.react(element, [trace], chartLayout, config));
  }
  if (!result.towns.length) emptyChart('town-chart', 'No towns match your flat-type, storey and year filters.');
  else {
    const element = document.getElementById('town-chart');
    element.classList.remove('chart-empty');
    if (!element.data) element.textContent = '';
    element.style.height = `${Math.max(300, result.towns.length * 27 + 65)}px`;
    const chartLayout = layout('');
    chartLayout.margin = {l: 122, r: 20, t: 10, b: 50};
    chartLayout.hovermode = 'closest';
    chartLayout.xaxis = {title: {text: 'Median resale price (S$)', font: {size: 10}}, tickformat: ',.0f', gridcolor: '#edf1ee', zeroline: false};
    chartLayout.yaxis = {type: 'category', autorange: 'reversed', categoryorder: 'array', categoryarray: result.towns.map(t => t.town), showgrid: false, tickfont: {size: 9}};
    tasks.push(window.Plotly.react(element, [{type: 'bar', orientation: 'h', x: result.towns.map(t => t.medianPrice), y: result.towns.map(t => t.town),
      marker: {color: result.towns.map((_, i) => i < 3 ? '#176b58' : '#93b9a5')}, hovertemplate: '%{y}<br>S$%{x:,.0f}<extra></extra>'}], chartLayout, config));
  }
  await Promise.all(tasks);
}

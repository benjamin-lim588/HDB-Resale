// Shared selection rules for sidebar controls, chips and chart clicks.
export const MAX_COMPARE_TOWNS = 3;

export function defaultFilters(options) {
  return {towns: [...options.towns], flatTypes: [...options.flatTypes], storeys: [...options.storeys],
    years: [options.years[0], options.years.at(-1)]};
}

export function isAllSelected(selected, available) {
  return selected.length === available.length && available.every(value => selected.includes(value));
}

export function toggleTown(filters, town, options) {
  if (!options.towns.includes(town)) return false;
  const selected = isAllSelected(filters.towns, options.towns) ? [] : filters.towns;
  const next = selected.includes(town) ? selected.filter(value => value !== town) : [...selected, town];
  if (next.length > MAX_COMPARE_TOWNS) return false;
  filters.towns = next;
  return true;
}

export function selectTown(filters, town, options) {
  if (!options.towns.includes(town)) return false;
  filters.towns = [town];
  return true;
}

export function activeChips(filters, options) {
  const chips = [];
  for (const [key, label] of [['towns', 'towns'], ['flatTypes', 'flat types'], ['storeys', 'storeys']]) {
    if (isAllSelected(filters[key], options[key])) continue;
    if (!filters[key].length) chips.push({key, value: null, label: `No ${label}`});
    else for (const value of filters[key]) chips.push({key, value, label: value});
  }
  if (filters.years[0] !== options.years[0] || filters.years[1] !== options.years.at(-1)) {
    chips.push({key: 'years', value: null, label: `${filters.years[0]}–${filters.years[1]}`});
  }
  return chips;
}

export function removeChip(filters, chip, options) {
  if (chip.key === 'years') filters.years = [options.years[0], options.years.at(-1)];
  else {
    const remaining = filters[chip.key].filter(value => value !== chip.value);
    // Removing the final chip removes that dimension's restriction.
    filters[chip.key] = remaining.length ? remaining : [...options[chip.key]];
  }
}

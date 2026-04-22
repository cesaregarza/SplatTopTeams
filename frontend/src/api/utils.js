export function toPositiveIntList(value) {
  if (Array.isArray(value)) {
    return value
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item) && item > 0)
      .map((item) => Math.trunc(item));
  }
  if (Number.isFinite(Number(value))) {
    return [Math.trunc(Number(value))];
  }
  if (typeof value === 'string') {
    return (value.match(/\d+/g) || [])
      .map((item) => Number(item))
      .filter((item) => Number.isFinite(item) && item > 0)
      .map((item) => Math.trunc(item));
  }
  return [];
}

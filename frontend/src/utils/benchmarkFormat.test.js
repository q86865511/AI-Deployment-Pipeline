import { calculateActualProgress, getStatusColor, formatDateTime, stepNameMap, statusNameMap } from './benchmarkFormat';

describe('calculateActualProgress', () => {
  test('null status -> zeros', () => {
    expect(calculateActualProgress(null)).toEqual({ completed: 0, total: 0, percent: 0 });
  });
  test('zero total -> zeros', () => {
    expect(calculateActualProgress({ total_combinations: 0 })).toEqual({ completed: 0, total: 0, percent: 0 });
  });
  test('conversion stage occupies the first ~33%', () => {
    const p = calculateActualProgress({ total_combinations: 4, current_combination_index: 1, current_step: 'conversion' });
    expect(p.completed).toBe(2); // index + 1
    expect(p.total).toBe(4);
    expect(p.percent).toBe(Math.round((2 / 4) * 33.33)); // 17
    expect(p.stageProgress).toContain('轉換階段');
  });
  test('validation stage adds the 33% base', () => {
    const p = calculateActualProgress({ total_combinations: 4, current_combination_index: 3, current_step: 'validation' });
    expect(p.percent).toBe(33 + Math.round((4 / 4) * 33.33)); // 66
  });
  test('inference stage adds 66% base and caps at 100', () => {
    const p = calculateActualProgress({ total_combinations: 2, current_combination_index: 1, current_step: 'inference' });
    expect(p.percent).toBe(100);
    expect(p.percent).toBeLessThanOrEqual(100);
  });
  test('default step uses completed_combinations', () => {
    const p = calculateActualProgress({ total_combinations: 5, completed_combinations: 5, current_step: 'unknown' });
    expect(p.percent).toBe(100);
  });
});

describe('getStatusColor', () => {
  test.each([
    ['pending', 'blue'], ['processing', 'orange'], ['completed', 'green'],
    ['failed', 'red'], ['aborted', 'gray'], ['mystery', 'default'],
  ])('%s -> %s', (status, color) => {
    expect(getStatusColor(status)).toBe(color);
  });
});

describe('formatDateTime', () => {
  test('empty/null -> empty string', () => {
    expect(formatDateTime('')).toBe('');
    expect(formatDateTime(null)).toBe('');
  });
  test('valid ISO string -> formatted string containing the year', () => {
    const out = formatDateTime('2026-06-24T09:41:42Z');
    expect(typeof out).toBe('string');
    expect(out).toContain('2026');
  });
});

describe('name maps', () => {
  test('step names', () => {
    expect(stepNameMap.conversion).toBe('模型轉換');
    expect(stepNameMap.inference).toBe('推論測試');
  });
  test('status names', () => {
    expect(statusNameMap.completed).toBe('已完成');
    expect(statusNameMap.aborted).toBe('已中止');
  });
});

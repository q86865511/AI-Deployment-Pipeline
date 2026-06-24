import { cleanDisplayName, getModelColor, cleanGpuLabel } from './performanceFormat';

describe('cleanDisplayName', () => {
  test('empty/null -> Unknown', () => {
    expect(cleanDisplayName('')).toBe('Unknown');
    expect(cleanDisplayName(null)).toBe('Unknown');
  });
  test('explicit original keeps batch size', () => {
    expect(cleanDisplayName('original_model', 4)).toBe('原始模型_batch4');
  });
  test('non-engine name is treated as original (default batch 1)', () => {
    expect(cleanDisplayName('yolov8n.pt')).toBe('原始模型_batch1');
  });
  test('engine fp16 with batch', () => {
    expect(cleanDisplayName('model_engine_fp16_batch8')).toBe('fp16_8');
  });
  test('engine int8 default batch', () => {
    expect(cleanDisplayName('x_engine_int8')).toBe('int8_1');
  });
  test('engine fp32 default when no precision token', () => {
    expect(cleanDisplayName('x_engine_batch3')).toBe('fp32_3');
  });
});

describe('getModelColor', () => {
  test('original batch1 -> purple', () => {
    expect(getModelColor('原始模型_batch1')).toBe('#722ed1');
  });
  test('deterministic for the same key', () => {
    expect(getModelColor('fp16_8')).toBe(getModelColor('fp16_8'));
  });
  test('returns a value from the palette', () => {
    const palette = ['#1890ff', '#52c41a', '#fa8c16', '#eb2f96', '#13c2c2', '#f5222d'];
    expect(palette).toContain(getModelColor('fp16_8'));
  });
});

describe('cleanGpuLabel', () => {
  test('removes trailing float', () => {
    expect(cleanGpuLabel('GPU 0.8345')).toBe('GPU');
  });
  test('keeps a label with no float', () => {
    expect(cleanGpuLabel('GPU-0')).toBe('GPU-0');
  });
});

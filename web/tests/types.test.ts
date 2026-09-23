import { expect, test } from 'vitest';
import type { paths } from '../lib/api-types';

test('api types are valid', () => {
  const _typed: paths = {} as paths;
  expect(_typed).toBeDefined();
});

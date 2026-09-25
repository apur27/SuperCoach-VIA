// Flat ESLint config: JS recommended + typescript-eslint + Astro. Generated files are excluded.
import js from '@eslint/js';
import tseslint from 'typescript-eslint';
import astro from 'eslint-plugin-astro';

export default [
  { ignores: ['dist/', '.astro/', '.e2e-dist/', 'node_modules/', 'test-results/', 'playwright-report/', 'tests/fixtures/', 'src/lib/**/*.generated.*', 'src/lib/validators/'] },
  js.configs.recommended,
  ...tseslint.configs.recommended,
  ...astro.configs.recommended,
  {
    languageOptions: {
      globals: { window: 'readonly', document: 'readonly', console: 'readonly', process: 'readonly', URL: 'readonly', fetch: 'readonly', Buffer: 'readonly' },
    },
    rules: {
      '@typescript-eslint/no-unused-vars': ['error', { argsIgnorePattern: '^_', varsIgnorePattern: '^_' }],
      // Security (PLAN 11.2/11.5): no dynamic code evaluation anywhere in the site or its tooling.
      'no-eval': 'error',
      'no-implied-eval': 'error',
      'no-new-func': 'error',
    },
  },
  {
    // Classic (non-module) scripts served as-is from public/, e.g. the pre-paint theme bootstrap.
    files: ['public/**/*.js'],
    languageOptions: { sourceType: 'script', globals: { document: 'readonly', localStorage: 'readonly' } },
  },
];

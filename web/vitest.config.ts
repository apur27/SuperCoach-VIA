/// <reference types="vitest/config" />
import { getViteConfig } from 'astro/config';

// getViteConfig lets unit tests render .astro components (experimental_AstroContainer).
export default getViteConfig({
  test: {
    include: ['tests/unit/**/*.test.{ts,tsx}'],
    environment: 'node',
    testTimeout: 20000,
  },
});

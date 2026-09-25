// Plain `tsc` cannot resolve .astro modules (astro check can). Unit tests import components
// only to render them with experimental_AstroContainer, so the factory type is enough here.
declare module '*.astro' {
  import type { AstroComponentFactory } from 'astro/runtime/server/index.js';
  const Component: AstroComponentFactory;
  export default Component;
}

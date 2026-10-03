# SuperCoach VIA URL alias

This small owner Pages site sends differently cased `/supercoach-via/` addresses to the existing `/SuperCoach-VIA/` project site. The repository name and project base stay unchanged. The root page offers a link and does not redirect.

Publish only the contents of `public/` at the root of `apur27/apur27.github.io`:

- `.nojekyll`
- `index.html`
- `404.html`
- `alias.css`
- `alias.mjs`
- `path-map.mjs`
- `supercoach-via/index.html`

Do not publish this README or `tests/`. The lowercase entry is a static page. Other case variants and deep paths use the owner site's `404.html` before the browser opens the canonical project path. Such alias requests may initially return HTTP 404; the JavaScript redirect then loads the project page. Without JavaScript, the pages offer a plain link to the project home.

The redirect matches only the first path segment, keeps the remaining path/query/fragment, and rejects path normalization that would escape the canonical prefix. It does not redirect canonical project addresses or unrelated paths. No destination is read from a query parameter.

Run from the repository root with Node 22:

```sh
node --test ops/pages-alias/tests/path-map.test.mjs
node --test ops/pages-alias/tests/browser.test.mjs
```

The browser check uses the existing `web/node_modules` Playwright installation and cached Chromium. It serves the alias files with JavaScript MIME types and a small mock project destination. After publishing, verify the live `.mjs` MIME types, the lowercase entry, a deep link with query/fragment, and an unrelated path. Keep the project Pages deployment configured for `/SuperCoach-VIA/`.

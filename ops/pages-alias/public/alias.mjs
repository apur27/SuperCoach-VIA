import { aliasTarget } from './path-map.mjs';

const target = aliasTarget(window.location.pathname, window.location.search, window.location.hash);
if (target) {
  document.getElementById('project-link').href = target;
  window.location.replace(target);
}

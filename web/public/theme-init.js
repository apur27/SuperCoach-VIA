// Runs synchronously at the start of <body>: mark JS as available and apply a persisted theme
// before first paint. Same-origin file so the CSP needs no inline-script allowance.
(function () {
  const d = document.documentElement;
  d.classList.add('js');
  try {
    const t = localStorage.getItem('supercoach-via:theme:v1');
    if (t === 'light' || t === 'dark') d.setAttribute('data-theme', t);
  } catch {
    // storage disabled: fall back to the system theme
  }
})();

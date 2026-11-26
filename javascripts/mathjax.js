/* Load equations only when needed, including after Material instant navigation. */
(function () {
  "use strict";
  let loading;
  let pending = Promise.resolve();
  window.MathJax = {
    tex: { inlineMath: [["\\(", "\\)"]], displayMath: [["\\[", "\\]"]], processEscapes: true, processEnvironments: true },
    options: { ignoreHtmlClass: ".*|", processHtmlClass: "arithmatex" },
    startup: { typeset: false }
  };

  function load() {
    if (!loading) {
      loading = new Promise(function (resolve, reject) {
        const script = document.createElement("script");
        script.src = "https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js";
        script.async = true;
        script.onload = function () { window.MathJax.startup.promise.then(resolve, reject); };
        script.onerror = function () { script.remove(); loading = undefined; reject(new Error("Equation renderer unavailable")); };
        document.head.appendChild(script);
      });
    }
    return loading;
  }

  function render() {
    if (!document.querySelector(".arithmatex")) return;
    pending = pending.then(load).then(function () {
      // Read current containers after loading: the user may already have navigated.
      const containers = Array.from(document.querySelectorAll(".arithmatex"));
      window.MathJax.typesetClear();
      window.MathJax.texReset();
      if (window.MathJax.startup.output && window.MathJax.startup.output.clearCache) window.MathJax.startup.output.clearCache();
      return window.MathJax.typesetPromise(containers);
    }).catch(function () {
      // Source equations remain readable; a later navigation can retry a failed load.
    });
  }

  if (window.document$) window.document$.subscribe(render);
  else if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", render, { once: true });
  else render();
})();

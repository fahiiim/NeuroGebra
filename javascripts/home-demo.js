/* An accessible viewer for library-generated results. No Python runs here. */
(function (scope) {
  "use strict";

  function formatNumber(value) {
    if (!Number.isFinite(value)) return "unavailable";
    if (Math.abs(value) < 0.0000001) return "0.0000";
    return Math.abs(value) < 0.0001 ? value.toExponential(3) : value.toFixed(4);
  }

  function sampleIndex(value, grid) {
    const number = Number(value);
    if (!Number.isFinite(number) || grid.length < 2) throw new Error("Invalid sample input");
    const step = (grid[grid.length - 1] - grid[0]) / (grid.length - 1);
    return Math.max(0, Math.min(grid.length - 1, Math.round((number - grid[0]) / step)));
  }

  function curveKey(name, beta) {
    return name === "swish" ? "swish-" + Number(beta) : name;
  }

  function chartGeometry(grid, curve) {
    const all = curve.values.concat(curve.derivatives, [0]);
    const low = Math.min.apply(null, all);
    const high = Math.max.apply(null, all);
    const padding = Math.max((high - low) * 0.12, 0.1);
    const minimum = low - padding;
    const maximum = high + padding;
    const x = (index) => 42 + (556 * index) / (grid.length - 1);
    const y = (value) => 194 - (164 * (value - minimum)) / (maximum - minimum);
    const path = (values) => values.map((value, index) =>
      (index ? "L" : "M") + x(index).toFixed(2) + "," + y(value).toFixed(2)).join(" ");
    let derivatives = path(curve.derivatives);
    if (curve.name === "relu") {
      // Keep the discontinuity visible rather than drawing a ramp through zero.
      const zero = sampleIndex(0, grid);
      derivatives = path(curve.derivatives.slice(0, zero)) + " L" + x(zero).toFixed(2) + "," + y(0).toFixed(2) +
        " M" + x(zero).toFixed(2) + "," + y(1).toFixed(2) +
        curve.derivatives.slice(zero + 1).map((value, index) => " L" + x(index + zero + 1).toFixed(2) + "," + y(value).toFixed(2)).join("");
    }
    return { x, y, minimum, maximum, values: path(curve.values), derivatives };
  }

  function codeFor(curve, x) {
    const params = curve.name === "swish" ? ', params={"beta": ' + curve.beta.toFixed(1) + "}" : "";
    return 'from neurogebra import MathForge\n\n' +
      'f = MathForge().get("' + curve.name + '"' + params + ')\n' +
      'value = f.eval(x=' + x.toFixed(2) + ')\n' +
      'slope = f.gradient("x").eval(x=' + x.toFixed(2) + ')';
  }

  function validateData(data) {
    if (!data || data.schema !== 1 || !Array.isArray(data.x) || data.x.length < 2) throw new Error("Invalid demonstration data");
    if (!data.x.every(Number.isFinite)) throw new Error("Invalid input grid");
    const step = data.x[1] - data.x[0];
    if (step <= 0 || !data.x.every((x, i) => Math.abs(x - data.x[0] - i * step) < 1e-7)) throw new Error("Input grid must be uniform");
    ["sigmoid", "tanh", "relu", "swish-0.5", "swish-1", "swish-2"].forEach((key) => {
      const curve = data.curves && data.curves[key];
      if (!curve || ![curve.values, curve.derivatives].every((values) =>
        Array.isArray(values) && values.length === data.x.length && values.every(Number.isFinite))) throw new Error("Invalid curve data");
    });
    if (!data.training || !Array.isArray(data.training.losses) || data.training.losses.length < 2 || !data.training.losses.every(Number.isFinite)) throw new Error("Invalid training data");
    return data;
  }

  const query = (root, selector) => root.querySelector(selector);
  const setText = (root, selector, text) => { query(root, selector).textContent = text; };
  const api = { formatNumber, sampleIndex, curveKey, chartGeometry, codeFor, validateData, initExplorer, initTabs, initTraining };
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (!scope.document) return;

  const initialized = new WeakSet();

  function initTabs(root) {
    const tablist = query(root, "[data-ng-tabs]");
    if (!tablist) return;
    const tabs = Array.from(tablist.querySelectorAll('[role="tab"]'));
    function select(tab, focus) {
      tabs.forEach((item) => {
        const active = item === tab;
        item.setAttribute("aria-selected", String(active));
        item.tabIndex = active ? 0 : -1;
        query(root, "#" + item.getAttribute("aria-controls")).hidden = !active;
      });
      if (focus) tab.focus();
    }
    tabs.forEach((tab, index) => {
      tab.addEventListener("click", () => select(tab, false));
      tab.addEventListener("keydown", (event) => {
        let next;
        if (event.key === "ArrowRight") next = (index + 1) % tabs.length;
        if (event.key === "ArrowLeft") next = (index + tabs.length - 1) % tabs.length;
        if (event.key === "Home") next = 0;
        if (event.key === "End") next = tabs.length - 1;
        if (next === undefined) return;
        event.preventDefault();
        select(tabs[next], true);
      });
    });
    select(tabs[0], false);
    tablist.hidden = false;
  }

  function initExplorer(root, data) {
    const slider = query(root, "[data-ng-x]");
    const beta = query(root, "[data-ng-beta]");
    const buttons = Array.from(root.querySelectorAll("[data-ng-function]"));
    let name = "sigmoid";
    let geometry;
    let curve;

    function inspect() {
      const index = sampleIndex(slider.value, data.x);
      const x = data.x[index];
      const value = curve.values[index];
      const derivative = curve.derivatives[index];
      setText(root, "[data-ng-x-value]", x.toFixed(2));
      setText(root, "[data-ng-value]", formatNumber(value));
      setText(root, "[data-ng-slope]", formatNumber(derivative));
      setText(root, "[data-ng-code]", codeFor(curve, x));
      slider.setAttribute("aria-valuetext", "x " + x.toFixed(2) + "; function " + formatNumber(value) + "; derivative " + formatNumber(derivative));
      query(root, "[data-ng-crosshair]").setAttribute("d", "M" + geometry.x(index) + " 24V200");
      [["[data-ng-value-dot]", value], ["[data-ng-slope-dot]", derivative]].forEach(([selector, result]) => {
        const dot = query(root, selector);
        dot.setAttribute("cx", geometry.x(index));
        dot.setAttribute("cy", geometry.y(result));
      });
    }

    function choose() {
      curve = data.curves[curveKey(name, beta.value)];
      geometry = chartGeometry(data.x, curve);
      buttons.forEach((button) => button.setAttribute("aria-pressed", String(button.dataset.ngFunction === name)));
      query(root, "[data-ng-beta-field]").hidden = name !== "swish";
      setText(root, "[data-ng-formula]", "f(x) = " + curve.formula);
      setText(root, "[data-ng-note]", curve.note + " Values are precomputed with NeuroGebra.");
      setText(root, "[data-ng-plot-description]", curve.label + " and its symbolic derivative, sampled from x=-4 to 4. " + curve.note);
      setText(root, "[data-ng-ymax]", geometry.maximum.toFixed(1));
      setText(root, "[data-ng-ymin]", geometry.minimum.toFixed(1));
      query(root, "[data-ng-curve]").setAttribute("d", geometry.values);
      query(root, "[data-ng-derivative]").setAttribute("d", geometry.derivatives);
      inspect();
    }

    slider.addEventListener("input", inspect);
    beta.addEventListener("change", choose);
    buttons.forEach((button) => button.addEventListener("click", () => {
      name = button.dataset.ngFunction;
      choose();
      setText(root, "[data-ng-status]", curve.label + " selected. Use the input slider to inspect a sample.");
    }));
    choose();
    root.querySelectorAll("[data-ng-controls]").forEach((control) => { control.hidden = false; });
    query(root, "[data-ng-plot]").removeAttribute("hidden");
    query(root, "[data-ng-fallback]").hidden = true;
  }

  function initTraining(root, data) {
    const losses = data.training.losses;
    const points = losses.map((loss, i) => (i ? "L" : "M") +
      (12 + i * 496 / (losses.length - 1)).toFixed(2) + "," +
      (88 - 76 * loss / Math.max.apply(null, losses)).toFixed(2)).join(" ");
    query(root, "[data-ng-loss-line]").setAttribute("d", points);
    query(root, "[data-ng-loss-plot]").removeAttribute("hidden");
  }

  function initialize() {
    const root = scope.document.querySelector("[data-ng-home]");
    if (!root || initialized.has(root)) return;
    initialized.add(root);
    initTabs(root);
    scope.fetch(new URL(root.dataset.ngSource, scope.document.baseURI)).then((response) => {
      if (!response.ok) throw new Error("Data request failed");
      return response.json();
    }).then(validateData).then((data) => {
      if (!root.isConnected) return;
      initExplorer(root, data);
      initTraining(root, data);
    }).catch(() => {
      if (root.isConnected) setText(root, "[data-ng-status]", "Interactive controls could not load. The static figure, recorded trace and Python examples are still available.");
    });
  }

  if (scope.document$) scope.document$.subscribe(initialize);
  else if (scope.document.readyState === "loading") scope.document.addEventListener("DOMContentLoaded", initialize, { once: true });
  else initialize();
})(typeof window !== "undefined" ? window : globalThis);

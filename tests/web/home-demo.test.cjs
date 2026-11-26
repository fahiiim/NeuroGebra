/* Dependency-free interaction checks; run with node --test. */
const test = require("node:test");
const assert = require("node:assert/strict");
const fs = require("node:fs");
const path = require("node:path");
const demo = require("../../docs/javascripts/home-demo.js");
const data = JSON.parse(fs.readFileSync(path.join(__dirname, "../../docs/assets/data/explorer.json"), "utf8"));

class Element {
  constructor() { this.hidden = true; this.listeners = {}; this.attributes = {}; this.dataset = {}; this.value = ""; this.textContent = ""; this.tabIndex = 0; }
  setAttribute(name, value) { this.attributes[name] = String(value); }
  getAttribute(name) { return this.attributes[name]; }
  removeAttribute(name) { delete this.attributes[name]; if (name === "hidden") this.hidden = false; }
  addEventListener(name, callback) { this.listeners[name] = callback; }
  focus() { this.focused = true; }
  fire(name, event = {}) { this.listeners[name](event); }
}

function fixture() {
  const selectors = ["x", "beta", "beta-field", "x-value", "value", "slope", "code", "crosshair", "value-dot", "slope-dot", "formula", "note", "plot-description", "ymax", "ymin", "curve", "derivative", "status", "plot", "fallback", "loss-line", "loss-plot"];
  const elements = new Map(selectors.map((key) => ["[data-ng-" + key + "]", new Element()]));
  elements.get("[data-ng-x]").value = "0";
  elements.get("[data-ng-beta]").value = "1";
  elements.get("[data-ng-plot]").setAttribute("hidden", "");
  elements.get("[data-ng-loss-plot]").setAttribute("hidden", "");
  const buttons = ["sigmoid", "tanh", "relu", "swish"].map((name) => {
    const button = new Element(); button.dataset.ngFunction = name; return button;
  });
  const controls = [new Element(), new Element()];
  const root = {
    querySelector: (selector) => { assert.ok(elements.has(selector), "Unknown element " + selector); return elements.get(selector); },
    querySelectorAll: (selector) => selector === "[data-ng-function]" ? buttons : controls
  };
  return { root, elements, buttons, controls, get: (name) => elements.get("[data-ng-" + name + "]") };
}

test("data validates and samples clamp correctly", () => {
  assert.equal(demo.validateData(data), data);
  assert.equal(demo.sampleIndex(0, data.x), 80);
  assert.equal(demo.sampleIndex(-50, data.x), 0);
  assert.equal(demo.sampleIndex(50, data.x), 160);
  assert.throws(() => demo.sampleIndex("bad", data.x));
  const bad = structuredClone(data); bad.curves.sigmoid.values[1] = NaN;
  assert.throws(() => demo.validateData(bad));
});

test("every graph has finite, in-bounds coordinates", () => {
  for (const curve of Object.values(data.curves)) {
    const graph = demo.chartGeometry(data.x, curve);
    assert.ok(!/NaN|Infinity/.test(graph.values + graph.derivatives));
    assert.equal((graph.values.match(/L/g) || []).length, 160);
    assert.equal((graph.derivatives.match(/M/g) || []).length, curve.name === "relu" ? 2 : 1);
    curve.values.concat(curve.derivatives).forEach((value) => assert.ok(graph.y(value) >= 24 && graph.y(value) <= 200));
  }
});

test("function buttons, slider, beta and Python stay in agreement", () => {
  const view = fixture();
  demo.initExplorer(view.root, data);
  assert.equal(view.get("value").textContent, "0.5000");
  assert.equal(view.get("slope").textContent, "0.2500");
  assert.ok(view.controls.every((element) => element.hidden === false));
  assert.equal(view.get("fallback").hidden, true);
  assert.equal(view.get("plot").hidden, false);
  assert.equal(view.get("plot").getAttribute("hidden"), undefined);
  view.buttons[2].fire("click");
  assert.equal(view.get("value").textContent, "0.0000");
  assert.equal(view.get("slope").textContent, "0.5000");
  assert.match(view.get("note").textContent, /not differentiable/);
  assert.equal(view.buttons[2].getAttribute("aria-pressed"), "true");
  view.get("x").value = "-2"; view.get("x").fire("input");
  assert.equal(view.get("slope").textContent, "0.0000");
  assert.match(view.get("x").getAttribute("aria-valuetext"), /x -2.00/);
  view.buttons[3].fire("click");
  view.get("beta").value = "2"; view.get("beta").fire("change");
  assert.equal(view.get("beta-field").hidden, false);
  assert.match(view.get("code").textContent, /params=\{"beta": 2.0\}/);
  assert.equal(view.get("value").textContent, demo.formatNumber(data.curves["swish-2"].values[40]));
});

test("training tabs support arrows, Home, End and correct ARIA states", () => {
  const tabs = [new Element(), new Element()];
  const panels = [new Element(), new Element()];
  tabs.forEach((tab, i) => tab.setAttribute("aria-controls", "panel-" + i));
  const tablist = new Element(); tablist.querySelectorAll = () => tabs;
  const root = { querySelector: (selector) => selector === "[data-ng-tabs]" ? tablist : panels[Number(selector.slice(-1))] };
  demo.initTabs(root);
  assert.equal(tablist.hidden, false);
  assert.equal(panels[0].hidden, false);
  assert.equal(panels[1].hidden, true);
  let prevented = false;
  tabs[0].fire("keydown", { key: "ArrowRight", preventDefault: () => { prevented = true; } });
  assert.equal(prevented, true);
  assert.equal(tabs[1].getAttribute("aria-selected"), "true");
  assert.equal(tabs[1].tabIndex, 0);
  assert.equal(tabs[1].focused, true);
  assert.equal(panels[1].hidden, false);
  tabs[1].fire("keydown", { key: "Home", preventDefault() {} });
  assert.equal(tabs[0].getAttribute("aria-selected"), "true");
  tabs[0].fire("keydown", { key: "End", preventDefault() {} });
  assert.equal(tabs[1].getAttribute("aria-selected"), "true");
});

test("recorded loss visualization uses every epoch", () => {
  const view = fixture(); demo.initTraining(view.root, data);
  const line = view.get("loss-line").getAttribute("d");
  assert.equal((line.match(/L/g) || []).length, data.training.epochs - 1);
  assert.ok(!/NaN|Infinity/.test(line));
  assert.equal(view.get("loss-plot").hidden, false);
  assert.equal(view.get("loss-plot").getAttribute("hidden"), undefined);
});

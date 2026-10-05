import assert from "node:assert/strict";
import vm from "node:vm";
import { access, readFile } from "node:fs/promises";

const context = { window: {} };
vm.runInNewContext(await readFile(new URL("./dist/site-data.js", import.meta.url), "utf8"), context);
const { samples, models } = context.window.SITE_DATA;

function predict(model, features) {
  const vector = model.features.map((name) => features[name]);
  const totals = Array(model.classes.length).fill(0);
  for (const tree of model.trees) {
    let node = 0;
    while (tree.l[node] !== -1) node = vector[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
    tree.v[node].forEach((value, index) => totals[index] += value);
  }
  return Object.fromEntries(model.classes.map((name, index) => [name, totals[index] / model.trees.length]));
}

for (const sample of Object.values(samples)) {
  for (const task of ["binary", "multiclass"]) {
    const actual = predict(models[task], sample.features);
    for (const name of models[task].classes) assert.ok(Math.abs(actual[name] - sample[task][name]) < 1e-6);
  }
}

await access(new URL("./dist/demo/index.html", import.meta.url));
await access(new URL("./dist/demo/demo.css", import.meta.url));
await access(new URL("./dist/demo/demo.js", import.meta.url));
const home = await readFile(new URL("./dist/index.html", import.meta.url), "utf8");
assert.match(home, /href="demo\/"/);

console.log(`Verified ${Object.keys(samples).length} samples, both forests, and the /demo route.`);

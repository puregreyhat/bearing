(() => {
  'use strict';

  const DATA = window.SITE_DATA;
  const $ = (selector, root = document) => root.querySelector(selector);
  const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  const coarsePointer = matchMedia('(pointer: coarse)').matches;
  const labels = { healthy: 'Healthy', inner: 'Inner Race', ball: 'Ball', outer: 'Outer Race' };

  // Signature background: one canvas, fixed-size cells, and only nearby fills.
  const grid = $('#grid');
  const gridContext = grid.getContext('2d');
  const pointer = { x: -9999, y: -9999 };
  let gridScale = devicePixelRatio || 1;

  function sizeGrid() {
    gridScale = Math.min(devicePixelRatio || 1, 2);
    grid.width = innerWidth * gridScale;
    grid.height = innerHeight * gridScale;
    gridContext.setTransform(gridScale, 0, 0, gridScale, 0, 0);
  }

  function drawGrid() {
    const size = innerWidth < 700 ? 18 : 22;
    gridContext.clearRect(0, 0, innerWidth, innerHeight);
    gridContext.strokeStyle = '#d8d3c7';
    gridContext.lineWidth = .55;
    gridContext.beginPath();
    for (let x = 0; x < innerWidth; x += size) { gridContext.moveTo(x, 0); gridContext.lineTo(x, innerHeight); }
    for (let y = 0; y < innerHeight; y += size) { gridContext.moveTo(0, y); gridContext.lineTo(innerWidth, y); }
    gridContext.stroke();
    if (!coarsePointer && !reducedMotion) {
      const radius = 105;
      for (let x = Math.floor((pointer.x - radius) / size) * size; x < pointer.x + radius; x += size) {
        for (let y = Math.floor((pointer.y - radius) / size) * size; y < pointer.y + radius; y += size) {
          const distance = Math.hypot(x + size / 2 - pointer.x, y + size / 2 - pointer.y);
          const opacity = Math.max(0, 1 - distance / radius) * .18;
          if (opacity) {
            gridContext.fillStyle = `rgba(159,85,56,${opacity})`;
            gridContext.fillRect(x + 1, y + 1, size - 2, size - 2);
          }
        }
      }
    }
    requestAnimationFrame(drawGrid);
  }

  addEventListener('resize', sizeGrid);
  addEventListener('pointermove', event => { pointer.x = event.clientX; pointer.y = event.clientY; }, { passive: true });
  sizeGrid();
  drawGrid();

  function prepareCanvas(canvas) {
    const ratio = Math.min(devicePixelRatio || 1, 2);
    const rect = canvas.getBoundingClientRect();
    const width = Math.max(1, Math.round(rect.width));
    const height = Math.max(1, Math.round(rect.height));
    if (canvas.width !== width * ratio || canvas.height !== height * ratio) {
      canvas.width = width * ratio;
      canvas.height = height * ratio;
    }
    const context = canvas.getContext('2d');
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width, height };
  }

  function chartGrid(context, width, height) {
    context.strokeStyle = 'rgba(36,35,31,.12)';
    context.lineWidth = 1;
    context.beginPath();
    for (let x = 0; x <= width; x += width / 10) { context.moveTo(x, 0); context.lineTo(x, height); }
    for (let y = 0; y <= height; y += height / 8) { context.moveTo(0, y); context.lineTo(width, y); }
    context.stroke();
  }

  function drawSeries(canvas, values, options = {}) {
    const { context, width, height } = prepareCanvas(canvas);
    context.clearRect(0, 0, width, height);
    chartGrid(context, width, height);
    const transformed = options.log ? values.map(value => Math.log1p(Math.max(0, value))) : values;
    const max = options.symmetric ? Math.max(...transformed.map(Math.abs), 1e-12) : Math.max(...transformed, 1e-12);
    const min = options.symmetric ? -max : 0;
    const padding = 22;
    context.strokeStyle = options.color || '#9f5538';
    context.lineWidth = options.lineWidth || 1.5;
    context.beginPath();
    transformed.forEach((value, index) => {
      const x = padding + index / Math.max(1, transformed.length - 1) * (width - padding * 2);
      const y = padding + (max - value) / Math.max(1e-12, max - min) * (height - padding * 2);
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.stroke();
    if (options.symmetric) {
      context.strokeStyle = 'rgba(36,35,31,.25)';
      context.setLineDash([4, 5]);
      context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
      context.setLineDash([]);
    }
  }

  // Hero trace uses a real CWRU window; movement is only a scrolling view.
  const heroCanvas = $('#hero-wave');
  const heroValues = DATA.samples.healthy.wave;
  let heroOffset = 0;
  function drawHero() {
    const rolled = heroValues.map((_, index) => heroValues[(index + heroOffset) % heroValues.length]);
    drawSeries(heroCanvas, rolled, { symmetric: true, lineWidth: 1.7 });
    if (!reducedMotion) heroOffset = (heroOffset + 1) % heroValues.length;
    requestAnimationFrame(drawHero);
  }
  drawHero();

  const waveCanvas = $('#wave-chart');
  const fftCanvas = $('#fft-chart');
  let currentSample = 'healthy';
  let activeWave = DATA.samples.healthy.wave;
  let activeSpectrum = DATA.samples.healthy.spectrum;

  function formatValue(value, digits = 4) {
    const magnitude = Math.abs(value);
    if (magnitude !== 0 && (magnitude >= 10000 || magnitude < .001)) return value.toExponential(3);
    return value.toFixed(digits);
  }

  function renderFeatures(sample) {
    const feature = sample.features;
    const fields = [
      ['RMS', 'RMS', 5], ['STD', 'Std', 5], ['PEAK', 'Peak', 5], ['KURTOSIS', 'Kurtosis', 3],
      ['CREST FACTOR', 'Crest_Factor', 3], ['FFT PEAK', 'FFT_Peak_Frequency', 1],
      ['SPECTRAL CENTROID', 'Spectral_Centroid', 1], ['SPECTRAL ENTROPY', 'Spectral_Entropy', 3],
    ];
    $('#feature-readouts').innerHTML = fields.map(([label, key, digits]) =>
      `<div class="readout"><span>${label}</span><b>${formatValue(feature[key], digits)}${key.includes('Frequency') || key === 'Spectral_Centroid' ? '<small> Hz</small>' : ''}</b></div>`
    ).join('');
    const energies = Array.from({ length: 8 }, (_, index) => feature[`Band_Energy_${index + 1}`]);
    const max = Math.max(...energies, 1e-12);
    $('#energy-bands').innerHTML = energies.map((value, index) => `<div style="--h:${Math.max(2, value / max * 100)}%" title="${formatValue(value)}"><span>B${index + 1}</span></div>`).join('');
  }

  function selectSignal(name) {
    currentSample = name;
    const sample = DATA.samples[name];
    activeWave = sample.wave;
    activeSpectrum = sample.spectrum;
    drawSeries(waveCanvas, activeWave, { symmetric: true });
    drawSeries(fftCanvas, activeSpectrum, { log: true, color: '#b07a32' });
    $('#wave-file').textContent = `FILE ${String(sample.file).padStart(3, '0')}`;
    $('#feature-class').textContent = labels[name].toUpperCase();
    renderFeatures(sample);
    $$('[data-sample]').forEach(button => button.classList.toggle('active', button.dataset.sample === name));
  }

  $$('[data-sample]').forEach(button => button.addEventListener('click', () => selectSignal(button.dataset.sample)));
  selectSignal('healthy');

  function attachCrosshair(canvas, getValues, readout, unit) {
    canvas.addEventListener('pointermove', event => {
      const rect = canvas.getBoundingClientRect();
      const values = getValues();
      const index = Math.max(0, Math.min(values.length - 1, Math.round((event.clientX - rect.left) / rect.width * (values.length - 1))));
      const x = index / Math.max(1, values.length - 1) * rect.width;
      readout.style.left = `${x}px`;
      readout.style.top = `${event.clientY - rect.top}px`;
      readout.style.opacity = '1';
      readout.textContent = unit === 'Hz' ? `${Math.round(index / (values.length - 1) * 6000)} Hz · ${formatValue(values[index], 3)}` : `${(index / (values.length - 1) * .1707).toFixed(4)} s · ${formatValue(values[index], 4)}`;
    });
    canvas.addEventListener('pointerleave', () => { readout.style.opacity = '0'; });
  }
  attachCrosshair(waveCanvas, () => activeWave, $('#wave-readout'), 's');
  attachCrosshair(fftCanvas, () => activeSpectrum, $('#fft-readout'), 'Hz');

  function mean(values) { return values.reduce((sum, value) => sum + value, 0) / values.length; }

  function moments(values) {
    const average = mean(values);
    let m2 = 0, m3 = 0, m4 = 0;
    for (const value of values) {
      const delta = value - average;
      const square = delta * delta;
      m2 += square; m3 += square * delta; m4 += square * square;
    }
    m2 /= values.length; m3 /= values.length; m4 /= values.length;
    if (m2 === 0) return { mean: average, std: 0, skew: 0, kurtosis: 0 };
    return { mean: average, std: Math.sqrt(m2), skew: m3 / Math.pow(m2, 1.5), kurtosis: m4 / (m2 * m2) };
  }

  function fftMagnitudes(values) {
    const n = values.length;
    if ((n & (n - 1)) !== 0) throw new Error('FFT window length must be a power of two.');
    const real = new Float64Array(n);
    const imag = new Float64Array(n);
    let hannSum = 0;
    for (let index = 0; index < n; index++) {
      const hann = .5 - .5 * Math.cos(2 * Math.PI * index / (n - 1));
      real[index] = values[index] * hann;
      hannSum += hann;
    }
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { [real[i], real[j]] = [real[j], real[i]]; [imag[i], imag[j]] = [imag[j], imag[i]]; }
    }
    for (let length = 2; length <= n; length <<= 1) {
      const angle = -2 * Math.PI / length;
      const wrBase = Math.cos(angle), wiBase = Math.sin(angle);
      for (let start = 0; start < n; start += length) {
        let wr = 1, wi = 0;
        for (let offset = 0; offset < length / 2; offset++) {
          const even = start + offset, odd = even + length / 2;
          const tr = wr * real[odd] - wi * imag[odd];
          const ti = wr * imag[odd] + wi * real[odd];
          real[odd] = real[even] - tr; imag[odd] = imag[even] - ti;
          real[even] += tr; imag[even] += ti;
          const nextWr = wr * wrBase - wi * wiBase;
          wi = wr * wiBase + wi * wrBase; wr = nextWr;
        }
      }
    }
    const magnitudes = Array.from({ length: n / 2 + 1 }, (_, index) => Math.hypot(real[index], imag[index]) * 2 / hannSum);
    magnitudes[0] = 0;
    return magnitudes;
  }

  function extractFeatures(windowValues) {
    if (windowValues.length !== 2048) throw new Error('Each model window must contain exactly 2048 samples.');
    const stats = moments(windowValues);
    if (!stats.std) throw new Error('Signal has no variance and cannot be diagnosed.');
    const absolute = windowValues.map(Math.abs);
    const rms = Math.sqrt(mean(windowValues.map(value => value * value)));
    const meanAbs = mean(absolute);
    const sqrtAbsMean = mean(absolute.map(Math.sqrt));
    const peak = Math.max(...absolute);
    const magnitudes = fftMagnitudes(windowValues);
    const power = magnitudes.map(value => value * value);
    const totalPower = power.reduce((sum, value) => sum + value, 0);
    const probabilities = power.map(value => value / totalPower);
    const frequencyStep = 12000 / windowValues.length;
    const centroid = probabilities.reduce((sum, value, index) => sum + index * frequencyStep * value, 0);
    const spread = Math.sqrt(probabilities.reduce((sum, value, index) => sum + Math.pow(index * frequencyStep - centroid, 2) * value, 0));
    const entropy = probabilities.reduce((sum, value) => value > 0 ? sum - value * Math.log2(value) : sum, 0);
    const spectralKurtosis = moments(power).kurtosis;
    const peakIndex = magnitudes.indexOf(Math.max(...magnitudes));
    const features = {
      RMS: rms, Std: stats.std, Peak: peak, Kurtosis: stats.kurtosis,
      Crest_Factor: peak / rms, FFT_Peak_Frequency: peakIndex * frequencyStep,
      Peak_to_Peak: Math.max(...windowValues) - Math.min(...windowValues), Mean: stats.mean,
      Skewness: stats.skew, Shape_Factor: rms / meanAbs, Impulse_Factor: peak / meanAbs,
      Margin_Factor: peak / (sqrtAbsMean * sqrtAbsMean), FFT_Peak_Amplitude: magnitudes[peakIndex],
      Spectral_Centroid: centroid, Spectral_Spread: spread, Spectral_Entropy: entropy,
      Spectral_Kurtosis: spectralKurtosis,
    };
    let offset = 0;
    const base = Math.floor(power.length / 8), extra = power.length % 8;
    for (let band = 0; band < 8; band++) {
      const length = base + (band < extra ? 1 : 0);
      features[`Band_Energy_${band + 1}`] = power.slice(offset, offset + length).reduce((sum, value) => sum + value, 0);
      offset += length;
    }
    return { features, magnitudes };
  }

  function predictForest(model, features) {
    const vector = model.features.map(name => features[name]);
    const totals = Array(model.classes.length).fill(0);
    for (const tree of model.trees) {
      let node = 0;
      while (tree.l[node] !== -1) node = vector[tree.f[node]] <= tree.t[node] ? tree.l[node] : tree.r[node];
      tree.v[node].forEach((value, index) => { totals[index] += value; });
    }
    return Object.fromEntries(model.classes.map((name, index) => [name, totals[index] / model.trees.length]));
  }

  function renderDiagnostic(binary, multiclass, windows, source) {
    const condition = Object.entries(binary).sort((a, b) => b[1] - a[1])[0];
    const fault = Object.entries(multiclass).sort((a, b) => b[1] - a[1])[0];
    const health = Math.round((binary.Healthy || 0) * 100);
    $('#condition-output').textContent = condition[0].toUpperCase();
    $('#condition-output').style.color = condition[0] === 'Healthy' ? '#aabd9f' : '#d48375';
    $('#confidence-output').textContent = `${(condition[1] * 100).toFixed(1)}% confidence`;
    $('#fault-output').textContent = fault[0].replace('_', ' ').toUpperCase();
    $('#health-output').textContent = health;
    $('#health-dial').style.setProperty('--health', `${health}%`);
    $('#window-output').textContent = `${String(windows).padStart(2, '0')} WINDOW${windows === 1 ? '' : 'S'}`;
    $('#input-state').textContent = source;
    $('#probability-bars').innerHTML = Object.entries(multiclass).map(([name, probability]) =>
      `<div><span>${name.replace('_', ' ')}</span><i style="--p:${probability * 100}%"></i><b>${(probability * 100).toFixed(1)}%</b></div>`
    ).join('');
  }

  function diagnoseSample(name) {
    const sample = DATA.samples[name];
    renderDiagnostic(sample.binary, sample.multiclass, 1, `CWRU / ${sample.file}`);
    $$('[data-diagnostic]').forEach(button => button.classList.toggle('active', button.dataset.diagnostic === name));
    $('#diagnostic-error').textContent = '';
  }

  $$('[data-diagnostic]').forEach(button => button.addEventListener('click', () => diagnoseSample(button.dataset.diagnostic)));
  diagnoseSample('healthy');

  $('#signal-upload').addEventListener('change', async event => {
    const file = event.target.files[0];
    if (!file) return;
    try {
      $('#diagnostic-error').textContent = '';
      const tokens = (await file.text()).trim().split(/[\s,;]+/);
      const signal = tokens.map(Number);
      if (signal.some(value => !Number.isFinite(value))) throw new Error('CSV must contain numeric values only.');
      const count = Math.floor(signal.length / 2048);
      if (!count) throw new Error('At least 2048 samples are required.');
      const binaryTotal = Object.fromEntries(DATA.models.binary.classes.map(name => [name, 0]));
      const multiclassTotal = Object.fromEntries(DATA.models.multiclass.classes.map(name => [name, 0]));
      let first;
      for (let index = 0; index < count; index++) {
        const result = extractFeatures(signal.slice(index * 2048, (index + 1) * 2048));
        if (!first) first = result;
        const binary = predictForest(DATA.models.binary, result.features);
        const multiclass = predictForest(DATA.models.multiclass, result.features);
        Object.keys(binaryTotal).forEach(name => { binaryTotal[name] += binary[name] / count; });
        Object.keys(multiclassTotal).forEach(name => { multiclassTotal[name] += multiclass[name] / count; });
      }
      renderDiagnostic(binaryTotal, multiclassTotal, count, 'UPLOADED CSV');
      $$('[data-diagnostic]').forEach(button => button.classList.remove('active'));
      activeWave = signal.slice(0, 2048).filter((_, index) => index % 4 === 0);
      activeSpectrum = first.magnitudes.filter((_, index) => index % 4 === 0);
      drawSeries(waveCanvas, activeWave, { symmetric: true });
      drawSeries(fftCanvas, activeSpectrum, { log: true, color: '#b07a32' });
      renderFeatures({ features: first.features });
      $('#feature-class').textContent = 'UPLOADED SIGNAL';
      $('#wave-file').textContent = 'UPLOADED CSV';
    } catch (error) {
      $('#diagnostic-error').textContent = error.message;
    }
  });

  const revealObserver = new IntersectionObserver(entries => {
    entries.forEach(entry => { if (entry.isIntersecting) { entry.target.classList.add('visible'); revealObserver.unobserve(entry.target); } });
  }, { threshold: .08, rootMargin: '0px 0px -40px' });
  $$('.reveal').forEach(element => revealObserver.observe(element));

  let resizeFrame;
  addEventListener('resize', () => {
    cancelAnimationFrame(resizeFrame);
    resizeFrame = requestAnimationFrame(() => selectSignal(currentSample));
  }, { passive: true });

  // Small public surface used by the repository smoke check.
  window.BearingLab = { extractFeatures, predictForest, fftMagnitudes };
})();

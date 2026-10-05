(() => {
  const data = window.SITE_DATA;
  const $ = selector => document.querySelector(selector);
  const $$ = selector => [...document.querySelectorAll(selector)];
  const labels = { healthy: 'Normal', inner: 'Inner Race', ball: 'Ball', outer: 'Outer Race' };
  const stages = [
    ['STAGE 01 / INPUT', 'Read numeric acceleration values', 'The browser reads one numeric channel from the CSV. Commas, semicolons, spaces and line breaks are accepted separators.'],
    ['STAGE 02 / VALIDATION', 'Reject malformed input early', 'Every token must be numeric and at least 2,048 values must exist. The website expects a signal already sampled at 12 kHz.'],
    ['STAGE 03 / WINDOWING', 'Split into complete 2,048-sample blocks', 'Each window covers 0.1707 seconds. No overlap is used; any incomplete final block is ignored.'],
    ['STAGE 04 / FEATURES', 'Turn each waveform into 25 measurements', 'Time statistics, impact ratios, FFT shape and eight frequency-band energies describe the signal without using file or load labels.'],
    ['STAGE 05 / CLASSIFICATION', 'Send the same features through two forests', 'Two exported 250-tree Random Forests estimate bearing condition and fault type for every complete window.'],
    ['STAGE 06 / RESULT', 'Average window probabilities', 'The final display reports Healthy or Faulty, the most likely fault class and a Health Index equal to the Healthy probability × 100.']
  ];
  const reducedMotion = matchMedia('(prefers-reduced-motion: reduce)').matches;
  let selected = 'healthy', stage = 0, timer;

  function sizeCanvas(canvas) {
    const ratio = devicePixelRatio || 1;
    canvas.width = Math.round(canvas.clientWidth * ratio);
    canvas.height = Math.round(canvas.clientHeight * ratio);
    return [canvas.getContext('2d'), canvas.width, canvas.height, ratio];
  }

  function drawWave() {
    const canvas = $('#wave');
    const [context, width, height, ratio] = sizeCanvas(canvas);
    const values = data.samples[selected].wave;
    const max = Math.max(...values.map(Math.abs)) || 1;
    context.clearRect(0, 0, width, height);
    context.strokeStyle = 'rgba(243,240,231,.08)';
    context.lineWidth = ratio;
    for (let x = 0; x <= width; x += 64 * ratio) { context.beginPath(); context.moveTo(x, 0); context.lineTo(x, height); context.stroke(); }
    for (let y = 0; y <= height; y += 64 * ratio) { context.beginPath(); context.moveTo(0, y); context.lineTo(width, y); context.stroke(); }
    context.strokeStyle = '#d79a7e';
    context.lineWidth = 1.4 * ratio;
    context.beginPath();
    values.forEach((value, index) => {
      const x = index / (values.length - 1) * width;
      const y = height / 2 - value / max * height * .4;
      index ? context.lineTo(x, y) : context.moveTo(x, y);
    });
    context.stroke();
  }

  function renderSample(name) {
    selected = name;
    const sample = data.samples[name];
    const binary = sample.binary;
    const fault = sample.multiclass;
    const condition = Object.entries(binary).sort((a, b) => b[1] - a[1])[0];
    const faultType = Object.entries(fault).sort((a, b) => b[1] - a[1])[0];
    $('#file-label').textContent = `FILE ${sample.file}`;
    $('#condition').textContent = condition[0].toUpperCase();
    $('#condition').style.color = condition[0] === 'Healthy' ? '#aabd9f' : '#d79a7e';
    $('#condition-confidence').textContent = `${(condition[1] * 100).toFixed(1)}% confidence`;
    $('#fault').textContent = faultType[0].replace('_', ' ').toUpperCase();
    $('#probabilities').innerHTML = Object.entries(fault).map(([key, probability]) => `<div><span>${key.replace('_', ' ')}</span><i style="--p:${probability * 100}%"></i><b>${(probability * 100).toFixed(1)}%</b></div>`).join('');
    $$('[data-sample]').forEach(button => button.classList.toggle('active', button.dataset.sample === name));
    drawWave();
  }

  function showStage(index) {
    stage = index % stages.length;
    const [kicker, title, body] = stages[stage];
    $('#stage-kicker').textContent = kicker;
    $('#stage-title').textContent = title;
    $('#stage-body').textContent = body;
    $$('#stages li').forEach((item, itemIndex) => {
      item.classList.toggle('active', itemIndex === stage);
      item.classList.toggle('done', itemIndex < stage);
    });
  }

  function play() {
    clearInterval(timer);
    showStage(0);
    $('#play').textContent = 'Playing stages';
    if (reducedMotion) { showStage(5); $('#play').textContent = 'Replay stages'; return; }
    timer = setInterval(() => {
      if (stage === stages.length - 1) { clearInterval(timer); $('#play').textContent = 'Replay stages'; return; }
      showStage(stage + 1);
    }, 1700);
  }

  function drawSampling(canvas, count, phase) {
    const [context, width, height, ratio] = sizeCanvas(canvas);
    context.clearRect(0, 0, width, height);
    context.strokeStyle = 'rgba(36,35,31,.14)';
    context.lineWidth = ratio;
    context.beginPath(); context.moveTo(0, height / 2); context.lineTo(width, height / 2); context.stroke();
    const points = Array.from({ length: count }, (_, index) => {
      const x = index / (count - 1) * width;
      const angle = index / (count - 1) * Math.PI * 6 + phase;
      return [x, height / 2 - (Math.sin(angle) + .28 * Math.sin(angle * 3.2)) * height * .28];
    });
    context.strokeStyle = '#9f5538'; context.lineWidth = 1.5 * ratio; context.beginPath();
    points.forEach(([x, y], index) => index ? context.lineTo(x, y) : context.moveTo(x, y)); context.stroke();
    context.fillStyle = '#9f5538';
    points.forEach(([x, y]) => { context.beginPath(); context.arc(x, y, 2.2 * ratio, 0, Math.PI * 2); context.fill(); });
  }

  function animateSampling(time = 0) {
    const phase = reducedMotion ? 0 : time / 1300;
    drawSampling($('#wave48'), 64, phase);
    drawSampling($('#wave12'), 16, phase);
    if (!reducedMotion) requestAnimationFrame(animateSampling);
  }

  $$('[data-sample]').forEach(button => button.addEventListener('click', () => renderSample(button.dataset.sample)));
  $('#play').addEventListener('click', play);
  addEventListener('resize', () => { drawWave(); if (reducedMotion) animateSampling(); });
  renderSample('healthy');
  showStage(0);
  requestAnimationFrame(animateSampling);
})();

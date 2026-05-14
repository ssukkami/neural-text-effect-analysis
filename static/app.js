document.addEventListener('DOMContentLoaded', () => {
  initCursorAura();
  initTerminalActions();
  initABActions();
  initABDiagnosticsModal();
  initDatasetActions();
});

function initCursorAura() {
  const aura = document.querySelector('.cursor-aura');
  if (!aura) return;

  aura.style.left = `${window.innerWidth / 2}px`;
  aura.style.top = `${window.innerHeight / 2}px`;

  document.addEventListener('mousemove', (e) => {
    requestAnimationFrame(() => {
      aura.style.left = `${e.clientX}px`;
      aura.style.top = `${e.clientY}px`;
    });
  });

  document.addEventListener('mousedown', () => {
    aura.style.transform = 'translate(-50%, -50%) scale(1.08)';
  });

  document.addEventListener('mouseup', () => {
    aura.style.transform = 'translate(-50%, -50%) scale(1)';
  });
}

function switchTab(tabId, event) {
  document.querySelectorAll('.view').forEach((v) => v.classList.remove('active-view'));
  const tab = document.getElementById(tabId);
  if (tab) tab.classList.add('active-view');

  document.querySelectorAll('.nav-btn').forEach((b) => b.classList.remove('active'));
  if (event && event.currentTarget) {
    event.currentTarget.classList.add('active');
  }
}
window.switchTab = switchTab;

function safeNum(x, d = 0) {
  return typeof x === 'number' && Number.isFinite(x) ? x : d;
}

function safeText(x, d = '--') {
  if (x === null || x === undefined) return d;
  const s = String(x);
  return s.length ? s : d;
}

function clamp01to100(x) {
  const v = safeNum(x, 0);
  return Math.max(0, Math.min(100, v));
}

function setBar(id, v) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${clamp01to100(v)}%`;
}

function setVal(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(txt);
}

function setText(id, txt) {
  const el = document.getElementById(id);
  if (el) el.textContent = String(txt);
}

function setWidth(id, pct) {
  const el = document.getElementById(id);
  if (el) el.style.width = `${Math.max(0, Math.min(100, pct))}%`;
}

function animateValue(element, start, end, duration, suffix = '') {
  if (!element) return;

  const finalValue = typeof end === 'number' && Number.isFinite(end) ? end : 0;
  let startTimestamp = null;

  const step = (ts) => {
    if (!startTimestamp) startTimestamp = ts;
    const progress = Math.min((ts - startTimestamp) / duration, 1);
    const value = progress * (finalValue - start) + start;
    element.innerHTML = `${value % 1 !== 0 ? value.toFixed(1) : Math.floor(value)}${suffix}`;

    if (progress < 1) {
      window.requestAnimationFrame(step);
    }
  };

  window.requestAnimationFrame(step);
}

const uiCharts = {
  termRadar: null,
  abCompareBars: null,
  abCompareSignals: null
};

function destroyChart(chartRefName) {
  if (uiCharts[chartRefName]) {
    try {
      uiCharts[chartRefName].destroy();
    } catch (_) {}
    uiCharts[chartRefName] = null;
  }
}

function renderSignalsChips(data, containerId) {
  const box = document.getElementById(containerId);
  if (!box) return;

  const profile = data?.auxiliary_analysis?.profile || {};
  const signals = profile?.signals || {};
  const style = profile?.style || {};

  const items = [
    { k: 'CTA', v: !!signals.cta },
    { k: 'HASHTAGS', v: safeNum(style.hashtag_count, 0) },
    { k: 'QUESTIONS', v: safeNum(style.question_count, 0) },
    { k: 'MENTIONS', v: safeNum(style.mention_count, 0) },
    { k: 'EMOJIS', v: safeNum(style.emoji_count, 0) }
  ];

  box.innerHTML = items.map((it) => {
    const isOn = typeof it.v === 'boolean' ? it.v : safeNum(it.v, 0) > 0;
    const val = typeof it.v === 'boolean' ? (it.v ? 'YES' : 'NO') : it.v;
    return `<div class="chip ${isOn ? 'chip-on' : ''}">${it.k}: ${val}</div>`;
  }).join('');
}

function renderRadar(data) {
  if (typeof Chart === 'undefined') return;

  const canvas = document.getElementById('termMetricsRadar');
  if (!canvas) return;

  const aux = data?.auxiliary_analysis || {};
  const lq = clamp01to100(aux.language_quality_score);
  const af = clamp01to100(aux.audience_fit_score);
  const ep = clamp01to100(aux.engagement_potential_score);

  destroyChart('termRadar');

  uiCharts.termRadar = new Chart(canvas, {
    type: 'radar',
    data: {
      labels: ['LANG', 'AUD', 'ENG'],
      datasets: [{
        data: [lq, af, ep],
        backgroundColor: 'rgba(255, 209, 220, 0.45)',
        borderColor: '#0A0A0A',
        borderWidth: 2,
        pointBackgroundColor: '#0A0A0A',
        pointRadius: 3
      }]
    },
    options: {
      responsive: true,
      maintainAspectRatio: false,
      plugins: { legend: { display: false } },
      scales: {
        r: {
          beginAtZero: true,
          suggestedMax: 100,
          ticks: { display: false }
        }
      }
    }
  });
}

function renderDashboardMetrics(data) {
  const aux = data?.auxiliary_analysis || {};
  const profile = aux?.profile || {};
  const r = profile?.readability || {};
  const s = profile?.style || {};

  const lq = clamp01to100(aux.language_quality_score);
  const af = clamp01to100(aux.audience_fit_score);
  const ep = clamp01to100(aux.engagement_potential_score);

  setVal('m-lang-val', `${lq.toFixed(1)}/100`);
  setVal('m-aud-val', `${af.toFixed(1)}/100`);
  setVal('m-eng-val', `${ep.toFixed(1)}/100`);

  setBar('m-lang-bar', lq);
  setBar('m-aud-bar', af);
  setBar('m-eng-bar', ep);

  const stageAUsable = !!data?.stage_a_status?.usable;
  setVal('m-stagea-val', stageAUsable ? 'YES' : 'NO');

  const pill = document.getElementById('m-stagea-pill');
  if (pill) {
    pill.textContent = stageAUsable ? 'USABLE' : 'TEXT-ONLY';
    pill.style.background = stageAUsable ? '#0A0A0A' : '#FFFFFF';
    pill.style.color = stageAUsable ? '#FFFFFF' : '#0A0A0A';
  }

  setVal('t-flesch', safeNum(r.flesch, 0).toFixed(1));
  setVal('t-fk', safeNum(r.fk_grade, 0).toFixed(2));
  setVal('t-asl', safeNum(s.avg_sentence_len, 0).toFixed(2));
  setVal('t-ttr', safeNum(s.lexical_diversity_ttr, 0).toFixed(3));
  setVal('t-wc', safeNum(s.word_count, 0));
  setVal('t-sc', safeNum(s.sentence_count, 0));

  renderRadar(data);
  renderSignalsChips(data, 'signals-chips');
}

function renderAdvice(containerId, recommendations) {
  const container = document.getElementById(containerId);
  if (!container) return;

  if (!recommendations || typeof recommendations !== 'object') {
    container.innerHTML = 'NO AUXILIARY RECOMMENDATIONS AVAILABLE';
    return;
  }

  const reasons = Array.isArray(recommendations.reasons) ? recommendations.reasons : [];
  const improvements = Array.isArray(recommendations.improvements) ? recommendations.improvements : [];

  let html = '';

  if (reasons.length) {
    html += '<div style="margin-bottom:5px; font-weight:900;">[OBSERVED SIGNALS]</div>';
    reasons.forEach((r) => {
      html += `<div class="advice-item">${String(r)}</div>`;
    });
  }

  if (improvements.length) {
    html += '<div style="margin-top:10px; margin-bottom:5px; font-weight:900;">[AUXILIARY IMPROVEMENTS]</div>';
    improvements.forEach((i) => {
      html += `<div class="advice-item">${String(i)}</div>`;
    });
  }

  container.innerHTML = html || 'NO AUXILIARY DEVIATIONS DETECTED';
}

function renderTerms(containerId, terms, limit = 6) {
  const box = document.getElementById(containerId);
  if (!box) return;

  if (Array.isArray(terms) && terms.length) {
    box.innerHTML = terms.slice(0, limit).map((p) => {
      const ph = String(p?.term ?? '').toUpperCase();
      const sc = safeNum(p?.score, 0);
      return `<div class="phrase-tag">${ph} <span class="score">[${sc}]</span></div>`;
    }).join('');
  } else {
    box.innerHTML = '<span style="font-size:0.8rem;">NO ATTENTION-DERIVED TERMS DETECTED</span>';
  }
}

function renderSentiment(prefix, pos, neu, neg) {
  setWidth(`${prefix}-bar-pos`, pos);
  setWidth(`${prefix}-bar-neu`, neu);
  setWidth(`${prefix}-bar-neg`, neg);

  setText(`${prefix}-lbl-pos`, `POSITIVE ${Math.round(pos)}%`);
  setText(`${prefix}-lbl-neu`, `NEUTRAL ${Math.round(neu)}%`);
  setText(`${prefix}-lbl-neg`, `NEGATIVE ${Math.round(neg)}%`);
}

async function postJSON(url, payload) {
  const res = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(payload || {})
  });

  let data = null;
  try {
    data = await res.json();
  } catch (_) {
    data = { ok: false, error: 'INVALID_JSON_RESPONSE' };
  }

  if (!data || data.ok === false) {
    const msg = data && data.error ? data.error : 'REQUEST_FAILED';
    throw new Error(msg);
  }

  return data;
}

function renderStageAStatus(data) {
  const box = document.getElementById('term-stagea');
  if (!box) return;

  const s = data?.stage_a_status || {};
  const missing = Array.isArray(s.missing_features) ? s.missing_features : [];
  const filled = Array.isArray(s.filled_with_zero) ? s.filled_with_zero : [];
  const unsupported = Array.isArray(s.unsupported_features) ? s.unsupported_features : [];

  let html = '';

  if (s.usable) {
    html += `<div class="advice-item">Stage A is usable. Context quality: ${safeText(s.quality, 'unknown')}.</div>`;
  } else {
    html += '<div class="advice-item">Stage A is not usable. Output is limited to text-only analysis.</div>';
  }

  if (missing.length) {
    html += `<div class="advice-item">Missing context features: ${missing.join(', ')}</div>`;
  }

  if (filled.length) {
    html += `<div class="advice-item">Zero-filled features (train-like fillna(0) behavior): ${filled.join(', ')}</div>`;
  }

  if (unsupported.length) {
    html += `<div class="advice-item">Unsupported inference features: ${unsupported.join(', ')}</div>`;
  }

  box.innerHTML = html;
}

function renderCentralDiagnostics(data) {
  const container = document.getElementById('ab-technical-compare');
  if (!container) return;

  const A = data?.A || {};
  const B = data?.B || {};
  const metric = data?.comparison?.metric || 'text_effect_relative_score';

  const scoreA = metric === 'engagement_relative_score'
    ? safeNum(A?.relative_scores?.engagement_relative_score, 0)
    : safeNum(A?.relative_scores?.text_effect_relative_score, 0);

  const scoreB = metric === 'engagement_relative_score'
    ? safeNum(B?.relative_scores?.engagement_relative_score, 0)
    : safeNum(B?.relative_scores?.text_effect_relative_score, 0);

  const teA = safeNum(A?.model_output?.text_residual_component, 0);
  const teB = safeNum(B?.model_output?.text_residual_component, 0);

  const fkA = safeNum(A?.auxiliary_analysis?.profile?.readability?.fk_grade, 0);
  const fkB = safeNum(B?.auxiliary_analysis?.profile?.readability?.fk_grade, 0);

  const semantic = [];
  const structure = [];
  const conclusion = [];

  if (Math.abs(scoreA - scoreB) < 0.2) {
    semantic.push('Primary scores are nearly identical at the current analysis scale.');
  } else if (scoreA > scoreB) {
    semantic.push(`Variant A has the higher primary score: ${scoreA.toFixed(1)} versus ${scoreB.toFixed(1)}.`);
  } else {
    semantic.push(`Variant B has the higher primary score: ${scoreB.toFixed(1)} versus ${scoreA.toFixed(1)}.`);
  }

  if (Math.abs(teA - teB) >= 0.05) {
    semantic.push(
      teA > teB
        ? 'Variant A has the stronger text residual component.'
        : 'Variant B has the stronger text residual component.'
    );
  } else {
    semantic.push('Text residual components are close to each other.');
  }

  if (Math.abs(fkA - fkB) >= 1) {
    structure.push(
      fkA < fkB
        ? 'Variant A is simpler by FK readability.'
        : 'Variant B is simpler by FK readability.'
    );
  }

  const styleA = A?.auxiliary_analysis?.profile?.style || {};
  const styleB = B?.auxiliary_analysis?.profile?.style || {};

  if (safeNum(styleA.hashtag_count, 0) !== safeNum(styleB.hashtag_count, 0)) {
    structure.push(`Hashtag count differs: A = ${safeNum(styleA.hashtag_count, 0)}, B = ${safeNum(styleB.hashtag_count, 0)}.`);
  }

  if (safeNum(styleA.emoji_count, 0) !== safeNum(styleB.emoji_count, 0)) {
    structure.push(`Emoji count differs: A = ${safeNum(styleA.emoji_count, 0)}, B = ${safeNum(styleB.emoji_count, 0)}.`);
  }

  conclusion.push(
    metric === 'engagement_relative_score'
      ? 'Both texts were compared under full prediction mode.'
      : 'Comparison was limited to text-only analysis because at least one variant lacked usable Stage A context.'
  );

  container.innerHTML = `
    <div class="ab-tech-group">
      <div class="ab-tech-title">PRIMARY INTERPRETATION</div>
      <div class="ab-tech-list">
        ${semantic.map(item => `<div class="ab-tech-item">${item}</div>`).join('')}
      </div>
    </div>

    <div class="ab-tech-group">
      <div class="ab-tech-title">STRUCTURAL DIFFERENCES</div>
      <div class="ab-tech-list">
        ${(structure.length ? structure : ['No strong structural differences were detected.'])
          .map(item => `<div class="ab-tech-item">${item}</div>`).join('')}
      </div>
    </div>

    <div class="ab-tech-group">
      <div class="ab-tech-title">TECHNICAL CONCLUSION</div>
      <div class="ab-tech-list">
        ${conclusion.map(item => `<div class="ab-tech-item">${item}</div>`).join('')}
      </div>
    </div>
  `;
}

function renderABCharts(data) {
  if (typeof Chart === 'undefined') return;

  const A = data?.A || {};
  const B = data?.B || {};
  const metric = data?.comparison?.metric || 'text_effect_relative_score';

  const scoreA = metric === 'engagement_relative_score'
    ? safeNum(A?.relative_scores?.engagement_relative_score, 0)
    : safeNum(A?.relative_scores?.text_effect_relative_score, 0);

  const scoreB = metric === 'engagement_relative_score'
    ? safeNum(B?.relative_scores?.engagement_relative_score, 0)
    : safeNum(B?.relative_scores?.text_effect_relative_score, 0);

  const barCanvas = document.getElementById('abCompareBars');
  if (barCanvas) {
    destroyChart('abCompareBars');

    uiCharts.abCompareBars = new Chart(barCanvas, {
      type: 'bar',
      data: {
        labels: ['PRIMARY SCORE', 'TEXT RESIDUAL', 'LANG QUALITY', 'AUD FIT', 'ENG POTENTIAL'],
        datasets: [
          {
            label: 'A',
            data: [
              scoreA,
              Math.max(0, Math.min(100, safeNum(A?.relative_scores?.text_effect_relative_score, 0))),
              safeNum(A?.auxiliary_analysis?.language_quality_score, 0),
              safeNum(A?.auxiliary_analysis?.audience_fit_score, 0),
              safeNum(A?.auxiliary_analysis?.engagement_potential_score, 0)
            ],
            backgroundColor: '#0A0A0A'
          },
          {
            label: 'B',
            data: [
              scoreB,
              Math.max(0, Math.min(100, safeNum(B?.relative_scores?.text_effect_relative_score, 0))),
              safeNum(B?.auxiliary_analysis?.language_quality_score, 0),
              safeNum(B?.auxiliary_analysis?.audience_fit_score, 0),
              safeNum(B?.auxiliary_analysis?.engagement_potential_score, 0)
            ],
            backgroundColor: '#FFD1DC'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: { y: { beginAtZero: true, max: 100 } }
      }
    });
  }

  const sigCanvas = document.getElementById('abCompareSignals');
  if (sigCanvas) {
    destroyChart('abCompareSignals');

    const styleA = A?.auxiliary_analysis?.profile?.style || {};
    const styleB = B?.auxiliary_analysis?.profile?.style || {};

    uiCharts.abCompareSignals = new Chart(sigCanvas, {
      type: 'bar',
      data: {
        labels: ['HASHTAGS', 'QUESTIONS', 'MENTIONS', 'EMOJIS', 'CTA'],
        datasets: [
          {
            label: 'A',
            data: [
              safeNum(styleA.hashtag_count, 0),
              safeNum(styleA.question_count, 0),
              safeNum(styleA.mention_count, 0),
              safeNum(styleA.emoji_count, 0),
              safeNum(styleA.cta_hits, 0)
            ],
            backgroundColor: '#0A0A0A'
          },
          {
            label: 'B',
            data: [
              safeNum(styleB.hashtag_count, 0),
              safeNum(styleB.question_count, 0),
              safeNum(styleB.mention_count, 0),
              safeNum(styleB.emoji_count, 0),
              safeNum(styleB.cta_hits, 0)
            ],
            backgroundColor: '#FFD1DC'
          }
        ]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { position: 'bottom' } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
      }
    });
  }
}

function clearDatasetUI() {
  setText('datasetStatus', 'WAITING');
  setText('ds-rows', '0');
  setText('ds-avg', '0');
  setText('ds-col-text', '--');
  setText('ds-col-metric', '--');
  setText('ds-metric-type', '--');
  setText('ds-valid-texts', '0');
  setText('ds-analyzed', '0');
  setText('ds-sampling', '--');

  if (window.dsCharts) {
    Object.values(window.dsCharts).forEach((chart) => {
      try { chart.destroy(); } catch (_) {}
    });
  }
  window.dsCharts = {};
}

function renderDatasetMeta(data) {
  const cols = data?.detected_cols || {};
  const metric = data?.metric_summary || {};
  const summary = data?.summary || {};
  const sampling = summary?.sampling || {};

  setText('ds-col-text', cols.text || 'NOT DETECTED');
  setText('ds-col-metric', metric.column || 'NOT DETECTED');
  setText('ds-metric-type', (metric.type || '--').toUpperCase());
  setText('ds-valid-texts', safeNum(summary.valid_text_rows, 0));
  setText('ds-analyzed', safeNum(summary.analyzed_rows, 0));

  const samplingText = sampling.used_sample
    ? `SAMPLED ${safeNum(sampling.sample_size, 0)} / LIMIT ${safeNum(sampling.sample_limit, 0)}`
    : 'FULL DATASET';

  setText('ds-sampling', samplingText);
}

function renderDatasetCharts(data) {
  if (typeof Chart === 'undefined') return;

  Chart.defaults.font.family = "'Inter', sans-serif";
  Chart.defaults.font.weight = '900';
  Chart.defaults.color = '#0A0A0A';

  const co = { responsive: true, maintainAspectRatio: false };
  if (!window.dsCharts) window.dsCharts = {};

  const analysis = data?.analysis || {};

  const impactCanvas = document.getElementById('impactChart');
  if (impactCanvas) {
    if (window.dsCharts.impact) {
      try { window.dsCharts.impact.destroy(); } catch (_) {}
    }

    const rows = Array.isArray(analysis.top_terms_by_frequency) ? analysis.top_terms_by_frequency : [];
    window.dsCharts.impact = new Chart(impactCanvas, {
      type: 'bar',
      data: {
        labels: rows.map(x => String(x.term || '').toUpperCase()),
        datasets: [{
          label: 'COUNT',
          data: rows.map(x => safeNum(x.count, 0)),
          backgroundColor: '#0A0A0A'
        }]
      },
      options: {
        ...co,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
      }
    });
  }

  const dist = analysis.sentiment_distribution || {};
  const sentCanvas = document.getElementById('sentimentDoughnut');
  if (sentCanvas) {
    if (window.dsCharts.sent) {
      try { window.dsCharts.sent.destroy(); } catch (_) {}
    }

    window.dsCharts.sent = new Chart(sentCanvas, {
      type: 'doughnut',
      data: {
        labels: ['Positive', 'Neutral', 'Negative'],
        datasets: [{
          data: [
            safeNum(dist.positive, 0),
            safeNum(dist.neutral, 0),
            safeNum(dist.negative, 0)
          ],
          backgroundColor: ['#FFD1DC', '#E0E0E0', '#0A0A0A']
        }]
      },
      options: {
        ...co,
        cutout: '70%',
        plugins: { legend: { position: 'bottom' } }
      }
    });
  }

  const radarCanvas = document.getElementById('topicsRadar');
  if (radarCanvas) {
    if (window.dsCharts.radar) {
      try { window.dsCharts.radar.destroy(); } catch (_) {}
    }

    const rows = Array.isArray(analysis.high_signal_terms) ? analysis.high_signal_terms : [];
    const vals = rows.map(x => Math.min(100, Math.abs(safeNum(x.mean_text_signal, 0)) * 100));

    window.dsCharts.radar = new Chart(radarCanvas, {
      type: 'radar',
      data: {
        labels: rows.map(x => String(x.term || '').toUpperCase()),
        datasets: [{
          data: vals,
          backgroundColor: 'rgba(255, 209, 220, 0.45)',
          borderColor: '#0A0A0A',
          borderWidth: 2,
          pointBackgroundColor: '#0A0A0A'
        }]
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: { legend: { display: false } },
        scales: {
          r: {
            beginAtZero: true,
            min: 0,
            max: 100,
            ticks: { display: false }
          }
        }
      }
    });
  }

  const lengthCanvas = document.getElementById('lengthDistributionChart');
  if (lengthCanvas) {
    if (window.dsCharts.lengths) {
      try { window.dsCharts.lengths.destroy(); } catch (_) {}
    }

    const ld = analysis.length_distribution || {};
    window.dsCharts.lengths = new Chart(lengthCanvas, {
      type: 'bar',
      data: {
        labels: ld.labels || [],
        datasets: [{
          data: ld.values || [],
          backgroundColor: '#FFD1DC',
          borderColor: '#0A0A0A',
          borderWidth: 1.5
        }]
      },
      options: {
        ...co,
        plugins: { legend: { display: false } },
        scales: { y: { beginAtZero: true, ticks: { precision: 0 } } }
      }
    });
  }

  const scatterCanvas = document.getElementById('correlationScatter');
  if (scatterCanvas) {
    if (window.dsCharts.scatter) {
      try { window.dsCharts.scatter.destroy(); } catch (_) {}
    }

    const rows = Array.isArray(analysis.length_vs_text_effect_scatter)
      ? analysis.length_vs_text_effect_scatter
      : [];

    window.dsCharts.scatter = new Chart(scatterCanvas, {
      type: 'scatter',
      data: {
        datasets: [{
          data: rows,
          backgroundColor: '#0A0A0A',
          pointRadius: 4
        }]
      },
      options: {
        ...co,
        plugins: { legend: { display: false } },
        scales: {
          x: { title: { display: true, text: 'WORD COUNT' } },
          y: {
            title: { display: true, text: 'TEXT-EFFECT RELATIVE SCORE' },
            beginAtZero: true,
            max: 100
          }
        }
      }
    });
  }
}

function buildPayloadFromTerminal(textOverride = null) {
  return {
    text: textOverride !== null ? textOverride : (document.getElementById('term-input')?.value || '').trim(),
    followers: document.getElementById('term-followers')?.value || null,
    following: document.getElementById('term-following')?.value || null,
    num_posts: document.getElementById('term-posts')?.value || null,
    type: document.getElementById('term-type')?.value || null,
    timestamp: document.getElementById('term-timestamp')?.value || null,
    audience: document.getElementById('term-audience')?.value || 'generic',
    image_grade: document.getElementById('term-image-grade')?.value || null,
    description_grade: document.getElementById('term-description-grade')?.value || null
  };
}

function initTerminalActions() {
  const termBtn = document.getElementById('term-btn');
  if (!termBtn) return;

  termBtn.addEventListener('click', async () => {
    const payload = buildPayloadFromTerminal();
    if (!payload.text) {
      alert('ERROR: ENTER TEXT');
      return;
    }

    const originalBtnContent = termBtn.innerHTML;
    termBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i><span>PROCESSING...</span><i class="fa-solid fa-spinner fa-spin"></i>';

    try {
      const data = await postJSON('/api/analyze', payload);

      const mode = data?.prediction_mode || '--';
      setText('term-mode', mode.toUpperCase());

      const modelOutput = data?.model_output || {};
      const relative = data?.relative_scores || {};
      const aux = data?.auxiliary_analysis || {};
      const profile = aux?.profile || {};

      const primaryScore = mode === 'full_prediction'
        ? safeNum(relative.engagement_relative_score, 0)
        : safeNum(relative.text_effect_relative_score, 0);

      animateValue(document.getElementById('term-score'), 0, primaryScore, 800);

      const predLog = modelOutput.predicted_engagement_log;
      setText('term-pred-log', predLog === null || predLog === undefined ? 'N/A' : Number(predLog).toFixed(4));

      animateValue(document.getElementById('term-textscore'), 0, safeNum(relative.text_effect_relative_score, 0), 800);
      setText('term-baseline', modelOutput.baseline_component === null || modelOutput.baseline_component === undefined ? 'N/A' : Number(modelOutput.baseline_component).toFixed(4));
      setText('term-texteffect', Number(safeNum(modelOutput.text_residual_component, 0)).toFixed(4));

      renderStageAStatus(data);
      renderTerms('term-phrases', data?.attention_analysis?.terms || [], 6);
      renderAdvice('term-advice', aux?.recommendations || {});
      renderDashboardMetrics(data);

      const sent = profile?.sentiment || {};
      renderSentiment(
        'term',
        safeNum(sent.pos_pct, 0),
        safeNum(sent.neu_pct, 0),
        safeNum(sent.neg_pct, 0)
      );
    } catch (e) {
      alert(`ERROR: ${e.message}`);
    } finally {
      termBtn.innerHTML = originalBtnContent;
    }
  });
}

function initABActions() {
  const abBtn = document.getElementById('ab-btn');
  if (!abBtn) return;

  abBtn.addEventListener('click', async () => {
    const textA = (document.getElementById('ab-input-a')?.value || '').trim();
    const textB = (document.getElementById('ab-input-b')?.value || '').trim();

    if (!textA || !textB) {
      alert('TWO TEXT VARIANTS ARE REQUIRED');
      return;
    }

    const common = buildPayloadFromTerminal();
    const payload = {
      textA,
      textB,
      followers: common.followers,
      following: common.following,
      num_posts: common.num_posts,
      type: common.type,
      timestamp: common.timestamp,
      audience: common.audience,
      image_grade: common.image_grade,
      description_grade: common.description_grade
    };

    const originalBtnContent = abBtn.innerHTML;
    abBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i><span>COMPARING...</span><i class="fa-solid fa-spinner fa-spin"></i>';

    try {
      const data = await postJSON('/api/ab_compare', payload);

      const metric = data?.comparison?.metric || 'text_effect_relative_score';

      const scoreA = metric === 'engagement_relative_score'
        ? safeNum(data?.A?.relative_scores?.engagement_relative_score, 0)
        : safeNum(data?.A?.relative_scores?.text_effect_relative_score, 0);

      const scoreB = metric === 'engagement_relative_score'
        ? safeNum(data?.B?.relative_scores?.engagement_relative_score, 0)
        : safeNum(data?.B?.relative_scores?.text_effect_relative_score, 0);

      animateValue(document.getElementById('ab-score-a'), 0, scoreA, 700);
      animateValue(document.getElementById('ab-score-b'), 0, scoreB, 700);

      setText('ab-texteffect-a', Number(safeNum(data?.A?.model_output?.text_residual_component, 0)).toFixed(4));
      setText('ab-texteffect-b', Number(safeNum(data?.B?.model_output?.text_residual_component, 0)).toFixed(4));

      renderAdvice('ab-advice-a', data?.A?.auxiliary_analysis?.recommendations || {});
      renderAdvice('ab-advice-b', data?.B?.auxiliary_analysis?.recommendations || {});
      renderSignalsChips(data?.A, 'ab-signals-a');
      renderSignalsChips(data?.B, 'ab-signals-b');
      renderCentralDiagnostics(data);
      renderABCharts(data);

      const diff = safeNum(data?.comparison?.delta_B_minus_A, 0);
      const centerBadge = document.getElementById('ab-center-badge');

      if (diff > 0) {
        abBtn.innerHTML = `<i class="fa-solid fa-check"></i><span>VARIANT B IS HIGHER (+${diff.toFixed(1)})</span><i class="fa-solid fa-b"></i>`;
        if (centerBadge) centerBadge.innerText = 'B';
      } else if (diff < 0) {
        const absDiff = Math.abs(diff);
        abBtn.innerHTML = `<i class="fa-solid fa-check"></i><span>VARIANT A IS HIGHER (+${absDiff.toFixed(1)})</span><i class="fa-solid fa-a"></i>`;
        if (centerBadge) centerBadge.innerText = 'A';
      } else {
        abBtn.innerHTML = '<i class="fa-solid fa-equals"></i><span>NO MEANINGFUL DIFFERENCE</span><i class="fa-solid fa-equals"></i>';
        if (centerBadge) centerBadge.innerText = '=';
      }
    } catch (e) {
      alert(`ERROR: ${e.message}`);
      abBtn.innerHTML = originalBtnContent;
    }
  });
}

function initABDiagnosticsModal() {
  const abModal = document.getElementById('ab-diagnostics-modal');
  const abOpenDiagnostics = document.getElementById('ab-open-diagnostics');
  const abCloseDiagnostics = document.getElementById('ab-close-diagnostics');

  const openAbModal = () => {
    if (abModal) abModal.classList.add('active');
  };

  const closeAbModal = () => {
    if (abModal) abModal.classList.remove('active');
  };

  if (abOpenDiagnostics) {
    abOpenDiagnostics.addEventListener('click', openAbModal);
  }

  if (abCloseDiagnostics) {
    abCloseDiagnostics.addEventListener('click', closeAbModal);
  }

  if (abModal) {
    abModal.addEventListener('click', (e) => {
      if (e.target === abModal) closeAbModal();
    });
  }

  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') closeAbModal();
  });
}

function initDatasetActions() {
  const fileInput = document.getElementById('csvFileInput');
  const visBtn = document.getElementById('visualizeBtn');

  if (fileInput) {
    fileInput.addEventListener('change', (e) => {
      if (e.target.files?.length) {
        const nameEl = document.getElementById('ds-upload-text');
        if (nameEl) nameEl.innerText = e.target.files[0].name.toUpperCase();
        if (visBtn) visBtn.classList.remove('btn-disabled');

        clearDatasetUI();
        setText('datasetStatus', 'FILE READY');
      }
    });
  }

  if (visBtn) {
    visBtn.addEventListener('click', async () => {
      if (!fileInput?.files?.[0]) return;

      clearDatasetUI();

      const fd = new FormData();
      fd.append('file', fileInput.files[0]);

      const originalBtnContent = visBtn.innerHTML;
      visBtn.innerHTML = '<i class="fa-solid fa-spinner fa-spin"></i><span>PROCESSING DATASET...</span><i class="fa-solid fa-spinner fa-spin"></i>';

      try {
        const r = await fetch('/api/upload_csv', {
          method: 'POST',
          body: fd
        });

        const data = await r.json();
        if (!data.ok) throw new Error(data.error || 'DATASET_REQUEST_FAILED');

        const summary = data?.summary || {};
        const metricSummary = data?.metric_summary || {};

        setText('datasetStatus', 'PROCESSED');
        animateValue(document.getElementById('ds-rows'), 0, safeNum(summary.total_rows, 0), 900);

        const meanValue = metricSummary.mean_value;
        if (typeof meanValue === 'number' && Number.isFinite(meanValue)) {
          animateValue(document.getElementById('ds-avg'), 0, meanValue, 900);
        } else {
          setText('ds-avg', 'N/A');
        }

        renderDatasetMeta(data);
        renderDatasetCharts(data);
      } catch (e) {
        alert(`DATASET ERROR: ${e.message}`);
        setText('datasetStatus', 'PROCESSING ERROR');
      } finally {
        visBtn.innerHTML = originalBtnContent;
      }
    });
  }
}
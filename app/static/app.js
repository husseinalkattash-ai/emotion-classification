/**
 * Emotion Classifier — Client-Side Application Logic
 *
 * Handles drag-and-drop upload, fetch to /api/predict,
 * animated result rendering, and session history.
 */

// ── Emotion Emoji Map ──────────────────────────────────────
const EMOTION_EMOJIS = {
  surprise: '😲',
  fear:     '😨',
  disgust:  '🤢',
  happy:    '😄',
  sad:      '😢',
  angry:    '😠',
  neutral:  '😐',
};

// ── DOM References ─────────────────────────────────────────
const uploadZone     = document.getElementById('upload-zone');
const fileInput      = document.getElementById('file-input');
const previewWrapper = document.getElementById('preview-wrapper');
const previewImg     = document.getElementById('preview-img');
const previewName    = document.getElementById('preview-filename');
const analyzeBtn     = document.getElementById('analyze-btn');
const clearBtn       = document.getElementById('clear-btn');
const spinnerOverlay = document.getElementById('spinner-overlay');
const resultContent  = document.getElementById('result-content');
const resultPlaceholder = document.getElementById('results-placeholder');
const predEmoji      = document.getElementById('pred-emoji');
const predLabel      = document.getElementById('pred-label');
const predConfidence = document.getElementById('pred-confidence');
const faceBadge      = document.getElementById('face-badge');
const uncertaintyNote = document.getElementById('uncertainty-note');
const explainRow     = document.getElementById('explain-row');
const cropImg        = document.getElementById('crop-img');
const gradcamImg     = document.getElementById('gradcam-img');
const probBars       = document.getElementById('prob-bars');
const historyList    = document.getElementById('history-list');
const historyEmpty   = document.getElementById('history-empty');
const errorToast     = document.getElementById('error-toast');

let selectedFile = null;
const history = [];

// ── Upload Zone Events ─────────────────────────────────────
uploadZone.addEventListener('click', () => fileInput.click());

uploadZone.addEventListener('keydown', (e) => {
  if (e.key === 'Enter' || e.key === ' ') {
    e.preventDefault();
    fileInput.click();
  }
});

uploadZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  uploadZone.classList.add('drag-over');
});

uploadZone.addEventListener('dragleave', () => {
  uploadZone.classList.remove('drag-over');
});

uploadZone.addEventListener('drop', (e) => {
  e.preventDefault();
  uploadZone.classList.remove('drag-over');
  const files = e.dataTransfer.files;
  if (files.length > 0) handleFile(files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) handleFile(fileInput.files[0]);
});

// ── File Handling ──────────────────────────────────────────
function handleFile(file) {
  // Validate type
  if (!file.type.startsWith('image/')) {
    showError('Please upload an image file (JPG, PNG, WebP, etc.).');
    return;
  }

  // Validate size (10 MB)
  if (file.size > 10 * 1024 * 1024) {
    showError('File exceeds the 10 MB size limit.');
    return;
  }

  selectedFile = file;

  // Show preview
  const reader = new FileReader();
  reader.onload = (e) => {
    previewImg.src = e.target.result;
    previewName.textContent = file.name;
    previewWrapper.classList.add('visible');
    uploadZone.style.display = 'none';
    analyzeBtn.disabled = false;
  };
  reader.readAsDataURL(file);

  // Hide previous results
  resultContent.classList.remove('visible');
  resultPlaceholder.style.display = 'block';
}

// ── Clear / Reset ──────────────────────────────────────────
clearBtn.addEventListener('click', resetUpload);

function resetUpload() {
  selectedFile = null;
  fileInput.value = '';
  previewWrapper.classList.remove('visible');
  uploadZone.style.display = 'block';
  analyzeBtn.disabled = true;
  resultContent.classList.remove('visible');
  resultPlaceholder.style.display = 'block';
  explainRow.classList.remove('visible');
  uncertaintyNote.classList.remove('visible');
}

// ── Analyze Button ─────────────────────────────────────────
analyzeBtn.addEventListener('click', async () => {
  if (!selectedFile) return;

  // Show spinner
  spinnerOverlay.classList.add('visible');
  analyzeBtn.disabled = true;

  try {
    const formData = new FormData();
    formData.append('file', selectedFile);

    const response = await fetch('/api/predict', {
      method: 'POST',
      body: formData,
    });

    if (!response.ok) {
      const err = await response.json().catch(() => ({}));
      throw new Error(err.detail || `Server error (${response.status})`);
    }

    const data = await response.json();
    renderResult(data);
    addToHistory(selectedFile.name, data);

  } catch (err) {
    showError(err.message || 'An unexpected error occurred.');
  } finally {
    spinnerOverlay.classList.remove('visible');
    analyzeBtn.disabled = false;
  }
});

// ── Render Result ──────────────────────────────────────────
function renderResult(data) {
  const { predicted, confidence, probabilities, face_detected } = data;

  // Prediction header
  predEmoji.textContent = EMOTION_EMOJIS[predicted] || '🤔';
  predLabel.textContent = predicted;
  predConfidence.innerHTML = `Confidence: <span>${(confidence * 100).toFixed(1)}%</span>`;

  // Face detection badge
  if (face_detected) {
    faceBadge.className = 'face-badge face-badge--detected';
    faceBadge.textContent = '✓ Face detected';
  } else {
    faceBadge.className = 'face-badge face-badge--fallback';
    faceBadge.textContent = '⚠ No face — used full image';
  }

  // Uncertainty note for ambiguous predictions (e.g. fear vs surprise)
  const cert = data.certainty;
  if (cert && !cert.confident) {
    const runnerPct = (cert.runner_up_prob * 100).toFixed(1);
    uncertaintyNote.innerHTML =
      `⚠ Low confidence — this could also be ` +
      `<strong>${cert.runner_up}</strong> (${runnerPct}%). Treat the result as uncertain.`;
    uncertaintyNote.classList.add('visible');
  } else {
    uncertaintyNote.classList.remove('visible');
  }

  // Explainability: analyzed crop + Grad-CAM heatmap
  if (data.crop_image && data.gradcam_image) {
    cropImg.src = data.crop_image;
    gradcamImg.src = data.gradcam_image;
    explainRow.classList.add('visible');
  } else {
    explainRow.classList.remove('visible');
  }

  // Probability bars
  probBars.innerHTML = '';
  const sortedEmotions = Object.entries(probabilities)
    .sort((a, b) => b[1] - a[1]);

  sortedEmotions.forEach(([emotion, prob], index) => {
    const isPredicted = emotion === predicted;
    const bar = document.createElement('div');
    bar.className = `prob-bar${isPredicted ? ' is-predicted' : ''}`;
    bar.dataset.emotion = emotion;
    bar.innerHTML = `
      <span class="prob-bar__label">${EMOTION_EMOJIS[emotion] || ''} ${emotion}</span>
      <div class="prob-bar__track">
        <div class="prob-bar__fill" style="width: 0%"></div>
      </div>
      <span class="prob-bar__value">${(prob * 100).toFixed(1)}%</span>
    `;
    probBars.appendChild(bar);

    // Animate the bar fill after a small stagger
    requestAnimationFrame(() => {
      setTimeout(() => {
        bar.querySelector('.prob-bar__fill').style.width = `${Math.max(prob * 100, 1)}%`;
      }, index * 80);
    });
  });

  // Show results, hide placeholder
  resultPlaceholder.style.display = 'none';
  resultContent.classList.add('visible');
}

// ── Session History ────────────────────────────────────────
function addToHistory(filename, data) {
  history.unshift({ filename, ...data, timestamp: Date.now() });
  renderHistory();
}

function renderHistory() {
  if (history.length === 0) {
    historyEmpty.style.display = 'block';
    historyList.innerHTML = '';
    return;
  }

  historyEmpty.style.display = 'none';
  historyList.innerHTML = history.map((item) => `
    <li class="history-item">
      <span class="history-item__emoji">${EMOTION_EMOJIS[item.predicted] || '🤔'}</span>
      <div class="history-item__info">
        <div class="history-item__name">${escapeHtml(item.filename)}</div>
        <div class="history-item__result">
          ${item.predicted} — <span>${(item.confidence * 100).toFixed(1)}%</span>
        </div>
      </div>
    </li>
  `).join('');
}

// ── Error Toast ────────────────────────────────────────────
let errorTimeout = null;

function showError(message) {
  errorToast.textContent = message;
  errorToast.classList.add('visible');

  if (errorTimeout) clearTimeout(errorTimeout);
  errorTimeout = setTimeout(() => {
    errorToast.classList.remove('visible');
  }, 4000);
}

// ── Utility ────────────────────────────────────────────────
function escapeHtml(str) {
  const div = document.createElement('div');
  div.textContent = str;
  return div.innerHTML;
}

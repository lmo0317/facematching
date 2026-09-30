// Global application state
let currentMainTab = 'compare'; // 'compare' | 'celeb'
let photo1Data = null; // dataURL (may be cropped/compressed)
let photo2Data = null; // dataURL (may be cropped/compressed)
let photo1OriginalData = null; // uncropped original dataURL
let photo2OriginalData = null; // uncropped original dataURL
let photoCelebData = null; // dataURL for celebrity search
let photoCelebOriginalData = null;
let celebExtraPhotos = []; // extra photos of the same person (max 2), face-cropped client-side, averaged server-side
let detectedFacesCeleb = [];
let selectedCelebGender = 'auto'; // 'auto' | 'male' | 'female'

let activeCropperTarget = null; // 1, 2, or 'celeb'
let cropperInstance = null; // Cropper.js instance
let activeWebcamTarget = null; // 1, 2, or 'celeb'
let webcamStream = null;
let currentSamplePresets = [];
let currentAnalysisResult = null;
let currentMode = 'family'; // 'family' | 'celebrity' | 'identical'
let detectedFaces1 = []; // detected faces in photo 1
let detectedFaces2 = []; // detected faces in photo 2
let activeFaceIndex = null;

// Tab switcher between 2-Photo Compare and Celebrity Search
function switchMainTab(tab) {
  currentMainTab = tab;
  const btnCompare = document.getElementById('tab-btn-compare');
  const btnCeleb = document.getElementById('tab-btn-celeb');
  const secCompare = document.getElementById('section-compare-tab');
  const secCeleb = document.getElementById('section-celeb-tab');

  const isCompare = tab === 'compare';
  secCompare.classList.toggle('hidden', !isCompare);
  secCeleb.classList.toggle('hidden', isCompare);
  btnCompare.classList.toggle('is-active', isCompare);
  btnCeleb.classList.toggle('is-active', !isCompare);
  btnCompare.setAttribute('aria-selected', String(isCompare));
  btnCeleb.setAttribute('aria-selected', String(!isCompare));
  lucide.createIcons();
}

function setCelebGenderFilter(gender) {
  selectedCelebGender = gender;
  ['auto', 'male', 'female'].forEach(g => {
    const btn = document.getElementById(`celeb-filter-${g}`);
    if (btn) btn.classList.toggle('is-active', g === gender);
  });
}

function setAnalysisMode(mode) {
  currentMode = mode;
  const btnText = document.querySelector('#btn-compare span');
  if (btnText) {
    btnText.textContent = '닮음 분석하기';
  }
}

// Dynamically detect base URL prefix (works at / or /facematching)
function getBaseUrl() {
  const path = window.location.pathname;
  if (path.startsWith('/facematching')) {
    return '/facematching';
  }
  return '';
}
const BASE_URL = getBaseUrl();

function initApp() {
  checkServerHealth();
  loadSamplePresetsList();
  setupClipboardPaste();
  setupKeyboardShortcuts();

  document.getElementById('btn-health-check')?.addEventListener('click', () => {
    checkServerHealth();
  });
}

// index.html injects this script dynamically, so DOMContentLoaded has usually fired already
if (document.readyState === 'loading') {
  document.addEventListener('DOMContentLoaded', initApp);
} else {
  initApp();
}

function setupKeyboardShortcuts() {
  window.addEventListener('keydown', (e) => {
    if (e.key === 'Escape') {
      const cropModal = document.getElementById('crop-modal');
      if (cropModal && !cropModal.classList.contains('hidden')) {
        closeCropper();
      }
      const webcamModal = document.getElementById('webcam-modal');
      if (webcamModal && !webcamModal.classList.contains('hidden')) {
        closeWebcam();
      }
    }
  });
}

// Check Gemma 4 llama-server connectivity
async function checkServerHealth() {
  const statusPill = document.getElementById('server-status-pill');
  const statusDot = document.getElementById('status-dot');
  const statusText = document.getElementById('status-text');

  try {
    const res = await fetch(BASE_URL + '/api/health');
    const data = await res.json();
    
    if (data.backend?.status === 'connected' && data.backend?.multimodal_enabled) {
      setServerStatus('ok', '분석 서버 연결됨', `분석 서버 연결됨 (${data.backend.target_model})`);
    } else if (data.backend?.status === 'connected') {
      setServerStatus('warn', 'AI 설명 확인 중', 'Gemma 서버는 연결됐지만 이미지 기능 확인 중');
    } else {
      setServerStatus('error', 'AI 설명 꺼짐', 'Gemma 서버에 연결할 수 없어 얼굴 인식 결과만 표시합니다');
    }
  } catch (err) {
    setServerStatus('error', '서버 응답 없음', '백엔드 서버 응답 없음');
  }
}

function setServerStatus(level, label, title) {
  const dotColor = { ok: 'bg-emerald-400', warn: 'bg-amber-400 animate-pulse', error: 'bg-rose-500' }[level];
  document.getElementById('status-dot').className = `w-2 h-2 rounded-full ${dotColor}`;
  document.getElementById('status-text').textContent = label;
  document.getElementById('server-status-pill').title = title;
}

// Fetch available sample presets
async function loadSamplePresetsList() {
  try {
    const res = await fetch(BASE_URL + '/api/samples');
    const data = await res.json();
    currentSamplePresets = data.samples || [];
  } catch (err) {
    console.warn('샘플 목록 로드 실패:', err);
  }
}

// Load a specific sample preset
async function loadSamplePreset(index) {
  const sample = currentSamplePresets[index];
  if (!sample) return;

  try {
    const url1 = sample.img1.startsWith('/') ? BASE_URL + sample.img1 : sample.img1;
    const url2 = sample.img2.startsWith('/') ? BASE_URL + sample.img2 : sample.img2;

    // Set photo 1
    const res1 = await fetch(url1);
    const blob1 = await res1.blob();
    setPhotoFromBlob(blob1, 1, `${sample.id}_1.jpg`);

    // Set photo 2
    const res2 = await fetch(url2);
    const blob2 = await res2.blob();
    setPhotoFromBlob(blob2, 2, `${sample.id}_2.jpg`);

    if (sample.mode) {
      setAnalysisMode(sample.mode);
    }

    // Scroll to action button smoothly
    setTimeout(() => {
      document.getElementById('btn-compare')?.scrollIntoView({ behavior: 'smooth', block: 'center' });
    }, 300);

  } catch (err) {
    alert(`샘플 이미지를 불러오는 중 오류가 발생했습니다: ${err.message}`);
  }
}

// Drag & Drop handlers
function handleDragOver(e, targetId) {
  e.preventDefault();
  e.stopPropagation();
  document.getElementById(`dropzone-${targetId}`).classList.add('dragover');
}

function handleDragLeave(e, targetId) {
  e.preventDefault();
  e.stopPropagation();
  document.getElementById(`dropzone-${targetId}`).classList.remove('dragover');
}

function handleDrop(e, targetId) {
  e.preventDefault();
  e.stopPropagation();
  document.getElementById(`dropzone-${targetId}`).classList.remove('dragover');

  const files = e.dataTransfer.files;
  if (files && files.length > 0) {
    handleFile(files[0], targetId);
  }
}

function triggerFileInput(targetId, e) {
  if (e && e.target && e.target.closest('button')) return;
  const input = document.getElementById(`file-input-${targetId}`);
  if (input) input.click();
}

function handleFileSelect(e, targetId) {
  const files = e.target.files;
  if (files && files.length > 0) {
    handleFile(files[0], targetId);
  }
}

function handleFile(file, targetId) {
  if (!file.type.startsWith('image/')) {
    alert('이미지 파일(JPG, PNG, WEBP 등)만 등록할 수 있습니다.');
    return;
  }
  setPhotoFromBlob(file, targetId, file.name);
}

// Client-side image compression to keep payload well under Nginx's 1MB limit
async function compressImage(blob, maxDim = 800, initialQuality = 0.85) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    const objectUrl = URL.createObjectURL(blob);
    img.onload = () => {
      URL.revokeObjectURL(objectUrl);
      let { width, height } = img;
      
      // Calculate scaled dimensions
      if (width > maxDim || height > maxDim) {
        if (width > height) {
          height = Math.round((height * maxDim) / width);
          width = maxDim;
        } else {
          width = Math.round((width * maxDim) / height);
          height = maxDim;
        }
      }

      const canvas = document.createElement('canvas');
      canvas.width = width;
      canvas.height = height;
      const ctx = canvas.getContext('2d');
      ctx.drawImage(img, 0, 0, width, height);

      let quality = initialQuality;
      let dataUrl = canvas.toDataURL('image/jpeg', quality);
      let base64Length = dataUrl.length - (dataUrl.indexOf(',') + 1);
      let sizeInBytes = Math.round((base64Length * 3) / 4);

      // If compressed size is still > 350KB, step down quality to guarantee < 350KB per photo
      if (sizeInBytes > 350 * 1024) {
        quality = 0.72;
        dataUrl = canvas.toDataURL('image/jpeg', quality);
        base64Length = dataUrl.length - (dataUrl.indexOf(',') + 1);
        sizeInBytes = Math.round((base64Length * 3) / 4);
      }

      resolve({
        dataUrl,
        sizeInKb: Math.round(sizeInBytes / 1024),
        width,
        height
      });
    };
    img.onerror = (err) => {
      URL.revokeObjectURL(objectUrl);
      reject(err);
    };
    img.src = objectUrl;
  });
}

async function setPhotoFromBlob(blob, targetId, filename = 'image.jpg') {
  try {
    const origSizeKb = Math.round(blob.size / 1024);
    // Compress and scale image in browser canvas
    const { dataUrl, sizeInKb, width, height } = await compressImage(blob, 800, 0.85);

    if (targetId === 1) {
      photo1Data = dataUrl;
      photo1OriginalData = dataUrl;
    } else if (targetId === 2) {
      photo2Data = dataUrl;
      photo2OriginalData = dataUrl;
    } else if (targetId === 'celeb') {
      photoCelebData = dataUrl;
      photoCelebOriginalData = dataUrl;
    }

    // Show preview
    document.getElementById(`empty-state-${targetId}`).classList.add('hidden');
    const previewContainer = document.getElementById(`preview-container-${targetId}`);
    const previewImg = document.getElementById(`preview-img-${targetId}`);
    previewContainer.classList.remove('hidden');
    previewImg.src = dataUrl;

    // Show clear and crop buttons, hide restore and cropped badge
    document.getElementById(`btn-clear-${targetId}`).classList.remove('hidden');
    document.getElementById(`btn-crop-${targetId}`)?.classList.remove('hidden');
    document.getElementById(`btn-restore-${targetId}`)?.classList.add('hidden');
    document.getElementById(`badge-crop-${targetId}`)?.classList.add('hidden');
    
    // Display file name with optimization info
    const infoText = origSizeKb > 500 
      ? `${filename} (${width}×${height}, ${sizeInKb} KB / 원본: ${origSizeKb} KB 압축됨)`
      : `${filename} (${sizeInKb} KB)`;
    document.getElementById(`file-info-${targetId}`).textContent = infoText;

    // Trigger face detection in background
    detectFaces(targetId, dataUrl);
    if (targetId === 'celeb') refreshCelebExtraMatches();

    updateCompareButtonState();
    updateCelebButtonState();
    lucide.createIcons();
  } catch (err) {
    console.error('이미지 압축 실패:', err);
    // Fallback: read directly if canvas fails
    const reader = new FileReader();
    reader.onload = (e) => {
      const dataUrl = e.target.result;
      if (targetId === 1) {
        photo1Data = dataUrl;
        photo1OriginalData = dataUrl;
      } else if (targetId === 2) {
        photo2Data = dataUrl;
        photo2OriginalData = dataUrl;
      } else if (targetId === 'celeb') {
        photoCelebData = dataUrl;
        photoCelebOriginalData = dataUrl;
      }
      document.getElementById(`empty-state-${targetId}`).classList.add('hidden');
      document.getElementById(`preview-container-${targetId}`).classList.remove('hidden');
      document.getElementById(`preview-img-${targetId}`).src = dataUrl;
      document.getElementById(`btn-clear-${targetId}`).classList.remove('hidden');
      document.getElementById(`btn-crop-${targetId}`)?.classList.remove('hidden');
      document.getElementById(`btn-restore-${targetId}`)?.classList.add('hidden');
      document.getElementById(`badge-crop-${targetId}`)?.classList.add('hidden');
      document.getElementById(`file-info-${targetId}`).textContent = `${filename} (${Math.round(blob.size / 1024)} KB)`;
      
      // Trigger face detection in background
      detectFaces(targetId, dataUrl);
      
      updateCompareButtonState();
      updateCelebButtonState();
      lucide.createIcons();
    };
    reader.readAsDataURL(blob);
  }
}

function clearCelebPhoto() {
  photoCelebData = null;
  photoCelebOriginalData = null;
  detectedFacesCeleb = [];
  document.getElementById('empty-state-celeb')?.classList.remove('hidden');
  document.getElementById('preview-container-celeb')?.classList.add('hidden');
  const img = document.getElementById('preview-img-celeb');
  if (img) img.src = '';
  document.getElementById('btn-clear-celeb')?.classList.add('hidden');
  document.getElementById('btn-crop-celeb')?.classList.add('hidden');
  document.getElementById('btn-restore-celeb')?.classList.add('hidden');
  document.getElementById('badge-crop-celeb')?.classList.add('hidden');
  document.getElementById('face-detect-pill-celeb')?.classList.add('hidden');
  const fileInfo = document.getElementById('file-info-celeb');
  if (fileInfo) fileInfo.textContent = '선택된 파일 없음';
  const inp = document.getElementById('file-input-celeb');
  if (inp) inp.value = '';
  celebExtraPhotos = [];
  renderCelebExtraPhotos();
  updateCelebButtonState();
}

const MAX_CELEB_EXTRA = 2;
// Below this ArcFace cosine to the main photo the server treats a photo as someone else
const EXTRA_SAME_PERSON_MIN = 0.25;

// Each extra photo: { original, faces, selected, cropped, status: 'detecting' | 'ready' | 'noface' | 'error' }
async function handleCelebExtraFiles(e) {
  const files = Array.from(e.target.files || []).filter(f => f.type.startsWith('image/'));
  e.target.value = '';
  for (const file of files) {
    if (celebExtraPhotos.length >= MAX_CELEB_EXTRA) break;
    try {
      const { dataUrl } = await compressImage(file, 800, 0.85);
      const item = { original: dataUrl, faces: [], selected: null, cropped: null, status: 'detecting' };
      celebExtraPhotos.push(item);
      renderCelebExtraPhotos();
      detectCelebExtraFaces(item);
    } catch (err) {
      console.warn('추가 사진 처리 실패:', err);
    }
  }
}

// Detect faces in an extra photo and auto-pick the one that looks like the main photo's person
async function detectCelebExtraFaces(item) {
  try {
    const res = await fetch(BASE_URL + '/api/detect-faces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: item.original, reference_image_base64: photoCelebData || null })
    });
    if (!res.ok) throw new Error(`HTTP ${res.status}`);
    const data = await res.json();
    item.faces = data.faces || [];
    if (!item.faces.length) {
      item.status = 'noface';
    } else {
      // Best match to the main photo when it exists, otherwise the largest face
      const largest = item.faces.reduce((b, f, i) =>
        f.box.width * f.box.height > item.faces[b].box.width * item.faces[b].box.height ? i : b, 0);
      await selectCelebExtraFace(item, data.best_match ?? largest, false);
      item.status = 'ready';
    }
  } catch (err) {
    console.warn('추가 사진 얼굴 감지 실패:', err);
    item.status = 'error';
  }
  renderCelebExtraPhotos();
}

// Crop the extra photo to the chosen face (same 1:1 head framing as the main photo cropper)
async function selectCelebExtraFace(item, index, rerender = true) {
  const face = item.faces[index];
  if (!face) return;
  item.selected = index;
  item.cropped = await cropDataUrl(item.original, face.padded_box);
  if (rerender) renderCelebExtraPhotos();
}

function cropDataUrl(dataUrl, box) {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => {
      const size = Math.min(512, Math.round(box.width));
      const canvas = document.createElement('canvas');
      canvas.width = size;
      canvas.height = size;
      canvas.getContext('2d').drawImage(img, box.x, box.y, box.width, box.height, 0, 0, size, size);
      resolve(canvas.toDataURL('image/jpeg', 0.9));
    };
    img.onerror = reject;
    img.src = dataUrl;
  });
}

// Tap a thumbnail to switch to the next detected face in that photo
function cycleCelebExtraFace(i) {
  const item = celebExtraPhotos[i];
  if (!item || item.faces.length < 2) return;
  selectCelebExtraFace(item, ((item.selected ?? -1) + 1) % item.faces.length);
}

function removeCelebExtraPhoto(index) {
  celebExtraPhotos.splice(index, 1);
  renderCelebExtraPhotos();
}

// When the main photo changes, re-pick faces in extra photos against the new person
function refreshCelebExtraMatches() {
  celebExtraPhotos.forEach(item => {
    if (item.status === 'ready') {
      item.status = 'detecting';
      detectCelebExtraFaces(item);
    }
  });
  renderCelebExtraPhotos();
}

function celebExtraPayload() {
  return celebExtraPhotos.filter(p => p.status === 'ready' && p.cropped).map(p => p.cropped);
}

function renderCelebExtraPhotos() {
  const list = document.getElementById('celeb-extra-list');
  const addBtn = document.getElementById('btn-add-celeb-extra');
  if (!list) return;
  list.innerHTML = '';
  celebExtraPhotos.forEach((item, i) => {
    const face = item.selected != null ? item.faces[item.selected] : null;
    const sim = face?.similarity;
    let badge = '';
    let border = 'border-white/10';
    if (item.status === 'detecting') {
      badge = '<span class="px-1.5 py-0.5 rounded bg-slate-700/90 text-slate-200">얼굴 찾는 중</span>';
    } else if (item.status === 'noface') {
      badge = '<span class="px-1.5 py-0.5 rounded bg-rose-600/90 text-white">얼굴 없음</span>';
      border = 'border-rose-500/60';
    } else if (item.status === 'error') {
      badge = '<span class="px-1.5 py-0.5 rounded bg-rose-600/90 text-white">감지 실패</span>';
      border = 'border-rose-500/60';
    } else if (sim != null && sim < EXTRA_SAME_PERSON_MIN) {
      badge = '<span class="px-1.5 py-0.5 rounded bg-amber-500/90 text-black">다른 사람?</span>';
      border = 'border-amber-400/70';
    } else {
      const multi = item.faces.length > 1 ? ` ${item.selected + 1}/${item.faces.length}` : '';
      badge = `<span class="px-1.5 py-0.5 rounded bg-emerald-600/90 text-white">얼굴 맞춤${multi}</span>`;
      border = 'border-emerald-500/60';
    }
    const cell = document.createElement('div');
    cell.className = 'flex flex-col items-center';
    cell.innerHTML = `
      <div class="relative w-16 h-16 rounded-lg overflow-hidden border-2 ${border} bg-slate-800 ${item.faces.length > 1 ? 'cursor-pointer' : ''}"
           ${item.faces.length > 1 ? `onclick="cycleCelebExtraFace(${i})" title="눌러서 다른 얼굴 선택"` : ''}>
        <img src="${item.cropped || item.original}" alt="추가 사진 ${i + 1}" class="w-full h-full object-cover">
        <button type="button" onclick="event.stopPropagation(); removeCelebExtraPhoto(${i})" title="삭제"
          class="absolute top-0.5 right-0.5 w-5 h-5 rounded-full bg-black/70 text-white text-xs leading-5 text-center hover:bg-rose-600">×</button>
      </div>
      <div class="mt-1 text-[10px] font-semibold">${badge}</div>`;
    list.appendChild(cell);
  });
  if (addBtn) addBtn.disabled = celebExtraPhotos.length >= MAX_CELEB_EXTRA;
  const multiHint = celebExtraPhotos.some(p => p.faces.length > 1);
  const hint = document.getElementById('celeb-extra-hint');
  if (hint) hint.classList.toggle('hidden', !multiHint);
}

function clearPhoto(targetId) {
  if (targetId === 1) {
    photo1Data = null;
    photo1OriginalData = null;
    detectedFaces1 = [];
  }
  if (targetId === 2) {
    photo2Data = null;
    photo2OriginalData = null;
    detectedFaces2 = [];
  }
  if (targetId === 'celeb') {
    clearCelebPhoto();
    return;
  }

  document.getElementById(`empty-state-${targetId}`).classList.remove('hidden');
  document.getElementById(`preview-container-${targetId}`).classList.add('hidden');
  document.getElementById(`preview-img-${targetId}`).src = '';
  document.getElementById(`btn-clear-${targetId}`).classList.add('hidden');
  document.getElementById(`btn-crop-${targetId}`)?.classList.add('hidden');
  document.getElementById(`btn-restore-${targetId}`)?.classList.add('hidden');
  document.getElementById(`badge-crop-${targetId}`)?.classList.add('hidden');
  document.getElementById(`face-detect-pill-${targetId}`)?.classList.add('hidden');
  document.getElementById(`file-info-${targetId}`).textContent = '선택된 파일 없음';
  document.getElementById(`file-input-${targetId}`).value = '';

  updateCompareButtonState();
}

// --- Face Detection API Client ---
async function detectFaces(targetId, dataUrl) {
  try {
    const pill = document.getElementById(`face-detect-pill-${targetId}`);
    const countEl = document.getElementById(`face-detect-count-${targetId}`);

    const res = await fetch(BASE_URL + '/api/detect-faces', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ image_base64: dataUrl })
    });
    if (!res.ok) return;
    const data = await res.json();
    const faces = data.faces || [];
    if (targetId === 1) detectedFaces1 = faces;
    else if (targetId === 2) detectedFaces2 = faces;
    else if (targetId === 'celeb') detectedFacesCeleb = faces;

    if (faces.length > 1 && pill && countEl) {
      pill.classList.remove('hidden');
      countEl.textContent = `${faces.length}명이 있어요 · 얼굴 고르기`;
      lucide.createIcons();
    } else if (pill) {
      pill.classList.add('hidden');
    }

    if (activeCropperTarget === targetId) {
      renderDetectedFacesBar(targetId);
    }
  } catch (err) {
    console.warn('Face detection error:', err);
  }
}

// --- Cropper.js Modal & Controls ---
function openCropper(targetId) {
  const originalData = targetId === 1 ? photo1OriginalData : (targetId === 2 ? photo2OriginalData : photoCelebOriginalData);
  if (!originalData) return;

  activeCropperTarget = targetId;
  const modal = document.getElementById('crop-modal');
  const cropImg = document.getElementById('cropper-image');
  const modalTitle = document.getElementById('crop-modal-title');
  
  if (modalTitle) {
    modalTitle.textContent = targetId === 'celeb'
      ? '내 사진 - 얼굴 맞춤 선택'
      : `사진 ${targetId} - 인물 얼굴 / 특정 영역 선택`;
  }

  // Helper to initialize cropper
  const initCropper = () => {
    if (cropperInstance) {
      cropperInstance.destroy();
      cropperInstance = null;
    }
    cropperInstance = new Cropper(cropImg, {
      viewMode: 1, // Restrict crop box to within image boundary
      dragMode: 'move',
      autoCropArea: 0.8,
      restore: false,
      guides: true,
      center: true,
      highlight: true,
      cropBoxMovable: true,
      cropBoxResizable: true,
      toggleDragModeOnDblclick: false,
      ready() {
        const faces = targetId === 1 ? detectedFaces1 : (targetId === 2 ? detectedFaces2 : detectedFacesCeleb);
        if (faces && faces.length > 0) {
          renderDetectedFacesBar(targetId);
        } else {
          setCropRatio(NaN);
          renderDetectedFacesBar(targetId);
        }
      }
    });
  };

  modal.classList.remove('hidden');
  modal.classList.add('flex');

  if (cropImg.src === originalData && cropImg.complete) {
    initCropper();
  } else {
    cropImg.onload = () => {
      cropImg.onload = null;
      initCropper();
    };
    cropImg.src = originalData;
  }

  lucide.createIcons();
}

function renderDetectedFacesBar(targetId) {
  const faces = targetId === 1 ? detectedFaces1 : (targetId === 2 ? detectedFaces2 : detectedFacesCeleb);
  const bar = document.getElementById('crop-faces-bar');
  const title = document.getElementById('crop-faces-title');
  const chipsContainer = document.getElementById('crop-faces-chips');

  if (!bar || !chipsContainer) return;

  if (!faces || faces.length === 0) {
    bar.classList.add('hidden');
    return;
  }

  bar.classList.remove('hidden');
  if (title) {
    title.textContent = `감지된 인물 (${faces.length}명)`;
  }
  chipsContainer.innerHTML = '';
  activeFaceIndex = null;

  faces.forEach((face, idx) => {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.id = `face-chip-${idx}`;
    btn.title = `${face.label} - 클릭 시 영역 맞춤, 더블클릭 시 즉시 확정`;
    btn.className = 'face-chip flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-800/90 hover:bg-indigo-600/30 border border-slate-700 hover:border-indigo-500/50 transition cursor-pointer text-xs group';
    btn.innerHTML = `
      <img src="${face.thumbnail}" class="w-6 h-6 rounded-full object-cover border border-white/20">
      <span class="font-medium text-slate-200 group-hover:text-white">${face.label}</span>
      <span class="text-[10px] px-1.5 py-0.5 rounded bg-emerald-500/20 text-emerald-300 border border-emerald-500/30">선택</span>
    `;
    btn.onclick = () => selectDetectedFace(idx, targetId);
    btn.ondblclick = () => {
      selectDetectedFace(idx, targetId);
      applyCroppedImage();
    };
    chipsContainer.appendChild(btn);
  });

  // If faces exist, auto-fit to the first face
  if (faces.length > 0) {
    selectDetectedFace(0, targetId);
  }
}

function selectDetectedFace(index, targetId) {
  const faces = targetId === 1 ? detectedFaces1 : (targetId === 2 ? detectedFaces2 : detectedFacesCeleb);
  if (!faces || !faces[index] || !cropperInstance) return;

  activeFaceIndex = index;
  const face = faces[index];

  // Update confirm button text in faces bar
  const confirmBtnText = document.getElementById('crop-face-confirm-text');
  if (confirmBtnText) {
    confirmBtnText.textContent = `${face.label} 확정 (확인)`;
  }

  // Highlight active chip
  faces.forEach((_, i) => {
    const chip = document.getElementById(`face-chip-${i}`);
    if (chip) {
      if (i === index) {
        chip.className = 'face-chip flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-emerald-950/60 border-2 border-emerald-500 shadow-md shadow-emerald-600/30 text-white text-xs cursor-pointer';
      } else {
        chip.className = 'face-chip flex items-center space-x-2 px-3 py-1.5 rounded-xl bg-slate-800/90 hover:bg-indigo-600/30 border border-slate-700 hover:border-indigo-500/50 transition cursor-pointer text-xs group';
      }
    }
  });

  // Set 1:1 aspect ratio and fit Cropper.js box to detected face (balanced 1:1 square)
  setCropRatio(1);
  const box = face.padded_box;
  cropperInstance.setData({
    x: Math.round(box.x),
    y: Math.round(box.y),
    width: Math.round(box.width),
    height: Math.round(box.height)
  });
}

function closeCropper() {
  const modal = document.getElementById('crop-modal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');
  if (cropperInstance) {
    cropperInstance.destroy();
    cropperInstance = null;
  }
  activeCropperTarget = null;
}

function setCropRatio(ratio) {
  if (!cropperInstance) return;
  cropperInstance.setAspectRatio(ratio);

  const ratioButtons = [
    { id: 'ratio-free', match: isNaN(ratio) },
    { id: 'ratio-1-1', match: ratio === 1 },
    { id: 'ratio-4-3', match: typeof ratio === 'number' && Math.abs(ratio - 4/3) < 0.05 }
  ];

  ratioButtons.forEach(btn => {
    const el = document.getElementById(btn.id);
    if (el) el.classList.toggle('is-active', btn.match);
  });
}

function cropperRotate(deg) {
  if (cropperInstance) {
    cropperInstance.rotate(deg);
  }
}

function cropperZoom(ratio) {
  if (cropperInstance) {
    cropperInstance.zoom(ratio);
  }
}

function cropperReset() {
  if (cropperInstance) {
    cropperInstance.reset();
    setCropRatio(NaN);
  }
}

async function applyCroppedImage() {
  if (!cropperInstance || !activeCropperTarget) return;

  const targetId = activeCropperTarget;
  const croppedCanvas = cropperInstance.getCroppedCanvas({
    maxWidth: 1024,
    maxHeight: 1024,
    imageSmoothingEnabled: true,
    imageSmoothingQuality: 'high',
  });

  if (!croppedCanvas) {
    alert('영역을 잘라내는 중 오류가 발생했습니다.');
    return;
  }

  croppedCanvas.toBlob(async (blob) => {
    if (!blob) return;

    try {
      const { dataUrl, sizeInKb, width, height } = await compressImage(blob, 800, 0.88);

      if (targetId === 1) photo1Data = dataUrl;
      else if (targetId === 2) photo2Data = dataUrl;
      else if (targetId === 'celeb') photoCelebData = dataUrl;

      // Update preview image
      const previewImg = document.getElementById(`preview-img-${targetId}`);
      if (previewImg) previewImg.src = dataUrl;

      // Show cropped badge & restore button
      document.getElementById(`badge-crop-${targetId}`)?.classList.remove('hidden');
      document.getElementById(`btn-restore-${targetId}`)?.classList.remove('hidden');

      // Update info text
      const infoEl = document.getElementById(`file-info-${targetId}`);
      if (infoEl) {
        infoEl.textContent = `선택 영역 (${width}×${height}, ${sizeInKb} KB)`;
      }

      closeCropper();
      if (targetId === 'celeb') refreshCelebExtraMatches();
      updateCompareButtonState();
      updateCelebButtonState();
      lucide.createIcons();
    } catch (err) {
      console.error('크롭 이미지 압축 실패:', err);
      const dataUrl = croppedCanvas.toDataURL('image/jpeg', 0.85);
      if (targetId === 1) photo1Data = dataUrl;
      else if (targetId === 2) photo2Data = dataUrl;
      else if (targetId === 'celeb') photoCelebData = dataUrl;
      const previewImg = document.getElementById(`preview-img-${targetId}`);
      if (previewImg) previewImg.src = dataUrl;
      document.getElementById(`badge-crop-${targetId}`)?.classList.remove('hidden');
      document.getElementById(`btn-restore-${targetId}`)?.classList.remove('hidden');
      closeCropper();
      updateCompareButtonState();
      updateCelebButtonState();
      lucide.createIcons();
    }
  }, 'image/jpeg', 0.9);
}

function restoreOriginalPhoto(targetId) {
  const originalData = targetId === 1 ? photo1OriginalData : (targetId === 2 ? photo2OriginalData : photoCelebOriginalData);
  if (!originalData) return;

  if (targetId === 1) photo1Data = originalData;
  else if (targetId === 2) photo2Data = originalData;
  else if (targetId === 'celeb') photoCelebData = originalData;
  if (targetId === 'celeb') refreshCelebExtraMatches();

  const previewImg = document.getElementById(`preview-img-${targetId}`);
  if (previewImg) previewImg.src = originalData;

  document.getElementById(`badge-crop-${targetId}`)?.classList.add('hidden');
  document.getElementById(`btn-restore-${targetId}`)?.classList.add('hidden');
  const infoEl = document.getElementById(`file-info-${targetId}`);
  if (infoEl) infoEl.textContent = '원본 사진으로 복원됨';

  updateCompareButtonState();
  updateCelebButtonState();
}

function updateCompareButtonState() {
  const btn = document.getElementById('btn-compare');
  const hint = document.getElementById('compare-hint');
  if (!btn) return;

  if (photo1Data && photo2Data) {
    btn.disabled = false;
    hint.textContent = '준비됐어요. 분석을 시작하세요.';
    hint.className = 'text-xs text-indigo-300 mt-2 text-center';
  } else {
    btn.disabled = true;
    hint.textContent = '두 사진을 모두 올리면 분석할 수 있어요.';
    hint.className = 'text-xs text-slate-500 mt-2 text-center';
  }
}

function updateCelebButtonState() {
  const btn = document.getElementById('btn-find-celeb');
  const hint = document.getElementById('celeb-hint');
  if (!btn) return;

  if (photoCelebData) {
    btn.disabled = false;
    if (hint) {
      hint.textContent = '준비됐어요. 사진을 더 추가하면 더 정확해요.';
      hint.className = 'text-xs text-indigo-300 mt-2 text-center';
    }
  } else {
    btn.disabled = true;
    if (hint) {
      hint.textContent = '내 사진을 올리면 찾을 수 있어요.';
      hint.className = 'text-xs text-slate-500 mt-2 text-center';
    }
  }
}

// Clipboard Paste Support (Ctrl+V)
function setupClipboardPaste() {
  window.addEventListener('paste', (e) => {
    const items = e.clipboardData?.items;
    if (!items) return;

    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile();
        if (currentMainTab === 'celeb') {
          setPhotoFromBlob(file, 'celeb', '클립보드_내사진.png');
        } else {
          if (!photo1Data) {
            setPhotoFromBlob(file, 1, '클립보드_사진1.png');
          } else if (!photo2Data) {
            setPhotoFromBlob(file, 2, '클립보드_사진2.png');
          } else {
            setPhotoFromBlob(file, 2, '클립보드_사진2.png');
          }
        }
        break;
      }
    }
  });
}

// Camera Capture
// Like phone camera apps: the front camera preview is mirrored and the photo is saved as previewed;
// the rear camera is shown and saved as-is.
let webcamFacing = 'user'; // 'user' (front) | 'environment' (rear)
let webcamMirrored = true;

async function startWebcamStream() {
  const video = document.getElementById('webcam-video');
  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
  webcamStream = await navigator.mediaDevices.getUserMedia({
    video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: { ideal: webcamFacing } }
  });
  video.srcObject = webcamStream;

  // Desktop webcams often report no facingMode; treat them as front cameras
  const actual = webcamStream.getVideoTracks()[0]?.getSettings().facingMode || webcamFacing;
  webcamMirrored = actual !== 'environment';
  video.style.transform = webcamMirrored ? 'scaleX(-1)' : '';
  const label = document.getElementById('webcam-facing-label');
  if (label) label.textContent = actual === 'environment' ? '후면 카메라' : '전면 카메라 (거울 모드)';
}

async function openWebcam(targetId) {
  activeWebcamTarget = targetId;
  const modal = document.getElementById('webcam-modal');

  try {
    await startWebcamStream();
    modal.classList.remove('hidden');
    modal.classList.add('flex');

    // Offer front/rear switching only when the device has more than one camera
    const devices = await navigator.mediaDevices.enumerateDevices();
    const cameras = devices.filter(d => d.kind === 'videoinput').length;
    document.getElementById('btn-webcam-switch')?.classList.toggle('hidden', cameras < 2);
    lucide.createIcons();
  } catch (err) {
    alert(`카메라에 접근할 수 없습니다: ${err.message}`);
  }
}

async function switchWebcamFacing() {
  const previous = webcamFacing;
  webcamFacing = webcamFacing === 'user' ? 'environment' : 'user';
  try {
    await startWebcamStream();
  } catch (err) {
    webcamFacing = previous;
    alert(`카메라를 전환할 수 없습니다: ${err.message}`);
    await startWebcamStream().catch(() => closeWebcam());
  }
}

function closeWebcam() {
  const modal = document.getElementById('webcam-modal');
  modal.classList.add('hidden');
  modal.classList.remove('flex');

  if (webcamStream) {
    webcamStream.getTracks().forEach(track => track.stop());
    webcamStream = null;
  }
}

function captureWebcam() {
  const video = document.getElementById('webcam-video');
  const canvas = document.getElementById('webcam-canvas');

  canvas.width = video.videoWidth || 640;
  canvas.height = video.videoHeight || 480;
  const ctx = canvas.getContext('2d');
  if (webcamMirrored) {
    // Save exactly what the mirrored preview showed
    ctx.translate(canvas.width, 0);
    ctx.scale(-1, 1);
  }
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  canvas.toBlob((blob) => {
    if (blob && activeWebcamTarget) {
      setPhotoFromBlob(blob, activeWebcamTarget, `webcam_${Date.now()}.jpg`);
    }
    closeWebcam();
  }, 'image/jpeg', 0.9);
}

// Execution: Compare Images
async function startComparison() {
  if (!photo1Data || !photo2Data) return;

  const btnCompare = document.getElementById('btn-compare');
  const loadingSection = document.getElementById('loading-section');
  const resultsSection = document.getElementById('results-section');
  const scanLine1 = document.getElementById('scan-line-1');
  const scanLine2 = document.getElementById('scan-line-2');

  // UI state
  btnCompare.disabled = true;
  loadingSection.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  scanLine1.classList.remove('hidden');
  scanLine2.classList.remove('hidden');

  loadingSection.scrollIntoView({ behavior: 'smooth', block: 'center' });

  // Progress animation steps
  const steps = [
    { title: '얼굴을 찾는 중...', step: '두 사진에서 얼굴을 찾아 정렬하고 있어요', progress: 25 },
    { title: '얼굴 특징 비교 중...', step: '얼굴 인식 AI가 이목구비 구조를 비교하고 있어요', progress: 50 },
    { title: '부위별로 살펴보는 중...', step: '눈·코·입·얼굴형을 하나씩 비교하고 있어요', progress: 75 },
    { title: '결과 정리 중...', step: '닮은 점과 다른 점을 정리하고 있어요', progress: 90 },
  ];

  let stepIdx = 0;
  const progressTimer = setInterval(() => {
    stepIdx = (stepIdx + 1) % steps.length;
    document.getElementById('loading-title').textContent = steps[stepIdx].title;
    document.getElementById('loading-step').textContent = steps[stepIdx].step;
    document.getElementById('loading-bar').style.width = `${steps[stepIdx].progress}%`;
  }, 2200);

  try {
    const payload = {
      image1_base64: photo1Data,
      image2_base64: photo2Data,
      mode: currentMode
    };

    const res = await fetch(BASE_URL + '/api/compare-json', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    clearInterval(progressTimer);

    if (!res.ok) {
      if (res.status === 413) {
        throw new Error('사진 파일 용량이 전송 제한을 초과했습니다. 자동으로 최적화되도록 사진을 다시 업로드해 주세요.');
      }
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `서버 오류 (${res.status})`);
    }

    const json = await res.json();
    currentAnalysisResult = json.data;
    renderResults(json.data);

  } catch (err) {
    clearInterval(progressTimer);
    alert(`분석 중 오류가 발생했습니다: ${err.message}`);
  } finally {
    btnCompare.disabled = false;
    loadingSection.classList.add('hidden');
    scanLine1.classList.add('hidden');
    scanLine2.classList.add('hidden');
  }
}

// Render Results View
function renderResults(data) {
  document.getElementById('compare-result-img1').src = photo1Data || '';
  document.getElementById('compare-result-img2').src = photo2Data || '';
  const resultsSection = document.getElementById('results-section');
  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const score = Math.max(0, Math.min(100, data.similarity_score || 0));
  const verdict = data.verdict || '판정 완료';
  const verdictSummary = data.verdict_summary || '';
  const detailed = data.detailed_scores || {};

  // Animate circular gauge
  const circle = document.getElementById('score-circle');
  const scoreValue = document.getElementById('score-value');
  const radius = 42;
  const circumference = 2 * Math.PI * radius; // ~263.89
  const offset = circumference - (score / 100) * circumference;

  let strokeColor = '#6366f1';
  let badgeClass = 'bg-indigo-500/20 text-indigo-300 border border-indigo-500/30';
  let dotColor = 'bg-indigo-400';

  if (score >= 80) {
    strokeColor = '#10b981'; // Green
    badgeClass = 'bg-emerald-500/20 text-emerald-300 border border-emerald-500/30';
    dotColor = 'bg-emerald-400';
  } else if (score >= 60) {
    strokeColor = '#0ea5e9'; // Sky / Blue
    badgeClass = 'bg-sky-500/20 text-sky-300 border border-sky-500/30';
    dotColor = 'bg-sky-400';
  } else if (score >= 45) {
    strokeColor = '#f59e0b'; // Amber
    badgeClass = 'bg-amber-500/20 text-amber-300 border border-amber-500/30';
    dotColor = 'bg-amber-400';
  } else {
    strokeColor = '#ef4444'; // Red
    badgeClass = 'bg-rose-500/20 text-rose-300 border border-rose-500/30';
    dotColor = 'bg-rose-400';
  }

  circle.style.stroke = strokeColor;
  circle.style.strokeDashoffset = offset;

  // Counter number animation
  let currentVal = 0;
  const stepTime = 15;
  const increment = Math.ceil(score / 40) || 1;
  const counter = setInterval(() => {
    currentVal += increment;
    if (currentVal >= score) {
      currentVal = score;
      clearInterval(counter);
    }
    scoreValue.textContent = `${currentVal}%`;
  }, stepTime);

  // Verdict Badge
  const badge = document.getElementById('verdict-badge');
  badge.className = `inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-bold mb-2 ${badgeClass}`;
  document.getElementById('verdict-dot').className = `w-2 h-2 rounded-full ${dotColor}`;
  document.getElementById('verdict-text').textContent = verdict;
  document.getElementById('verdict-summary').textContent = verdictSummary;

  // Description text
  document.getElementById('verdict-desc').textContent = 
    score >= 85 ? '눈매, 콧날, 웃는 모습과 얼굴형에서 감탄이 나올 만큼 높은 유전적 붕어빵 싱크로율을 보여줍니다.'
    : score >= 70 ? '핵심 이목구비와 특유의 표정 습관이 매우 많이 닮아 한눈에 가족/닮은꼴임을 알 수 있습니다.'
    : score >= 50 ? '전체적인 인상과 특정 이목구비에서 상당한 유사성이 관찰되는 닮은꼴입니다.'
    : score >= 35 ? '일부 특정 부위에서 닮은 느낌이 있으나, 각자의 뚜렷한 개성이 더 돋보입니다.'
    : '얼굴 골격과 이목구비 전반에 걸쳐 서로 다른 고유한 개성을 지니고 있습니다.';

  const breakdownEl = document.getElementById('score-breakdown');
  const bd = data.score_breakdown || {};
  if (breakdownEl && bd.face_recognition != null) {
    const llmPart = bd.llm != null && bd.face_weight < 1
      ? ` · AI 시각 판단 ${bd.llm}% (반영 ${Math.round((1 - bd.face_weight) * 100)}%)` : '';
    breakdownEl.textContent = `얼굴 인식 AI 측정 ${bd.face_recognition}% (반영 ${Math.round(bd.face_weight * 100)}%)${llmPart}`;
    breakdownEl.classList.remove('hidden');
  } else if (breakdownEl) {
    breakdownEl.classList.add('hidden');
  }

  // Component Scores
  const updateBar = (id, val) => {
    const v = Math.max(0, Math.min(100, val || 0));
    document.getElementById(`score-${id}`).textContent = `${v}%`;
    const bar = document.getElementById(`bar-${id}`);
    bar.style.width = `${v}%`;
  };

  updateBar('face-shape', detailed.face_shape);
  updateBar('eyes', detailed.eyes);
  updateBar('nose', detailed.nose);
  updateBar('mouth', detailed.mouth);
  updateBar('features', detailed.features);

  // Similarities List
  const simList = document.getElementById('list-similarities');
  simList.innerHTML = '';
  const sims = data.similarities || [];
  if (sims.length === 0) {
    simList.innerHTML = '<li class="text-slate-500 italic">감지된 공통점이 없습니다.</li>';
  } else {
    sims.forEach(item => {
      const li = document.createElement('li');
      li.className = 'flex items-start gap-2';
      li.innerHTML = `<span class="text-emerald-400">•</span><span>${escapeHtml(item)}</span>`;
      simList.appendChild(li);
    });
  }

  // Differences List
  const diffList = document.getElementById('list-differences');
  diffList.innerHTML = '';
  const diffs = data.differences || [];
  if (diffs.length === 0) {
    diffList.innerHTML = '<li class="text-slate-500 italic">감지된 차이점이 없습니다.</li>';
  } else {
    diffs.forEach(item => {
      const li = document.createElement('li');
      li.className = 'flex items-start gap-2';
      li.innerHTML = `<span class="text-amber-400">•</span><span>${escapeHtml(item)}</span>`;
      diffList.appendChild(li);
    });
  }

  // Environmental Factors
  document.getElementById('text-environmental').textContent = 
    data.environmental_factors || '외적 변수 영향 없음';

  // Comprehensive Text
  document.getElementById('text-comprehensive').textContent = 
    data.comprehensive_analysis || '종합 분석 완료';

  lucide.createIcons();
  prepareResultCard('compare');

}

function copyReport() {
  if (!currentAnalysisResult) return;

  const r = currentAnalysisResult;
  const modeLabel = currentMode === 'family' ? '가족·붕어빵 닮음도' : currentMode === 'celebrity' ? '닮은꼴 싱크로율' : '동일 인물 정밀 대조';
  const reportText = `[FaceMatch 두 사람 닮음 분석 · ${modeLabel}]
닮음 지수 ${r.similarity_score}% · ${r.verdict}
${r.verdict_summary}

부위별: 눈 ${r.detailed_scores?.eyes || 0}% · 코 ${r.detailed_scores?.nose || 0}% · 입 ${r.detailed_scores?.mouth || 0}% · 얼굴형 ${r.detailed_scores?.face_shape || 0}% · 전체 인상 ${r.detailed_scores?.features || 0}%

닮은 점
${(r.similarities || []).map(x => `- ${x}`).join('\n')}

다른 점
${(r.differences || []).map(x => `- ${x}`).join('\n')}

종합 소견
${r.comprehensive_analysis || '-'}
`;

  navigator.clipboard.writeText(reportText).then(() => {
    alert('결과를 글로 복사했어요.');
  }).catch(() => {
    alert('클립보드 복사에 실패했습니다.');
  });
}

function resetAll() {
  clearPhoto(1);
  clearPhoto(2);
  document.getElementById('results-section').classList.add('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==========================================
// Celebrity Lookalike Finder Execution & UI
// ==========================================
async function startCelebritySearch() {
  if (!photoCelebData) return;

  const btn = document.getElementById('btn-find-celeb');
  const loadingSection = document.getElementById('celeb-loading-section');
  const resultsSection = document.getElementById('celeb-results-section');
  const scanLine = document.getElementById('scan-line-celeb');

  btn.disabled = true;
  loadingSection.classList.remove('hidden');
  resultsSection.classList.add('hidden');
  scanLine?.classList.remove('hidden');

  loadingSection.scrollIntoView({ behavior: 'smooth', block: 'center' });

  const steps = [
    { title: '얼굴 특징 추출 중...', step: '얼굴을 찾아 특징을 읽고 있어요', progress: 25 },
    { title: '연예인과 비교 중...', step: '연예인 3천여 명의 얼굴과 비교하고 있어요', progress: 50 },
    { title: 'TOP 5 고르는 중...', step: '가장 닮은 순서대로 정리하고 있어요', progress: 75 },
    { title: '닮은 점 설명 작성 중...', step: 'AI가 사진을 비교해 설명을 쓰고 있어요', progress: 90 },
  ];

  let stepIdx = 0;
  const timer = setInterval(() => {
    stepIdx = (stepIdx + 1) % steps.length;
    document.getElementById('celeb-loading-title').textContent = steps[stepIdx].title;
    document.getElementById('celeb-loading-step').textContent = steps[stepIdx].step;
    document.getElementById('celeb-loading-bar').style.width = `${steps[stepIdx].progress}%`;
  }, 2000);

  try {
    const payload = {
      image_base64: photoCelebData,
      extra_images_base64: celebExtraPayload(),
      gender_filter: selectedCelebGender,
      describe: false  // ranking in ~2s; the detailed write-up follows via fetchCelebDescriptions
    };
    const token = ++celebSearchToken;

    const res = await fetch(BASE_URL + '/api/find-celebrity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });

    clearInterval(timer);

    if (!res.ok) {
      const err = await res.json().catch(() => ({ detail: res.statusText }));
      throw new Error(err.detail || `서버 오류 (${res.status})`);
    }

    const json = await res.json();
    renderCelebrityResults(json);
    fetchCelebDescriptions(json, token);

  } catch (err) {
    clearInterval(timer);
    alert(`연예인 분석 중 오류가 발생했습니다: ${err.message}`);
  } finally {
    btn.disabled = false;
    loadingSection.classList.add('hidden');
    scanLine?.classList.add('hidden');
  }
}

function escapeHtml(text) {
  return String(text ?? '').replace(/[&<>"']/g, ch => (
    { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[ch]
  ));
}

const PART_LABELS = { eyes: '눈', nose: '코', mouth: '입', face_shape: '얼굴형', features: '인상' };
let celebSearchToken = 0;

function renderCelebrityResults(data, { scroll = true } = {}) {
  const resultsSection = document.getElementById('celeb-results-section');
  resultsSection.classList.remove('hidden');
  if (scroll) resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
  lastCelebResult = data;
  const pending = !!data.descriptions_pending;
  const waiting = '설명 작성 중…';

  const top = data.top_celebrity || {};
  const vibe = data.face_features || {};
  const candidates = data.candidates || [];

  const note = document.getElementById('celeb-photo-note');
  const used = data.photos_used || 1;
  const rejected = data.photos_rejected || 0;
  note.textContent = rejected > 0
    ? `사진 ${used}장을 종합했어요 · 다른 사람으로 보이는 ${rejected}장은 제외`
    : used > 1 ? `사진 ${used}장을 종합했어요` : '';
  note.classList.toggle('hidden', !note.textContent);

  // 1. Photos, name, score ring
  document.getElementById('celeb-result-user-img').src = photoCelebData;
  document.getElementById('celeb-result-match-img').src = top.photo_url || '';
  document.getElementById('top-celeb-label').textContent = top.name || '연예인';
  document.getElementById('top-celeb-name').textContent = top.name || '알 수 없음';
  document.getElementById('top-celeb-cat').textContent = top.category || '';

  const score = Math.max(0, Math.min(100, Number(top.similarity_percent) || 0));
  const circumference = 2 * Math.PI * 42;
  document.getElementById('celeb-score-circle').style.strokeDashoffset = circumference - (score / 100) * circumference;
  const scoreValue = document.getElementById('celeb-score-value');
  if (scroll) {
    let currentVal = 0;
    const counter = setInterval(() => {
      currentVal = Math.min(score, currentVal + 2);
      scoreValue.textContent = `${currentVal}%`;
      if (currentVal >= score) clearInterval(counter);
    }, 20);
  } else {
    scoreValue.textContent = `${score}%`;
  }

  // One wording source: the server's match strength (percentile among everyone's #1 match)
  const strength = data.match_strength;
  const badge = document.getElementById('celeb-match-badge');
  badge.textContent = strength?.label || '가장 가까운 후보';
  badge.className = (strength?.percentile || 0) >= 60 ? 'chip chip-pink' : 'chip';
  document.getElementById('celeb-strength').textContent = strength?.detail || '';
  document.getElementById('top-celeb-summary').textContent = top.summary || (pending ? waiting : '');

  // 2. Where they look alike: bars, reason, per-part notes, differences, keywords
  const det = top.detailed_scores || {};
  ['eyes', 'nose', 'mouth', 'face_shape', 'features'].forEach(k => {
    const id = k === 'face_shape' ? 'face-shape' : k;
    const v = Math.max(0, Math.min(100, Number(det[k]) || score));
    document.getElementById(`celeb-score-${id}`).textContent = `${v}%`;
    document.getElementById(`celeb-bar-${id}`).style.width = `${v}%`;
  });
  document.getElementById('top-celeb-reason').textContent = top.reason || (pending ? waiting : '');

  const notes = top.part_notes || {};
  document.getElementById('top-celeb-part-notes').innerHTML = Object.keys(PART_LABELS)
    .filter(k => notes[k])
    .map(k => `<div class="flex gap-3"><dt class="w-12 shrink-0 text-xs font-semibold text-indigo-300 pt-0.5">${PART_LABELS[k]}</dt><dd class="text-slate-300 leading-relaxed">${escapeHtml(notes[k])}</dd></div>`)
    .join('');

  const diffs = top.differences || [];
  document.getElementById('top-celeb-differences-wrap').classList.toggle('hidden', !diffs.length);
  document.getElementById('top-celeb-differences').innerHTML = diffs.map(d => `<li>• ${escapeHtml(d)}</li>`).join('');

  document.getElementById('top-celeb-points').innerHTML =
    (top.matching_points || []).slice(0, 4).map(p => `<span class="chip">${escapeHtml(p)}</span>`).join('');

  // 3. My face features
  const setText = (id, value) => { document.getElementById(id).textContent = value || (pending ? waiting : '-'); };
  document.getElementById('user-vibe-meta').textContent = [vibe.gender, vibe.age_group].filter(Boolean).join(' · ');
  document.getElementById('user-vibe-keywords').innerHTML =
    (vibe.keywords || []).map(k => `<span class="chip chip-pink">${escapeHtml(k)}</span>`).join('');
  setText('user-vibe-type', vibe.face_type);
  setText('user-vibe-shape', vibe.face_shape);
  setText('user-vibe-eyes', vibe.eyes);
  setText('user-vibe-nose', vibe.nose || (!pending && vibe.nose_mouth) || '');
  setText('user-vibe-mouth', vibe.mouth);
  setText('user-vibe-overall', vibe.overall_vibe);

  // 4. Other candidates: simple rows (photo, name, category, score); only #1 gets an AI write-up
  const candContainer = document.getElementById('celeb-candidates-container');
  candContainer.innerHTML = '';
  candidates.forEach(c => {
    const pct = Number(c.similarity_percent) || 0;
    const row = document.createElement('div');
    row.className = 'flex items-center gap-3 py-3 px-1';
    row.innerHTML = `
      <span class="w-5 text-center text-sm font-bold text-slate-500 shrink-0">${Number(c.rank) || ''}</span>
      <img src="${escapeHtml(c.photo_url || '')}" alt="${escapeHtml(c.name)}" class="w-14 h-14 rounded-xl object-cover bg-slate-800 shrink-0" onerror="this.style.visibility='hidden'">
      <div class="flex-1 min-w-0">
        <p class="text-sm font-semibold text-white truncate">${escapeHtml(c.name)}</p>
        <p class="text-xs text-slate-500 truncate">${escapeHtml(c.category || '')}${Math.abs(score - pct) <= 3 ? ' · 1위와 비슷한 수준' : ''}</p>
      </div>
      <span class="chip shrink-0">${pct}%</span>
    `;
    candContainer.appendChild(row);
  });

  prepareResultCard('celeb');
  lucide.createIcons();
}

// Second phase: fetch the Gemma write-up in two parts so each request stays short
// (the public reverse proxy cuts requests at 60s): 1) my features + #1 (~20s), 2) #2..#5 (~30s).
const DESCRIBE_TIMEOUT_MS = 80000;

async function requestCelebDescription(data, part) {
  const all = data.all_celebrities || [];
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), DESCRIBE_TIMEOUT_MS);
  try {
    const res = await fetch(BASE_URL + '/api/describe-celebrity', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      signal: controller.signal,
      body: JSON.stringify({
        image_base64: photoCelebData,
        qids: all.map(c => c.qid),
        percents: all.map(c => c.similarity_percent),
        part
      })
    });
    const json = res.ok ? await res.json() : null;
    if (!json?.success) throw new Error(`describe ${part} failed`);
    return json;
  } finally {
    clearTimeout(timer);
  }
}

function mergeCelebDescription(data, desc) {
  const byQid = Object.fromEntries((desc.matches || []).map(m => [m.qid, m]));
  // keep rank / score / photo from the ranking, take the texts from the description
  const all = (data.all_celebrities || []).map(c => {
    const d = byQid[c.qid];
    return d ? { ...c, ...d, rank: c.rank, similarity_percent: c.similarity_percent, cosine: c.cosine } : c;
  });
  return {
    ...data,
    face_features: desc.face_features || data.face_features,
    all_celebrities: all,
    top_celebrity: all[0],
    candidates: all.slice(1),
  };
}

function setDescribeStatus(text, { spinning = true, retry = false } = {}) {
  const status = document.getElementById('celeb-describe-status');
  status.classList.toggle('hidden', !text);
  document.getElementById('celeb-describe-text').textContent = text || '';
  status.querySelector('span')?.classList.toggle('hidden', !spinning);
  document.getElementById('celeb-describe-retry').classList.toggle('hidden', !retry);
}

async function fetchCelebDescriptions(data, token) {
  let current = { ...data, descriptions_pending: true };
  const parts = [
    { part: 'top', label: '1위 연예인과 내 얼굴 특징을 분석하고 있어요 (약 10초)' },
  ];
  for (const { part, label } of parts) {
    // skip parts that already arrived (e.g. when retrying after a failure)
    if (part === 'top' && current.top_celebrity?.reason) continue;
    setDescribeStatus(label);
    try {
      const desc = await requestCelebDescription(current, part);
      if (token !== celebSearchToken) return;  // a newer search replaced this one
      current = mergeCelebDescription(current, desc);
      renderCelebrityResults(current, { scroll: false });
    } catch (err) {
      if (token !== celebSearchToken) return;
      renderCelebrityResults({ ...current, descriptions_pending: false }, { scroll: false });
      setDescribeStatus('설명을 불러오지 못했어요. 순위와 점수는 그대로예요.', { spinning: false, retry: true });
      return;
    }
  }
  current.descriptions_pending = false;
  renderCelebrityResults(current, { scroll: false });
  setDescribeStatus('');
}

function retryCelebDescriptions() {
  if (lastCelebResult) fetchCelebDescriptions(lastCelebResult, celebSearchToken);
}

let lastCelebResult = null;

function resetCelebSearch() {
  document.getElementById('celeb-results-section').classList.add('hidden');
  clearCelebPhoto();
  lastCelebResult = null;
  resultCardCache.celeb = null;
  celebSearchToken++;
  document.getElementById('celeb-describe-status').classList.add('hidden');
  window.scrollTo({ top: 0, behavior: 'smooth' });
}

// ==========================================
// Result image card (save / share)
// ==========================================
// The result is drawn on a canvas as a 1080px-wide portrait card so people can save or share
// just the result. Cards are pre-rendered when results arrive: iOS Safari only opens the share
// sheet right after a tap, so there is no time to build the image at click time.
const CARD_W = 1080;
const CARD_PAD = 72;
const CARD_FONT = "'Pretendard', 'Apple SD Gothic Neo', 'Malgun Gothic', 'Noto Sans KR', sans-serif";
const resultCardCache = { compare: null, celeb: null };

function loadImage(src) {
  return new Promise(resolve => {
    if (!src) return resolve(null);
    const img = new Image();
    img.onload = () => resolve(img);
    img.onerror = () => resolve(null);
    img.src = src;
  });
}

function roundRectPath(ctx, x, y, w, h, r) {
  ctx.beginPath();
  ctx.moveTo(x + r, y);
  ctx.arcTo(x + w, y, x + w, y + h, r);
  ctx.arcTo(x + w, y + h, x, y + h, r);
  ctx.arcTo(x, y + h, x, y, r);
  ctx.arcTo(x, y, x + w, y, r);
  ctx.closePath();
}

function drawCoverImage(ctx, img, x, y, size, radius) {
  ctx.save();
  roundRectPath(ctx, x, y, size, size, radius);
  ctx.clip();
  ctx.fillStyle = '#1e293b';
  ctx.fillRect(x, y, size, size);
  if (img) {
    const s = Math.min(img.width, img.height);
    ctx.drawImage(img, (img.width - s) / 2, (img.height - s) / 2, s, s, x, y, size, size);
  }
  ctx.restore();
}

// Word-based wrapping (Korean breaks between words); overly long words break by character
function wrapLines(ctx, text, maxWidth, maxLines = 0) {
  const words = String(text || '').split(/\s+/).filter(Boolean);
  const lines = [];
  let line = '';
  for (const word of words) {
    const test = line ? `${line} ${word}` : word;
    if (ctx.measureText(test).width <= maxWidth) {
      line = test;
      continue;
    }
    if (line) lines.push(line);
    line = word;
    while (ctx.measureText(line).width > maxWidth) {
      let i = line.length;
      while (i > 1 && ctx.measureText(line.slice(0, i)).width > maxWidth) i--;
      lines.push(line.slice(0, i));
      line = line.slice(i);
    }
  }
  if (line) lines.push(line);
  if (maxLines && lines.length > maxLines) {
    const kept = lines.slice(0, maxLines);
    let last = kept[maxLines - 1];
    while (last && ctx.measureText(`${last}…`).width > maxWidth) last = last.slice(0, -1);
    kept[maxLines - 1] = `${last}…`;
    return kept;
  }
  return lines;
}

// Draws (wrapped) text and returns the y just below it
function drawText(ctx, text, x, y, { size = 32, weight = 400, color = '#e2e8f0', align = 'left', maxWidth = 0, lineHeight = 1.45, maxLines = 0 } = {}) {
  ctx.font = `${weight} ${size}px ${CARD_FONT}`;
  ctx.fillStyle = color;
  ctx.textAlign = align;
  ctx.textBaseline = 'top';
  const lines = maxWidth ? wrapLines(ctx, text, maxWidth, maxLines) : [String(text || '')];
  lines.forEach((l, i) => ctx.fillText(l, x, y + i * size * lineHeight));
  return y + lines.length * size * lineHeight;
}

function drawLabel(ctx, text, x, y, alignRight = false) {
  ctx.font = `700 26px ${CARD_FONT}`;
  const w = ctx.measureText(text).width + 28;
  const bx = alignRight ? x - w : x;
  ctx.fillStyle = 'rgba(0,0,0,0.6)';
  roundRectPath(ctx, bx, y, w, 44, 12);
  ctx.fill();
  drawText(ctx, text, bx + 14, y + 8, { size: 26, weight: 700, color: '#fff' });
}

function drawRing(ctx, cx, cy, r, pct, color) {
  ctx.fillStyle = '#111827';
  ctx.beginPath(); ctx.arc(cx, cy, r + 16, 0, Math.PI * 2); ctx.fill();
  ctx.lineWidth = 18;
  ctx.lineCap = 'round';
  ctx.strokeStyle = 'rgba(255,255,255,0.12)';
  ctx.beginPath(); ctx.arc(cx, cy, r, 0, Math.PI * 2); ctx.stroke();
  ctx.strokeStyle = color;
  ctx.beginPath(); ctx.arc(cx, cy, r, -Math.PI / 2, -Math.PI / 2 + Math.PI * 2 * Math.max(0.01, pct / 100)); ctx.stroke();
  ctx.font = `800 54px ${CARD_FONT}`;
  ctx.fillStyle = '#fff';
  ctx.textAlign = 'center';
  ctx.textBaseline = 'middle';
  ctx.fillText(`${pct}%`, cx, cy + 2);
}

function drawPartBars(ctx, scores, x, y, width) {
  const items = [['눈', scores.eyes], ['코', scores.nose], ['입', scores.mouth], ['얼굴형', scores.face_shape], ['전체 인상', scores.features]];
  const colW = (width - 48) / 2;
  items.forEach(([label, value], i) => {
    const full = i === 4;
    const cx = full ? x : x + (i % 2) * (colW + 48);
    const cy = y + Math.floor(i / 2) * 92;
    const w = full ? width : colW;
    const v = Math.max(0, Math.min(100, Number(value) || 0));
    drawText(ctx, label, cx, cy, { size: 28, weight: 600, color: '#cbd5e1' });
    drawText(ctx, `${v}%`, cx + w, cy, { size: 28, weight: 700, color: '#e2e8f0', align: 'right' });
    ctx.fillStyle = '#1e293b';
    roundRectPath(ctx, cx, cy + 46, w, 14, 7); ctx.fill();
    const grad = ctx.createLinearGradient(cx, 0, cx + w, 0);
    grad.addColorStop(0, '#6366f1'); grad.addColorStop(1, '#a855f7');
    ctx.fillStyle = grad;
    roundRectPath(ctx, cx, cy + 46, Math.max(14, w * v / 100), 14, 7); ctx.fill();
  });
  return y + 3 * 92;
}

function drawDivider(ctx, y) {
  ctx.fillStyle = 'rgba(255,255,255,0.08)';
  ctx.fillRect(CARD_PAD, y, CARD_W - CARD_PAD * 2, 2);
  return y + 2;
}

async function buildResultCanvas(kind) {
  if (document.fonts?.ready) await document.fonts.ready;
  const canvas = document.createElement('canvas');
  canvas.width = CARD_W;
  canvas.height = 8000;
  const ctx = canvas.getContext('2d');
  const P = CARD_PAD, W = CARD_W - CARD_PAD * 2, mid = CARD_W / 2;
  ctx.fillStyle = '#0b0f19';
  ctx.fillRect(0, 0, CARD_W, canvas.height);

  let y = P;
  drawText(ctx, 'FaceMatch', P, y, { size: 32, weight: 800, color: '#a5b4fc' });
  drawText(ctx, kind === 'celeb' ? '닮은 연예인 찾기' : '두 사람 닮음 분석', CARD_W - P, y + 2, { size: 28, color: '#64748b', align: 'right' });
  y += 96;

  const size = (W - 32) / 2;
  if (kind === 'celeb') {
    const data = lastCelebResult;
    const top = data.top_celebrity || {};
    const vibe = data.face_features || {};
    const labels = { eyes: '눈', nose: '코', mouth: '입', face_shape: '얼굴형' };
    y = drawText(ctx, '나와 가장 닮은 연예인', mid, y, { size: 38, weight: 700, color: '#cbd5e1', align: 'center' }) + 36;
    const [mine, theirs] = await Promise.all([loadImage(photoCelebData), loadImage(top.photo_url)]);
    drawCoverImage(ctx, mine, P, y, size, 32);
    drawCoverImage(ctx, theirs, P + size + 32, y, size, 32);
    drawLabel(ctx, '나', P + 18, y + 18);
    drawLabel(ctx, top.name || '', CARD_W - P - 18, y + 18, true);
    drawRing(ctx, mid, y + size, 92, Number(top.similarity_percent) || 0, '#ec4899');
    y += size + 92 + 48;
    y = drawText(ctx, top.name || '', mid, y, { size: 78, weight: 800, color: '#fff', align: 'center' });
    y = drawText(ctx, [top.category, data.match_strength?.label].filter(Boolean).join(' · '), mid, y + 6, { size: 30, weight: 600, color: '#f9a8d4', align: 'center' }) + 10;
    if (data.match_strength?.detail) y = drawText(ctx, data.match_strength.detail, mid, y, { size: 26, color: '#64748b', align: 'center', maxWidth: W }) + 10;
    y = drawText(ctx, top.summary || '', mid, y + 14, { size: 36, color: '#e2e8f0', align: 'center', maxWidth: W - 40, maxLines: 3 }) + 44;

    // Where they look alike
    y = drawDivider(ctx, y) + 44;
    y = drawText(ctx, '어디가 닮았나요?', P, y, { size: 36, weight: 700, color: '#fff' }) + 28;
    y = drawPartBars(ctx, top.detailed_scores || {}, P, y, W) + 8;
    if (top.reason) y = drawText(ctx, top.reason, P, y, { size: 30, color: '#e2e8f0', maxWidth: W }) + 24;
    const notes = top.part_notes || {};
    Object.keys(labels).filter(k => notes[k]).forEach(k => {
      drawText(ctx, labels[k], P, y, { size: 28, weight: 700, color: '#a5b4fc' });
      y = drawText(ctx, notes[k], P + 110, y, { size: 28, color: '#cbd5e1', maxWidth: W - 110 }) + 14;
    });
    if ((top.differences || []).length) {
      y = drawText(ctx, '다른 점', P, y + 10, { size: 28, weight: 700, color: '#fcd34d' }) + 8;
      top.differences.forEach(d => { y = drawText(ctx, `• ${d}`, P, y, { size: 28, color: '#94a3b8', maxWidth: W }) + 6; });
    }

    // My face features
    if (vibe.face_shape || vibe.overall_vibe) {
      y = drawDivider(ctx, y + 30) + 44;
      y = drawText(ctx, '내 얼굴 특징', P, y, { size: 36, weight: 700, color: '#fff' }) + 12;
      const meta = [vibe.gender, vibe.age_group, ...(vibe.keywords || [])].filter(Boolean).join(' · ');
      if (meta) y = drawText(ctx, meta, P, y, { size: 28, weight: 600, color: '#f9a8d4', maxWidth: W }) + 20;
      [['인상', vibe.face_type], ['얼굴형', vibe.face_shape], ['눈매', vibe.eyes],
       ['코', vibe.nose || vibe.nose_mouth], ['입', vibe.mouth]].forEach(([label, text]) => {
        if (!text || text === '분석 정보 없음') return;
        drawText(ctx, label, P, y, { size: 28, weight: 700, color: '#a5b4fc' });
        y = drawText(ctx, text, P + 110, y, { size: 28, color: '#cbd5e1', maxWidth: W - 110 }) + 14;
      });
      if (vibe.overall_vibe) y = drawText(ctx, vibe.overall_vibe, P, y + 8, { size: 30, color: '#e2e8f0', maxWidth: W }) + 10;
    }

    // Other candidates: photo, name, category, score
    const others = data.candidates || [];
    if (others.length) {
      y = drawDivider(ctx, y + 30) + 44;
      y = drawText(ctx, '다른 후보', P, y, { size: 36, weight: 700, color: '#fff' }) + 24;
      const imgs = await Promise.all(others.map(c => loadImage(c.photo_url)));
      others.forEach((c, i) => {
        drawText(ctx, String(c.rank), P, y + 30, { size: 30, weight: 700, color: '#64748b' });
        drawCoverImage(ctx, imgs[i], P + 44, y, 96, 22);
        drawText(ctx, c.name, P + 164, y + 10, { size: 34, weight: 700, color: '#fff' });
        drawText(ctx, c.category || '', P + 164, y + 56, { size: 26, color: '#64748b' });
        drawText(ctx, `${c.similarity_percent}%`, CARD_W - P, y + 28, { size: 34, weight: 700, color: '#c7d2fe', align: 'right' });
        y += 124;
      });
    }
  } else {
    const r = currentAnalysisResult;
    y = drawText(ctx, '두 사람은 얼마나 닮았을까?', mid, y, { size: 38, weight: 700, color: '#cbd5e1', align: 'center' }) + 36;
    const [a, b] = await Promise.all([loadImage(photo1Data), loadImage(photo2Data)]);
    drawCoverImage(ctx, a, P, y, size, 32);
    drawCoverImage(ctx, b, P + size + 32, y, size, 32);
    const ringColor = document.getElementById('score-circle')?.style.stroke || '#6366f1';
    drawRing(ctx, mid, y + size, 92, Number(r.similarity_score) || 0, ringColor);
    y += size + 92 + 48;
    y = drawText(ctx, r.verdict || '', mid, y, { size: 56, weight: 800, color: '#fff', align: 'center', maxWidth: W }) + 20;
    y = drawText(ctx, r.verdict_summary || '', mid, y, { size: 34, color: '#e2e8f0', align: 'center', maxWidth: W - 40, maxLines: 3 }) + 44;
    y = drawDivider(ctx, y) + 44;
    y = drawText(ctx, '부위별 닮음', P, y, { size: 36, weight: 700, color: '#fff' }) + 28;
    y = drawPartBars(ctx, r.detailed_scores || {}, P, y, W) + 8;
    const section = (title, items, color) => {
      if (!items?.length) return;
      y = drawText(ctx, title, P, y + 16, { size: 32, weight: 700, color }) + 12;
      items.slice(0, 2).forEach(t => {
        y = drawText(ctx, `• ${t}`, P, y, { size: 29, color: '#cbd5e1', maxWidth: W, maxLines: 2 }) + 8;
      });
    };
    section('닮은 점', r.similarities, '#6ee7b7');
    section('다른 점', r.differences, '#fcd34d');
  }

  y += 48;
  y = drawDivider(ctx, y) + 32;
  y = drawText(ctx, 'minohlee.mooo.com/facematching', mid, y, { size: 26, color: '#475569', align: 'center' }) + P - 20;

  const out = document.createElement('canvas');
  out.width = CARD_W;
  out.height = Math.ceil(y);
  out.getContext('2d').drawImage(canvas, 0, 0);
  return out;
}

function prepareResultCard(kind) {
  resultCardCache[kind] = buildResultCanvas(kind)
    .then(c => new Promise(resolve => c.toBlob(resolve, 'image/png')))
    .catch(err => { console.warn('결과 이미지 생성 실패:', err); return null; });
}

async function getResultCardBlob(kind) {
  if (!resultCardCache[kind]) prepareResultCard(kind);
  const blob = await resultCardCache[kind];
  if (!blob) throw new Error('image build failed');
  return blob;
}

async function saveResultImage(kind) {
  try {
    const blob = await getResultCardBlob(kind);
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = `facematch_${kind === 'celeb' ? 'celebrity' : 'compare'}_${Date.now()}.png`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    setTimeout(() => URL.revokeObjectURL(a.href), 5000);
  } catch (err) {
    alert('결과 이미지를 만들지 못했어요. 잠시 후 다시 시도해 주세요.');
  }
}

async function shareResultImage(kind) {
  try {
    const blob = await getResultCardBlob(kind);
    const file = new File([blob], `facematch_${kind}.png`, { type: 'image/png' });
    if (navigator.canShare && navigator.canShare({ files: [file] })) {
      await navigator.share({ files: [file], title: 'FaceMatch 결과' });
      return;
    }
    await saveResultImage(kind);
    alert('이 기기에서는 바로 공유할 수 없어 이미지로 저장했어요.');
  } catch (err) {
    if (err?.name !== 'AbortError') alert('공유하지 못했어요.');
  }
}

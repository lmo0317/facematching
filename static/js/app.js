// Global application state
let currentMainTab = 'compare'; // 'compare' | 'celeb'
let photo1Data = null; // dataURL (may be cropped/compressed)
let photo2Data = null; // dataURL (may be cropped/compressed)
let photo1OriginalData = null; // uncropped original dataURL
let photo2OriginalData = null; // uncropped original dataURL
let photoCelebData = null; // dataURL for celebrity search
let photoCelebOriginalData = null;
let celebExtraPhotos = []; // extra dataURLs of the same person (max 2), averaged server-side
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

  if (tab === 'compare') {
    secCompare.classList.remove('hidden');
    secCeleb.classList.add('hidden');
    btnCompare.className = 'px-5 sm:px-6 py-3 rounded-xl text-xs sm:text-sm font-bold transition flex items-center space-x-2 bg-gradient-to-r from-indigo-600 to-indigo-700 text-white shadow-lg shadow-indigo-600/30';
    btnCeleb.className = 'px-5 sm:px-6 py-3 rounded-xl text-xs sm:text-sm font-bold text-slate-400 hover:text-white transition flex items-center space-x-2 hover:bg-slate-800/60';
  } else {
    secCompare.classList.add('hidden');
    secCeleb.classList.remove('hidden');
    btnCeleb.className = 'px-5 sm:px-6 py-3 rounded-xl text-xs sm:text-sm font-bold transition flex items-center space-x-2 bg-gradient-to-r from-pink-600 to-indigo-600 text-white shadow-lg shadow-pink-600/30';
    btnCompare.className = 'px-5 sm:px-6 py-3 rounded-xl text-xs sm:text-sm font-bold text-slate-400 hover:text-white transition flex items-center space-x-2 hover:bg-slate-800/60';
  }
  lucide.createIcons();
}

function setCelebGenderFilter(gender) {
  selectedCelebGender = gender;
  ['auto', 'male', 'female'].forEach(g => {
    const btn = document.getElementById(`celeb-filter-${g}`);
    if (!btn) return;
    if (g === gender) {
      btn.className = 'px-3 py-1.5 rounded-md font-bold transition bg-gradient-to-r from-pink-600 to-indigo-600 text-white shadow';
    } else {
      btn.className = 'px-3 py-1.5 rounded-md font-medium text-slate-400 hover:text-white transition';
    }
  });
}

function setAnalysisMode(mode) {
  currentMode = mode;
  const btnText = document.querySelector('#btn-compare span');
  if (btnText) {
    btnText.textContent = '두 사람 얼마나 닮았나? 붕어빵 지수 측정';
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
      statusDot.className = 'w-2 h-2 rounded-full bg-emerald-400 animate-pulse';
      statusText.textContent = `Gemma 4 E4B 비전 연결됨 (${data.backend.target_model})`;
      statusPill.className = 'flex items-center space-x-2 px-3 py-1.5 rounded-full bg-emerald-950/40 border border-emerald-500/30 text-emerald-300';
    } else if (data.backend?.status === 'connected') {
      statusDot.className = 'w-2 h-2 rounded-full bg-amber-400 animate-pulse';
      statusText.textContent = 'Gemma 4 연결됨 (비전 확인 중)';
      statusPill.className = 'flex items-center space-x-2 px-3 py-1.5 rounded-full bg-amber-950/40 border border-amber-500/30 text-amber-300';
    } else {
      statusDot.className = 'w-2 h-2 rounded-full bg-rose-500';
      statusText.textContent = 'Gemma 4 서버 연결 실패';
      statusPill.className = 'flex items-center space-x-2 px-3 py-1.5 rounded-full bg-rose-950/40 border border-rose-500/30 text-rose-300';
    }
  } catch (err) {
    statusDot.className = 'w-2 h-2 rounded-full bg-rose-500';
    statusText.textContent = '백엔드 서버 응답 없음';
    statusPill.className = 'flex items-center space-x-2 px-3 py-1.5 rounded-full bg-rose-950/40 border border-rose-500/30 text-rose-300';
  }
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

async function handleCelebExtraFiles(e) {
  const files = Array.from(e.target.files || []).filter(f => f.type.startsWith('image/'));
  for (const file of files) {
    if (celebExtraPhotos.length >= MAX_CELEB_EXTRA) break;
    try {
      const { dataUrl } = await compressImage(file, 800, 0.85);
      celebExtraPhotos.push(dataUrl);
    } catch (err) {
      console.warn('추가 사진 처리 실패:', err);
    }
  }
  e.target.value = '';
  renderCelebExtraPhotos();
}

function removeCelebExtraPhoto(index) {
  celebExtraPhotos.splice(index, 1);
  renderCelebExtraPhotos();
}

function renderCelebExtraPhotos() {
  const list = document.getElementById('celeb-extra-list');
  const addBtn = document.getElementById('btn-add-celeb-extra');
  if (!list) return;
  list.innerHTML = '';
  celebExtraPhotos.forEach((url, i) => {
    const item = document.createElement('div');
    item.className = 'relative w-16 h-16 rounded-lg overflow-hidden border border-white/10 bg-slate-800';
    item.innerHTML = `
      <img src="${url}" alt="추가 사진 ${i + 1}" class="w-full h-full object-cover">
      <button type="button" onclick="removeCelebExtraPhoto(${i})" title="삭제"
        class="absolute top-0.5 right-0.5 w-5 h-5 rounded-full bg-black/70 text-white text-xs leading-5 text-center hover:bg-rose-600">×</button>`;
    list.appendChild(item);
  });
  if (addBtn) addBtn.disabled = celebExtraPhotos.length >= MAX_CELEB_EXTRA;
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

    if (faces.length > 0 && pill && countEl) {
      pill.classList.remove('hidden');
      countEl.textContent = `${faces.length}명의 인물 얼굴 감지됨`;
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
    if (!el) return;
    if (btn.match) {
      el.className = 'px-2.5 py-1 rounded-md bg-indigo-600 text-white font-medium transition shadow-sm';
    } else {
      el.className = 'px-2.5 py-1 rounded-md bg-slate-800 hover:bg-slate-700 text-slate-300 font-medium transition';
    }
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
    hint.textContent = '사진 2장이 준비되었습니다. 분석 시작 버튼을 누르세요.';
    hint.className = 'text-xs text-indigo-400 mt-2 font-medium';
  } else {
    btn.disabled = true;
    hint.textContent = '두 사진을 모두 등록하면 분석 버튼이 활성화됩니다.';
    hint.className = 'text-xs text-slate-400 mt-2';
  }
}

function updateCelebButtonState() {
  const btn = document.getElementById('btn-find-celeb');
  const hint = document.getElementById('celeb-hint');
  if (!btn) return;

  if (photoCelebData) {
    btn.disabled = false;
    if (hint) {
      hint.textContent = '내 사진 등록 완료! 버튼을 누르면 닮은 연예인 사진을 찾아옵니다.';
      hint.className = 'text-xs text-pink-400 mt-2 font-medium';
    }
  } else {
    btn.disabled = true;
    if (hint) {
      hint.textContent = '내 사진을 등록하면 닮은 연예인 찾기 버튼이 활성화됩니다.';
      hint.className = 'text-xs text-slate-400 mt-2';
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

// Webcam Capture
async function openWebcam(targetId) {
  activeWebcamTarget = targetId;
  const modal = document.getElementById('webcam-modal');
  const video = document.getElementById('webcam-video');

  try {
    webcamStream = await navigator.mediaDevices.getUserMedia({
      video: { width: { ideal: 1280 }, height: { ideal: 720 }, facingMode: 'user' }
    });
    video.srcObject = webcamStream;
    modal.classList.remove('hidden');
    modal.classList.add('flex');
    lucide.createIcons();
  } catch (err) {
    alert(`웹캠에 접근할 수 없습니다: ${err.message}`);
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
    { title: '사진 최적화 및 안면 검출 중...', step: '1단계: EXIF 회전 보정 및 768px 해상도 최적화', progress: 25 },
    { title: 'Gemma 4 E4B 비전 인코더 전송 중...', step: '2단계: 멀티모달 프로젝터를 통한 안면 특징 벡터 임베딩', progress: 50 },
    { title: '이목구비 골격 구조 정밀 대조 중...', step: '3단계: 얼굴형, 눈, 코, 입, 골격 비율 상호 비교', progress: 75 },
    { title: '최종 유사도 판정 및 감정서 작성 중...', step: '4단계: 종합 유사도 점수 산출 및 전문 소견 생성', progress: 90 },
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
  badge.className = `inline-flex items-center space-x-2 px-3 py-1 rounded-full text-xs font-bold mb-2 shadow-sm ${badgeClass}`;
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
    if (v >= 75) bar.className = 'bg-emerald-500 h-1.5 rounded-full transition-all duration-1000';
    else if (v >= 50) bar.className = 'bg-indigo-500 h-1.5 rounded-full transition-all duration-1000';
    else bar.className = 'bg-rose-500 h-1.5 rounded-full transition-all duration-1000';
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
      li.className = 'flex items-start space-x-2';
      li.innerHTML = `<span class="text-emerald-400 mt-0.5">•</span><span>${item}</span>`;
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
      li.className = 'flex items-start space-x-2';
      li.innerHTML = `<span class="text-amber-400 mt-0.5">•</span><span>${item}</span>`;
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
}

function copyReport() {
  if (!currentAnalysisResult) return;

  const r = currentAnalysisResult;
  const modeLabel = currentMode === 'family' ? '가족·붕어빵 닮음도' : currentMode === 'celebrity' ? '닮은꼴 싱크로율' : '동일 인물 정밀 대조';
  const reportText = `[FaceMatch AI - Gemma 4 E4B 붕어빵·닮은꼴 정밀 감정서]
- 분석 모드: ${modeLabel}
- 일시: ${new Date().toLocaleString()}
- 닮은꼴·붕어빵 지수: ${r.similarity_score}%
- 최종 판정: ${r.verdict}
- 핵심 소견: ${r.verdict_summary}

[부위별 붕어빵 닮음도]
- 얼굴형 & 턱선: ${r.detailed_scores?.face_shape || 0}%
- 눈매 & 눈웃음: ${r.detailed_scores?.eyes || 0}%
- 콧대 & 코끝: ${r.detailed_scores?.nose || 0}%
- 입술 & 하관/미소: ${r.detailed_scores?.mouth || 0}%
- 고유 인상 & 분위기: ${r.detailed_scores?.features || 0}%

[쏙 빼닮은 붕어빵 포인트]
${(r.similarities || []).map(s => `- ${s}`).join('\n')}

[각자의 개성적 포인트]
${(r.differences || []).map(d => `- ${d}`).join('\n')}

[나이 및 성별 보정 요인]
${r.environmental_factors || '-'}

[종합 감정 소견]
${r.comprehensive_analysis || '-'}
`;

  navigator.clipboard.writeText(reportText).then(() => {
    alert('붕어빵 분석 결과 감정서가 클립보드에 복사되었습니다.');
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
    { title: '얼굴 검출 및 특징 추출 중...', step: '1단계: 얼굴 정렬 후 ArcFace 512차원 얼굴 특징 벡터 추출', progress: 25 },
    { title: '연예인 얼굴 데이터베이스와 비교 중...', step: '2단계: 연예인별 여러 장의 사진으로 만든 얼굴 특징과 유사도 계산', progress: 50 },
    { title: '가장 닮은 연예인 TOP 3 선정 중...', step: '3단계: 유사도 순위 및 닮음 점수 산출', progress: 75 },
    { title: '닮은 부위 설명 작성 중...', step: '4단계: Gemma 4가 사진을 비교해 닮은 포인트 설명', progress: 90 },
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
      extra_images_base64: celebExtraPhotos,
      gender_filter: selectedCelebGender
    };

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

function renderCelebrityResults(data) {
  const resultsSection = document.getElementById('celeb-results-section');
  resultsSection.classList.remove('hidden');
  resultsSection.scrollIntoView({ behavior: 'smooth', block: 'start' });

  const top = data.top_celebrity || {};
  const vibe = data.face_features || {};
  const candidates = data.candidates || [];

  const note = document.getElementById('celeb-photo-note');
  if (note) {
    const used = data.photos_used || 1;
    const rejected = data.photos_rejected || 0;
    note.textContent = rejected > 0
      ? `사진 ${used}장을 종합해 분석했습니다. 다른 사람으로 보이는 사진 ${rejected}장은 제외했습니다.`
      : used > 1 ? `사진 ${used}장을 종합해 분석했습니다.` : '';
    note.classList.toggle('hidden', !note.textContent);
  }

  // 1. Photos
  document.getElementById('celeb-result-user-img').src = photoCelebData;
  const matchImg = document.getElementById('celeb-result-match-img');
  matchImg.src = top.photo_url || '';
  document.getElementById('top-celeb-label').textContent = `${top.name || '연예인'} (${top.category || '배우'})`;

  // 2. Titles & Summaries
  document.getElementById('top-celeb-name').textContent = top.name || '알 수 없음';
  document.getElementById('top-celeb-cat').textContent = `(${top.category || '연예인'})`;
  document.getElementById('top-celeb-summary').textContent = top.summary || '';
  document.getElementById('top-celeb-reason').textContent = top.reason || '';

  // 3. Gauge Score & Counter
  const score = Math.max(0, Math.min(100, top.similarity_percent || 85));
  const circle = document.getElementById('celeb-score-circle');
  const scoreValue = document.getElementById('celeb-score-value');
  const radius = 42;
  const circumference = 2 * Math.PI * radius;
  const offset = circumference - (score / 100) * circumference;
  circle.style.strokeDashoffset = offset;

  let currentVal = 0;
  const counter = setInterval(() => {
    currentVal += 2;
    if (currentVal >= score) {
      currentVal = score;
      clearInterval(counter);
    }
    scoreValue.textContent = `${currentVal}%`;
  }, 20);

  const badge = document.getElementById('celeb-match-badge');
  if (score >= 85) {
    badge.textContent = '🔥 도플갱어 싱크로율 (완벽한 닮은꼴)';
    badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-pink-500/20 text-pink-300 border border-pink-500/40';
  } else if (score >= 75) {
    badge.textContent = '✨ 매우 높은 닮은꼴 (한눈에 알아볼 정도)';
    badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-purple-500/20 text-purple-300 border border-purple-500/40';
  } else {
    badge.textContent = '💫 은근한 매력 닮은꼴 (분위기 싱크로)';
    badge.className = 'px-3 py-1 rounded-full text-xs font-bold bg-indigo-500/20 text-indigo-300 border border-indigo-500/40';
  }

  // 4. Feature Bars
  const det = top.detailed_scores || {
    eyes: Math.min(100, score + 2),
    nose: Math.max(50, score - 3),
    mouth: Math.min(100, score + 1),
    face_shape: Math.max(50, score - 2),
    features: score
  };

  const setBar = (id, val) => {
    const scoreEl = document.getElementById(`celeb-score-${id}`);
    const barEl = document.getElementById(`celeb-bar-${id}`);
    if (scoreEl) scoreEl.textContent = `${val}%`;
    if (barEl) barEl.style.width = `${val}%`;
  };
  setBar('eyes', det.eyes || score);
  setBar('nose', det.nose || score);
  setBar('mouth', det.mouth || score);
  setBar('face-shape', det.face_shape || score);
  setBar('features', det.features || score);

  // 5. Matching Points Tags
  const pointsContainer = document.getElementById('top-celeb-points');
  pointsContainer.innerHTML = '';
  const pts = top.matching_points || ['선한 눈매', '자연스러운 미소', '턱선 비율'];
  pts.forEach(p => {
    const chip = document.createElement('span');
    chip.className = 'px-2.5 py-1 rounded-lg bg-pink-500/15 text-pink-300 border border-pink-500/30 text-xs font-medium';
    chip.textContent = `#${p}`;
    pointsContainer.appendChild(chip);
  });

  // 6. User Vibe / Animal face
  document.getElementById('user-vibe-type').textContent = vibe.face_type || '매력적인 훈남상';
  document.getElementById('user-vibe-shape').textContent = vibe.face_shape || '부드러운 계란형 윤곽';
  document.getElementById('user-vibe-eyes').textContent = vibe.eyes || '깊고 차분한 눈빛';
  document.getElementById('user-vibe-nose-mouth').textContent = vibe.nose_mouth || '오뚝한 콧대와 단정한 입매';
  document.getElementById('user-vibe-overall').textContent = vibe.overall_vibe || '전반적으로 단정하고 신뢰감을 주는 호감형 인상입니다.';

  // 7. Candidates (2nd & 3rd)
  const candContainer = document.getElementById('celeb-candidates-container');
  candContainer.innerHTML = '';
  candidates.forEach(c => {
    const card = document.createElement('div');
    card.className = 'glass-card rounded-2xl p-4 border border-white/10 flex items-center space-x-4 bg-slate-900/60';
    
    const medal = { 2: '🥈 2위', 3: '🥉 3위' }[c.rank] || `${c.rank}위`;
    card.innerHTML = `
      <div class="w-16 h-16 rounded-xl overflow-hidden shrink-0 border border-white/10 bg-slate-800">
        <img src="${escapeHtml(c.photo_url || '')}" alt="${escapeHtml(c.name)}" class="w-full h-full object-cover" onerror="this.style.visibility='hidden'">
      </div>
      <div class="flex-1 min-w-0">
        <div class="flex items-center justify-between mb-1">
          <span class="text-xs font-bold text-amber-300">${medal}: ${escapeHtml(c.name)} <span class="text-[11px] text-slate-400 font-normal">(${escapeHtml(c.category || '연예인')})</span></span>
          <span class="px-2 py-0.5 rounded-full text-[11px] font-extrabold bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">${Number(c.similarity_percent) || 0}%</span>
        </div>
        ${Math.abs((top.similarity_percent || 0) - (Number(c.similarity_percent) || 0)) <= 3
          ? '<span class="inline-block mb-1 px-1.5 py-0.5 rounded text-[10px] font-semibold bg-amber-500/15 text-amber-300 border border-amber-500/30">1위와 비슷한 수준</span>' : ''}
        <p class="text-xs text-slate-300 truncate">${escapeHtml(c.summary || c.reason || '')}</p>
        <p class="text-[11px] text-slate-400 line-clamp-1 mt-0.5">${escapeHtml(c.reason || '')}</p>
      </div>
    `;
    candContainer.appendChild(card);
  });

  lucide.createIcons();
}

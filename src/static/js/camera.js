document.addEventListener('DOMContentLoaded', () => {
    const cameraModal = document.getElementById('cameraModal');
    const cameraContainer = document.getElementById('cameraContainer');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');
    const cancelCameraBtn = document.getElementById('cancelCameraBtn');
    const openCameraBtn = document.getElementById('openCameraBtn');
    const fileInput = document.getElementById('fileInput');
    const recognizeBtn = document.getElementById('recognizeBtn');
    const resultsBlock = document.getElementById('results');

    let stream = null;
    let selectedFile = null; // файл, выбранный из галереи или снятый на камеру

    // ------------------------------------------------------------
    // Открытие камеры
    // ------------------------------------------------------------
    async function openCamera() {
        try {
            stream = await navigator.mediaDevices.getUserMedia({
                video: {
                    facingMode: { ideal: 'environment' },
                    width: { ideal: 1280 },
                    height: { ideal: 720 }
                },
                audio: false
            });

            video.srcObject = stream;
            video.onloadedmetadata = () => adjustContainerToVideo();
            cameraModal.classList.add('active');
        } catch (err) {
            console.error('Ошибка доступа к камере:', err);
            alert('Не удалось получить доступ к камере');
        }
    }

    function adjustContainerToVideo() {
        const vw = video.videoWidth;
        const vh = video.videoHeight;
        if (!vw || !vh) return;

        const ratio = vw / vh;
        const maxW = Math.min(window.innerWidth - 32, 640);
        const maxH = window.innerHeight - 200;

        let w, h;
        if (ratio >= 1) {
            w = maxW;
            h = w / ratio;
            if (h > maxH) { h = maxH; w = h * ratio; }
        } else {
            h = maxH;
            w = h * ratio;
            if (w > maxW) { w = maxW; h = w / ratio; }
        }

        cameraContainer.style.width = w + 'px';
        cameraContainer.style.height = h + 'px';
        cameraContainer.style.aspectRatio = 'auto';
    }

    function closeCamera() {
        if (stream) {
            stream.getTracks().forEach(t => t.stop());
            stream = null;
        }
        video.srcObject = null;
        cameraModal.classList.remove('active');
    }

    // ------------------------------------------------------------
    // Захват кадра
    // ------------------------------------------------------------
    function capturePhoto() {
        if (!video.videoWidth) return;

        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;

        const ctx = canvas.getContext('2d');
        ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

        canvas.toBlob((blob) => {
            if (!blob) return;
            selectedFile = new File([blob], 'capture.jpg', { type: 'image/jpeg' });
            console.log('Фото с камеры готово:', selectedFile.size, 'байт');
            closeCamera();
            // Можно сразу отправить на распознавание:
            sendToSearch(selectedFile);
        }, 'image/jpeg', 0.9);
    }

    // ------------------------------------------------------------
    // Отправка на /search
    // ------------------------------------------------------------
    async function sendToSearch(file) {
        if (!file) {
            alert('Сначала выберите или снимите фото');
            return;
        }

        const formData = new FormData();
        formData.append('photo', file);   // ← имя поля как в openapi: "photo"
        formData.append('top_k', 5);

        // Индикация загрузки
        resultsBlock.innerHTML = '<p>Распознаём...</p>';

        try {
            const response = await fetch('/search', {   // ← путь как в openapi
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errText = await response.text();
                throw new Error(`HTTP ${response.status}: ${errText}`);
            }

            const data = await response.json();
            console.log('Результат распознавания:', data);
            renderResults(data);
        } catch (err) {
            console.error('Ошибка запроса:', err);
            resultsBlock.innerHTML = `<p style="color:red">Ошибка: ${err.message}</p>`;
        }
    }

    // ------------------------------------------------------------
    // Отрисовка результатов
    // ------------------------------------------------------------
    function renderResults(data) {
        const results = data.results || [];

        if (!results.length) {
            resultsBlock.innerHTML = '<p>Ничего не найдено</p>';
            return;
        }

        const html = results.map(r => {
            const meta = r.metadata || {};
            const title = meta.title || meta.name || `ID ${r.id}`;
            const link = r.link ? `<a href="${r.link}" target="_blank">Подробнее</a>` : '';
            return `
                <div class="result-card">
                    <h3>${title}</h3>
                    <p><small>score: ${r.score.toFixed(3)}</small></p>
                    ${link}
                </div>
            `;
        }).join('');

        resultsBlock.innerHTML = `<h2>Найдено: ${results.length}</h2>${html}`;
    }

    // ------------------------------------------------------------
    // Выбор файла из галереи
    // ------------------------------------------------------------
    if (fileInput) {
        fileInput.addEventListener('change', (e) => {
            const file = e.target.files[0];
            if (!file) return;
            selectedFile = file;
            console.log('Файл выбран:', file.name, file.size, 'байт');
        });
    }

    // ------------------------------------------------------------
    // Обработчики событий
    // ------------------------------------------------------------
    if (openCameraBtn) openCameraBtn.addEventListener('click', openCamera);
    if (captureBtn) captureBtn.addEventListener('click', capturePhoto);
    if (cancelCameraBtn) cancelCameraBtn.addEventListener('click', closeCamera);

    if (recognizeBtn) {
        recognizeBtn.addEventListener('click', () => {
            if (selectedFile) {
                sendToSearch(selectedFile);
            } else if (fileInput && fileInput.files[0]) {
                sendToSearch(fileInput.files[0]);
            } else {
                alert('Сначала выберите фото или снимите на камеру');
            }
        });
    }

    window.addEventListener('resize', () => {
        if (cameraModal.classList.contains('active') && video.videoWidth) {
            adjustContainerToVideo();
        }
    });

    document.addEventListener('keydown', (e) => {
        if (e.key === 'Escape' && cameraModal.classList.contains('active')) {
            closeCamera();
        }
    });
});
document.addEventListener('DOMContentLoaded', function() {
    // Получаем элементы
    const recognizeBtn = document.getElementById('recognizeBtn');
    const cameraModal = document.getElementById('cameraModal');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');
    const cancelBtn = document.getElementById('cancelCameraBtn');
    const fileInput = document.getElementById('fileInput');
    const bottleTemplate = document.getElementById('bottleTemplate');
    
    let selectedFile = null;
    let stream = null;

    // ===== ФУНКЦИЯ ОТПРАВКИ НА СЕРВЕР =====
    async function sendToServer() {
        if (!selectedFile) {
            return;
        }

        const formData = new FormData();
        formData.append('photo', selectedFile);
        formData.append('top_k', '5');

        try {
            recognizeBtn.textContent = '⏳ Распознаем...';
            recognizeBtn.disabled = true;

            const response = await fetch('/search', {
                method: 'POST',
                body: formData
            });

            // Проверяем статус ответа
            if (!response.ok) {
                console.error('Ошибка сервера:', response.status);
                recognizeBtn.textContent = 'Распознать вино';
                recognizeBtn.disabled = false;
                recognizeBtn.style.background = '';
                return;
            }

            const result = await response.json();
            console.log('Результат распознавания:', result);

            // ===== ПРОВЕРЯЕМ РАЗНЫЕ ФОРМАТЫ ОТВЕТА =====
            let wineLink = null;
            let wineName = null;

            // Вариант 1: result.results[0].link
            if (result.results && result.results.length > 0) {
                const firstResult = result.results[0];
                wineLink = firstResult.link || firstResult.content?.link || firstResult.metadata?.link;
                wineName = firstResult.content?.filename || firstResult.metadata?.filename || 'вино';
            }
            // Вариант 2: result.link (если ответ без results)
            else if (result.link) {
                wineLink = result.link;
                wineName = result.content?.filename || result.metadata?.filename || 'вино';
            }
            // Вариант 3: result.content?.link
            else if (result.content?.link) {
                wineLink = result.content.link;
                wineName = result.content.filename || 'вино';
            }
            
            if (wineLink) {
                console.log('🔗 Перенаправление на:', wineLink);
                console.log('🍷 Найдено вино:', wineName);
                
                // ПЕРЕНАПРАВЛЯЕМ НА СТРАНИЦУ ВИНА
                window.location.href = wineLink;
            } else {
                console.warn('Вино не найдено или нет ссылки');
                recognizeBtn.textContent = 'Распознать вино';
                recognizeBtn.disabled = false;
                recognizeBtn.style.background = '';
            }

        } catch (error) {
            console.error('Ошибка:', error);
            recognizeBtn.textContent = 'Распознать вино';
            recognizeBtn.disabled = false;
            recognizeBtn.style.background = '';
        }
    }

    // ===== ФУНКЦИЯ ОТКРЫТИЯ КАМЕРЫ =====
    async function openCamera() {
        try {
            cameraModal.classList.add('active');
            
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1920 },
                    height: { ideal: 1080 }
                }
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            await video.play();
            
            // После запуска видео адаптируем трафарет на всю ширину и высоту
            setTimeout(() => {
                adaptBottleSize();
            }, 100);
            
            console.log('✅ Камера запущена');
        } catch (error) {
            console.error('Ошибка камеры:', error);
            closeCamera();
        }
    }

    // ===== ФУНКЦИЯ ЗАКРЫТИЯ КАМЕРЫ =====
    function closeCamera() {
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        video.srcObject = null;
        cameraModal.classList.remove('active');
    }

    // ===== ФУНКЦИЯ АДАПТАЦИИ ТРАФАРЕТА =====
    function adaptBottleSize() {
        // Трафарет на всю ширину и высоту контейнера
        bottleTemplate.style.width = '100%';
        bottleTemplate.style.height = '100%';
        bottleTemplate.style.objectFit = 'contain';
        bottleTemplate.style.objectPosition = 'center';
        
        console.log('✅ Трафарет адаптирован на всю ширину и высоту');
    }

    // ===== ФУНКЦИЯ ПОКАЗА ПРЕВЬЮ =====
    function showPreview(file) {
        // Удаляем старое превью
        const oldPreview = document.querySelector('.preview-container');
        if (oldPreview) {
            oldPreview.remove();
        }
        
        const reader = new FileReader();
        reader.onload = function(e) {
            const container = document.createElement('div');
            container.className = 'preview-container';
            container.style.cssText = 'margin: 15px 0; text-align: center;';
            
            const img = document.createElement('img');
            img.src = e.target.result;
            img.style.cssText = 'max-width: 100%; max-height: 250px; border-radius: 12px; box-shadow: 0 4px 20px rgba(0,0,0,0.1);';
            
            const fileName = document.createElement('p');
            fileName.style.cssText = 'font-size: 13px; color: #666; margin-top: 8px;';
            fileName.textContent = `📎 ${file.name}`;
            
            container.appendChild(img);
            container.appendChild(fileName);
            
            const uploadArea = document.querySelector('.upload-area');
            uploadArea.parentNode.insertBefore(container, uploadArea.nextSibling);
        };
        reader.readAsDataURL(file);
    }

    // ===== ОБРАБОТЧИКИ СОБЫТИЙ =====

    // 1. ГЛАВНАЯ КНОПКА "РАСПОЗНАТЬ"
    recognizeBtn.addEventListener('click', function(e) {
        e.preventDefault();
        e.stopPropagation();
        
        console.log('🔘 Кнопка нажата');
        console.log('📁 selectedFile:', selectedFile ? selectedFile.name : 'null');
        
        // ЕСЛИ ЕСТЬ ФАЙЛ - ОТПРАВЛЯЕМ НА СЕРВЕР
        if (selectedFile) {
            console.log('📤 Отправляем на сервер...');
            sendToServer();
            return;
        }
        
        // ЕСЛИ НЕТ ФАЙЛА - ОТКРЫВАЕМ КАМЕРУ
        console.log('📷 Открываем камеру...');
        openCamera();
    });

    // 2. КНОПКА "СФОТОГРАФИРОВАТЬ"
    captureBtn.addEventListener('click', function() {
        if (!video.videoWidth || !video.videoHeight) {
            return;
        }
        
        console.log('📸 Делаем фото...');
        
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        canvas.toBlob(function(blob) {
            selectedFile = new File([blob], 'photo_from_camera.jpg', { type: 'image/jpeg' });
            console.log('✅ Фото сделано:', selectedFile.name);
            
            closeCamera();
            showPreview(selectedFile);
            
            recognizeBtn.textContent = '📸 Распознать (фото готово)';
            recognizeBtn.style.background = '#2d7d2d';
            
        }, 'image/jpeg', 0.95);
    });

    // 3. КНОПКА "ОТМЕНА"
    cancelBtn.addEventListener('click', closeCamera);

    // 4. ЗАКРЫТИЕ ПО КЛИКУ НА ФОН
    cameraModal.addEventListener('click', function(e) {
        if (e.target === cameraModal) {
            closeCamera();
        }
    });

    // 5. ВЫБОР ФАЙЛА ИЗ ГАЛЕРЕИ
    fileInput.addEventListener('change', function(e) {
        if (this.files.length > 0) {
            selectedFile = this.files[0];
            console.log('✅ Файл выбран:', selectedFile.name);
            showPreview(selectedFile);
            recognizeBtn.textContent = '📸 Распознать (фото готово)';
            recognizeBtn.style.background = '#2d7d2d';
        }
        this.value = '';
    });

    // 6. АДАПТАЦИЯ ТРАФАРЕТА ПРИ ИЗМЕНЕНИИ РАЗМЕРА
    window.addEventListener('resize', function() {
        if (cameraModal.classList.contains('active')) {
            adaptBottleSize();
        }
    });

    // 7. АДАПТАЦИЯ ПРИ ЗАГРУЗКЕ ВИДЕО
    video.addEventListener('loadedmetadata', function() {
        adaptBottleSize();
    });

    console.log('✅ Скрипт загружен');
    console.log('💡 Если есть фото - кнопка отправит на сервер');
    console.log('💡 Если нет фото - кнопка откроет камеру');
});

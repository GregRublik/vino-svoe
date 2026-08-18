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
    let resizeObserver = null;

    // Функция адаптивного изменения размера трафарета
    function adaptBottleSize() {
        if (!video.videoWidth || !video.videoHeight) return;
        
        // Получаем размеры видео
        const videoWidth = video.videoWidth;
        const videoHeight = video.videoHeight;
        const aspectRatio = videoWidth / videoHeight;
        
        // Получаем размеры контейнера
        const container = video.parentElement;
        const containerWidth = container.clientWidth;
        const containerHeight = container.clientHeight;
        
        // Определяем базовый размер в зависимости от пропорций видео
        let baseSize = 0.75; // 75% от размера контейнера по умолчанию
        
        // Адаптируем под разные пропорции
        if (aspectRatio > 1.5) {
            // Широкоформатное видео (горизонтальное)
            baseSize = 0.65;
        } else if (aspectRatio < 0.8) {
            // Узкое видео (вертикальное)
            baseSize = 0.85;
        } else {
            // Квадратное или близкое к квадратному
            baseSize = 0.75;
        }
        
        // Учитываем размер экрана
        const screenWidth = window.innerWidth;
        if (screenWidth < 400) {
            baseSize = Math.min(baseSize * 1.1, 0.9); // На маленьких экранах делаем чуть больше
        } else if (screenWidth > 768) {
            baseSize = Math.min(baseSize * 0.9, 0.8); // На больших экранах делаем чуть меньше
        }
        
        // Применяем размеры
        const sizePercent = baseSize * 100;
        bottleTemplate.style.width = sizePercent + '%';
        bottleTemplate.style.height = (sizePercent * 0.9) + '%'; // Сохраняем пропорции
        
        console.log('Адаптация трафарета:', {
            videoSize: `${videoWidth}x${videoHeight}`,
            aspectRatio: aspectRatio,
            containerSize: `${containerWidth}x${containerHeight}`,
            sizePercent: sizePercent + '%'
        });
    }

    // Функция запуска камеры
    async function startCamera() {
        try {
            // Запрашиваем разрешение камеры с оптимальными настройками
            const constraints = {
                video: {
                    facingMode: 'environment',
                    width: { ideal: 1920 },
                    height: { ideal: 1080 },
                    aspectRatio: { ideal: 4/3 }
                }
            };
            
            stream = await navigator.mediaDevices.getUserMedia(constraints);
            video.srcObject = stream;
            await video.play();
            
            // Ждем загрузки видео и адаптируем трафарет
            video.addEventListener('loadedmetadata', function() {
                adaptBottleSize();
            });
            
            // Адаптируем при изменении размера окна
            if (resizeObserver) {
                resizeObserver.disconnect();
            }
            resizeObserver = new ResizeObserver(() => {
                adaptBottleSize();
            });
            resizeObserver.observe(video.parentElement);
            
            // Адаптируем при изменении размера окна
            window.addEventListener('resize', adaptBottleSize);
            
            console.log('Камера запущена');
        } catch (error) {
            console.error('Ошибка доступа к камере:', error);
            alert('Не удалось получить доступ к камере. Пожалуйста, разрешите доступ или выберите фото из галереи.');
            cameraModal.classList.remove('active');
        }
    }

    // 1. ОТКРЫТИЕ КАМЕРЫ ПРИ НАЖАТИИ КНОПКИ "РАСПОЗНАТЬ ВИНО"
    recognizeBtn.addEventListener('click', async function() {
        // Показываем модальное окно
        cameraModal.classList.add('active');
        await startCamera();
    });

    // 2. СДЕЛАТЬ ФОТО
    captureBtn.addEventListener('click', function() {
        if (!video.videoWidth || !video.videoHeight) {
            alert('Камера еще не готова. Подождите немного.');
            return;
        }
        
        // Устанавливаем размеры canvas
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        
        // Рисуем текущий кадр
        const context = canvas.getContext('2d');
        context.drawImage(video, 0, 0, canvas.width, canvas.height);
        
        // Конвертируем в файл
        canvas.toBlob(function(blob) {
            selectedFile = new File([blob], 'photo_from_camera.jpg', { type: 'image/jpeg' });
            console.log('Фото сделано:', selectedFile.name);
            
            // Закрываем камеру
            closeCamera();
            
            // Показываем превью
            showPreview(selectedFile);
            
            // Меняем текст кнопки
            recognizeBtn.textContent = '📸 Распознать (фото готово)';
            recognizeBtn.style.background = '#2d7d2d';
            
        }, 'image/jpeg', 0.95);
    });

    // 3. ЗАКРЫТИЕ КАМЕРЫ
    function closeCamera() {
        // Останавливаем все треки
        if (stream) {
            stream.getTracks().forEach(track => track.stop());
            stream = null;
        }
        video.srcObject = null;
        video.removeEventListener('loadedmetadata', adaptBottleSize);
        window.removeEventListener('resize', adaptBottleSize);
        if (resizeObserver) {
            resizeObserver.disconnect();
            resizeObserver = null;
        }
        cameraModal.classList.remove('active');
    }

    cancelBtn.addEventListener('click', closeCamera);

    // Закрытие при клике на фон
    cameraModal.addEventListener('click', function(e) {
        if (e.target === cameraModal) {
            closeCamera();
        }
    });

    // 4. ЗАГРУЗКА ФАЙЛА ИЗ ГАЛЕРЕИ
    fileInput.addEventListener('change', function(e) {
        if (this.files.length > 0) {
            selectedFile = this.files[0];
            console.log('Файл выбран:', selectedFile.name);
            showPreview(selectedFile);
            recognizeBtn.textContent = '📸 Распознать (фото готово)';
            recognizeBtn.style.background = '#2d7d2d';
        }
    });

    // 5. ПОКАЗ ПРЕВЬЮ
    function showPreview(file) {
        const reader = new FileReader();
        reader.onload = function(e) {
            // Удаляем старое превью
            const oldPreview = document.querySelector('.preview-container');
            if (oldPreview) {
                oldPreview.remove();
            }
            
            // Создаем контейнер для превью
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
            
            // Вставляем после upload-area
            const uploadArea = document.querySelector('.upload-area');
            uploadArea.parentNode.insertBefore(container, uploadArea.nextSibling);
        };
        reader.readAsDataURL(file);
    }

    // 6. КНОПКА "РАСПОЗНАТЬ ВИНО" (отправка)
    recognizeBtn.addEventListener('click', async function() {
        // Если камера открыта, не отправляем
        if (cameraModal.classList.contains('active')) {
            return;
        }

        if (!selectedFile) {
            alert('Пожалуйста, сделайте фото через камеру или выберите файл');
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

            const result = await response.json();
            console.log('Результат:', result);

        } catch (error) {
            console.error('Ошибка:', error);
            alert('Ошибка при распознавании');
        } finally {
            recognizeBtn.textContent = 'Распознать вино';
            recognizeBtn.disabled = false;
            recognizeBtn.style.background = ''; // Сброс цвета
        }
    });
});
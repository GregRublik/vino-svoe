document.addEventListener('DOMContentLoaded', function() {
    // Получаем элементы
    const recognizeBtn = document.getElementById('recognizeBtn');
    const cameraModal = document.getElementById('cameraModal');
    const video = document.getElementById('video');
    const canvas = document.getElementById('canvas');
    const captureBtn = document.getElementById('captureBtn');
    const cancelBtn = document.getElementById('cancelCameraBtn');
    const galleryBtn = document.getElementById('galleryBtn');
    const fileInput = document.getElementById('fileInput');
    const bottleTemplate = document.getElementById('bottleTemplate');
    const resultSection = document.getElementById('resultSection');
    const resultContent = document.getElementById('resultContent');
    
    let selectedFile = null;
    let stream = null;
    const defaultErrorMessage = 'Сервис временно недоступен. Попробуйте ещё раз.';

    function addTextElement(parent, tagName, className, text) {
        const element = document.createElement(tagName);
        element.className = className;
        element.textContent = text;
        parent.appendChild(element);
        return element;
    }

    function createWineCard(result, compact = false) {
        const card = result.card || {};
        const article = document.createElement('article');
        article.className = compact ? 'wine-card wine-card-compact' : 'wine-card';

        const fallbackName = card.slug || result.content?.filename || 'Вино';
        addTextElement(article, 'h3', 'wine-card-title', card.name || fallbackName);

        const meta = document.createElement('div');
        meta.className = 'wine-card-meta';
        const fields = [
            ['Винодельня', card.winery],
            ['Регион', card.region],
            ['Сорт винограда', card.grape_variety],
            ['Категория', card.category],
            ['Цвет', card.color],
        ];

        if (card.slug) {
            fields.push([
                'Рейтинг Роскачества',
                card.roskachestvo_rating || 'Нет данных',
            ]);
        }

        fields.forEach(([label, value]) => {
            if (value === null || value === undefined || String(value).trim() === '') {
                return;
            }
            const row = document.createElement('div');
            row.className = 'wine-card-meta-row';
            addTextElement(row, 'span', 'wine-card-meta-label', label);
            addTextElement(row, 'span', 'wine-card-meta-value', String(value));
            meta.appendChild(row);
        });
        if (meta.childElementCount > 0) {
            article.appendChild(meta);
        }

        if (card.description) {
            const description = document.createElement('div');
            description.className = 'wine-card-block';
            addTextElement(description, 'div', 'wine-card-block-title', 'Описание');
            addTextElement(description, 'div', 'wine-card-block-text', card.description);
            article.appendChild(description);
        }

        const servingRecommendation = typeof card.serving_recommendation === 'string'
            ? card.serving_recommendation.trim()
            : '';
        if (card.slug && servingRecommendation) {
            const serving = document.createElement('div');
            serving.className = 'wine-card-block';
            addTextElement(serving, 'div', 'wine-card-block-title', 'Рекомендации к подаче');
            addTextElement(
                serving,
                'div',
                'wine-card-block-text',
                servingRecommendation
            );
            article.appendChild(serving);
        }

        const link = card.link || result.link;
        if (link) {
            const linkElement = document.createElement('a');
            linkElement.className = 'wine-card-link';
            linkElement.href = link;
            linkElement.target = '_blank';
            linkElement.rel = 'noopener noreferrer';
            linkElement.textContent = 'Открыть карточку на сайте «Своё вино»';
            article.appendChild(linkElement);
        }

        if (!compact && card.slug) {
            const pairingButton = document.createElement('button');
            pairingButton.type = 'button';
            pairingButton.className = 'pairing-button';
            pairingButton.textContent = '🍽 Подобрать блюдо';
            pairingButton.addEventListener('click', () => {
                loadPairing(card.slug, article, pairingButton);
            });
            article.appendChild(pairingButton);
        }

        if (!card.description) {
            addTextElement(
                article,
                'div',
                'result-description',
                'Подробное описание для этой позиции пока недоступно.'
            );
        }

        return article;
    }

    async function loadPairing(slug, article, button) {
        const previousResult = article.querySelector('.pairing-result');
        if (previousResult) {
            previousResult.remove();
        }

        button.disabled = true;
        button.textContent = '⏳ Подбираем сочетание...';
        try {
            const response = await fetch(`/pairing/${encodeURIComponent(slug)}`);
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) {
                throw new Error(safeErrorMessage(payload.detail));
            }

            const pairingBlock = document.createElement('div');
            pairingBlock.className = 'wine-card-block pairing-result';
            addTextElement(pairingBlock, 'div', 'wine-card-block-title', 'Сочетание');
            addTextElement(
                pairingBlock,
                'div',
                'wine-card-block-text',
                payload.recommendation || 'Подходящее блюдо пока не определено.'
            );
            if (payload.rationale) {
                addTextElement(pairingBlock, 'div', 'pairing-rationale', payload.rationale);
            }
            article.appendChild(pairingBlock);
        } catch (error) {
            const message = error instanceof Error ? error.message : defaultErrorMessage;
            addTextElement(
                article,
                'div',
                'result-message result-message-error pairing-result',
                message
            );
        } finally {
            button.disabled = false;
            button.textContent = '🍽 Подобрать блюдо';
        }
    }

    function showResultMessage(title, message, isError = false) {
        resultContent.replaceChildren();
        resultSection.hidden = false;
        addTextElement(resultContent, 'h2', 'result-heading', title);
        addTextElement(
            resultContent,
            'div',
            isError ? 'result-message result-message-error' : 'result-message',
            message
        );
    }

    function safeErrorMessage(detail) {
        if (typeof detail !== 'string' || detail.trim() === '') {
            return defaultErrorMessage;
        }

        // Не показываем пользователю пути и служебные данные backend.
        if (/[\\/]|[A-Za-z]:/.test(detail)) {
            return defaultErrorMessage;
        }
        return detail;
    }

    function renderSearchResult(searchResult) {
        resultContent.replaceChildren();
        resultSection.hidden = false;
        const results = Array.isArray(searchResult.results) ? searchResult.results : [];

        if (searchResult.found && results.length > 0) {
            addTextElement(resultContent, 'h2', 'result-heading', 'Вино найдено');
            resultContent.appendChild(createWineCard(results[0]));
        } else {
            addTextElement(resultContent, 'h2', 'result-heading', 'Точное совпадение не найдено');
            addTextElement(
                resultContent,
                'p',
                'result-description',
                'Мы не можем точно определить вино по этому фото. Ниже — похожие позиции из каталога.'
            );

            const alternatives = results.filter(result => result.card);
            if (alternatives.length > 0) {
                addTextElement(resultContent, 'h3', 'wine-card-block-title', 'Похожие вина');
                const grid = document.createElement('div');
                grid.className = 'alternatives-grid';
                alternatives.forEach(result => {
                    grid.appendChild(createWineCard(result, true));
                });
                resultContent.appendChild(grid);
            } else {
                showResultMessage(
                    'Вино не найдено',
                    'В каталоге нет подходящей карточки. Попробуйте сделать фото ближе и при хорошем освещении.'
                );
            }
        }

        resultSection.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }

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

            const response = await fetch('/search/details', {
                method: 'POST',
                body: formData
            });

            if (!response.ok) {
                const errorBody = await response.json().catch(() => ({}));
                showResultMessage(
                    'Не удалось распознать фото',
                    safeErrorMessage(errorBody.detail),
                    true
                );
                recognizeBtn.textContent = 'Распознать вино';
                recognizeBtn.disabled = false;
                recognizeBtn.style.background = '';
                return;
            }

            const result = await response.json();
            renderSearchResult(result);
            recognizeBtn.textContent = 'Распознать ещё раз';
            recognizeBtn.disabled = false;
            recognizeBtn.style.background = '';

        } catch (error) {
            console.error('Ошибка:', error);
            showResultMessage(
                'Ошибка соединения',
                'Не удалось связаться с сервисом. Проверьте подключение и повторите попытку.',
                true
            );
            recognizeBtn.textContent = 'Распознать вино';
            recognizeBtn.disabled = false;
            recognizeBtn.style.background = '';
        }
    }

    // ===== ФУНКЦИЯ ОТКРЫТИЯ КАМЕРЫ =====
    async function openCamera() {
        if (!window.isSecureContext || !navigator.mediaDevices?.getUserMedia) {
            showResultMessage(
                'Камера недоступна',
                'Для камеры нужен HTTPS. Выберите готовое фото из устройства.',
                true
            );
            return;
        }

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
            showResultMessage(
                'Камера недоступна',
                'Разрешите доступ к камере или выберите готовое фото из устройства.',
                true
            );
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

    function handleSelectedFile(file) {
        if (!file) {
            return;
        }

        selectedFile = file;
        console.log('✅ Файл выбран:', selectedFile.name);
        showPreview(selectedFile);
        recognizeBtn.textContent = '📸 Распознать (фото готово)';
        recognizeBtn.style.background = '#2d7d2d';
    }

    async function chooseFile() {
        // В Chromium это не позволяет восстановить удалённую папку проекта из
        // состояния системного диалога. Значение startIn — имя стандартной
        // папки ОС, а не путь приложения. В остальных браузерах используется
        // обычный input type=file.
        if (window.isSecureContext && typeof window.showOpenFilePicker === 'function') {
            try {
                const [fileHandle] = await window.showOpenFilePicker({
                    multiple: false,
                    startIn: 'downloads',
                    types: [
                        {
                            description: 'Изображения',
                            accept: {
                                'image/*': ['.jpg', '.jpeg', '.png', '.webp'],
                            },
                        },
                    ],
                });
                handleSelectedFile(await fileHandle.getFile());
                return;
            } catch (error) {
                // Отмена пользователем — штатный сценарий. При проблеме с
                // поддержкой стартовой директории переходим к fallback input.
                if (error?.name === 'AbortError') {
                    return;
                }
                console.warn('Не удалось открыть современный file picker:', error);
            }
        }

        fileInput.value = '';
        fileInput.click();
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
    galleryBtn.addEventListener('click', chooseFile);
    fileInput.addEventListener('change', function(e) {
        handleSelectedFile(this.files[0]);
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

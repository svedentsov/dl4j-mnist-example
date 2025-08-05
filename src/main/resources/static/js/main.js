document.addEventListener('DOMContentLoaded', () => {
    // === ОБЩИЕ ЭЛЕМЕНТЫ ===
    const spinner = document.getElementById('spinner');
    const resultContainer = document.getElementById('result-container');
    const predictionOutput = document.getElementById('prediction-output');
    const errorContainer = document.getElementById('error-container');
    const errorMessage = document.getElementById('error-message');
    const feedbackElements = [spinner, resultContainer, errorContainer];

    // === ЛОГИКА ЗАГРУЗКИ ФАЙЛА (ЛЕВАЯ КОЛОНКА) ===
    const form = document.getElementById('prediction-form');
    const fileInput = document.getElementById('image-file-input');
    const fileNameDisplay = document.getElementById('file-name-display');
    const submitButton = document.getElementById('submit-button');

    fileInput.addEventListener('change', () => {
        if (fileInput.files.length > 0) {
            fileNameDisplay.textContent = fileInput.files[0].name;
            submitButton.disabled = false;
        } else {
            fileNameDisplay.textContent = 'Выберите файл...';
            submitButton.disabled = true;
        }
    });

    form.addEventListener('submit', (event) => {
        event.preventDefault();
        const file = fileInput.files[0];
        if (!file) return;

        const formData = new FormData();
        formData.append('image', file);
        sendPrediction(formData);
    });

    // === ЛОГИКА РИСОВАНИЯ (ПРАВАЯ КОЛОНКА) ===
    const canvas = document.getElementById('drawing-canvas');
    const ctx = canvas.getContext('2d');
    const clearButton = document.getElementById('clear-button');

    let isDrawing = false;
    let hasDrawn = false;
    ctx.strokeStyle = 'white';
    ctx.lineWidth = 20;
    ctx.lineCap = 'round';
    ctx.lineJoin = 'round';

    function clearCanvas() {
        ctx.fillStyle = 'black';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        hasDrawn = false;
        hideFeedback();
    }

    function getMousePos(event) {
        const rect = canvas.getBoundingClientRect();
        const scaleX = canvas.width / rect.width;
        const scaleY = canvas.height / rect.height;
        let clientX = event.touches ? event.touches[0].clientX : event.clientX;
        let clientY = event.touches ? event.touches[0].clientY : event.clientY;
        return {
            x: (clientX - rect.left) * scaleX,
            y: (clientY - rect.top) * scaleY
        };
    }

    function startDrawing(event) {
        hideFeedback();
        isDrawing = true;
        hasDrawn = true;
        const { x, y } = getMousePos(event);
        ctx.beginPath();
        ctx.moveTo(x, y);
    }

    function draw(event) {
        if (!isDrawing) return;
        event.preventDefault();
        const { x, y } = getMousePos(event);
        ctx.lineTo(x, y);
        ctx.stroke();
    }

    function stopDrawing() {
        if (!isDrawing) return;
        isDrawing = false;
        ctx.closePath();
        if (hasDrawn) {
            canvas.toBlob((blob) => {
                const formData = new FormData();
                formData.append('image', blob, 'digit.png');
                sendPrediction(formData);
            }, 'image/png');
        }
    }

    canvas.addEventListener('mousedown', startDrawing);
    canvas.addEventListener('mousemove', draw);
    canvas.addEventListener('mouseup', stopDrawing);
    canvas.addEventListener('mouseout', stopDrawing);
    canvas.addEventListener('touchstart', startDrawing);
    canvas.addEventListener('touchmove', draw);
    canvas.addEventListener('touchend', stopDrawing);
    clearButton.addEventListener('click', clearCanvas);


    // === ОБЩАЯ ФУНКЦИЯ ОТПРАВКИ И ОБРАБОТКИ РЕЗУЛЬТАТА ===
    async function sendPrediction(formData) {
        showFeedback(spinner);
        submitButton.disabled = true;
        clearButton.disabled = true;

        try {
            const response = await fetch('/api/v1/predict', {
                method: 'POST',
                body: formData,
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || `Ошибка сервера: ${response.status}`);
            }

            const data = await response.json();
            predictionOutput.textContent = data.predictedDigit;
            showFeedback(resultContainer);

        } catch (error) {
            console.error('Ошибка при отправке запроса:', error);
            errorMessage.textContent = error.message || 'Не удалось связаться с сервером.';
            showFeedback(errorContainer);
        } finally {
            if (fileInput.files.length > 0) {
                 submitButton.disabled = false;
            }
            clearButton.disabled = false;
        }
    }

    // === УПРАВЛЕНИЕ ВИДИМОСТЬЮ ОБРАТНОЙ СВЯЗИ ===
    function hideFeedback() {
        feedbackElements.forEach(el => el.classList.remove('visible'));
    }

    function showFeedback(elementToShow) {
        hideFeedback();
        elementToShow.classList.add('visible');
    }

    // === ИНИЦИАЛИЗАЦИЯ ===
    clearCanvas();
});
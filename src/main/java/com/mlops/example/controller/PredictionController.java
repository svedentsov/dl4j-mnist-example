package com.mlops.example.controller;

import com.mlops.example.config.GlobalExceptionHandler;
import com.mlops.example.controller.dto.PredictionResponse;
import com.mlops.example.exception.InvalidInputDataException;
import com.mlops.example.exception.ModelProcessingException;
import com.mlops.example.service.ModelServingService;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import io.micrometer.core.instrument.Timer;
import jakarta.annotation.PostConstruct;
import lombok.RequiredArgsConstructor;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.PostMapping;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RequestParam;
import org.springframework.web.bind.annotation.RestController;
import org.springframework.web.multipart.MultipartFile;

import java.util.Arrays;
import java.util.List;
import java.util.stream.Collectors;

/**
 * REST-контроллер, отвечающий за обработку HTTP-запросов, связанных с предсказанием.
 * <p>
 * Предоставляет два основных эндпоинта:
 * <ul>
 *     <li>{@code /api/v1/predict}: для предсказания по одному изображению.</li>
 *     <li>{@code /api/v1/predict-batch}: для пакетной обработки нескольких изображений.</li>
 * </ul>
 * Интегрирован с Micrometer для сбора и экспорта метрик производительности (RPS, latency)
 * и бизнес-метрик (распределение предсказанных классов) в Prometheus.
 */
@Slf4j
@RestController
@RequestMapping("/api/v1")
@RequiredArgsConstructor
public class PredictionController {

    private final ModelServingService modelService;
    private final MeterRegistry meterRegistry;

    /**
     * Счетчик общего количества запросов на предсказание. Инкрементируется для каждого полученного изображения.
     * Метрика в Prometheus: {@code predictions_requests_total_total}.
     */
    private Counter totalPredictionsCounter;

    /**
     * Таймер для измерения задержки (latency) выполнения предсказаний.
     * Публикует перцентили (P50, P95, P99) для анализа производительности.
     * Метрика в Prometheus: {@code predictions_latency_seconds}.
     */
    private Timer predictionLatencyTimer;

    /**
     * Инициализирует все метрики Micrometer после создания бина контроллера.
     * Использует аннотацию {@link PostConstruct} для гарантии выполнения после внедрения зависимостей.
     */
    @PostConstruct
    public void initMetrics() {
        totalPredictionsCounter = Counter.builder("predictions.requests.total")
                .description("Общее количество запросов на предсказание (успешных и неуспешных).")
                .register(meterRegistry);
        predictionLatencyTimer = Timer.builder("predictions.latency.seconds")
                .description("Время выполнения запроса на предсказание.")
                .publishPercentiles(0.5, 0.95, 0.99)
                .register(meterRegistry);
    }

    /**
     * Обрабатывает POST-запрос для предсказания цифры по одному изображению.
     *
     * @param imageFile Файл изображения, переданный в multipart/form-data с именем 'image'.
     * @return {@link ResponseEntity} с объектом {@link PredictionResponse} в теле, содержащим предсказанную цифру,
     * и HTTP-статусом 200 (OK). В случае ошибки, обработка передается в {@link GlobalExceptionHandler}.
     * @throws InvalidInputDataException если файл не предоставлен, пуст или имеет неверный тип.
     * @throws ModelProcessingException  если происходит ошибка во время обработки изображения моделью.
     */
    @PostMapping("/predict")
    public ResponseEntity<PredictionResponse> predict(@RequestParam("image") MultipartFile imageFile) {
        totalPredictionsCounter.increment();
        Timer.Sample sample = Timer.start(meterRegistry);
        try {
            validateImageFile(imageFile);
            log.info("Получен файл '{}' ({} байт) для одиночного предсказания.", imageFile.getOriginalFilename(), imageFile.getSize());
            int prediction = modelService.predict(imageFile);
            log.info("Результат предсказания для файла '{}': {}", imageFile.getOriginalFilename(), prediction);
            // Инкрементируем счетчик для конкретного предсказанного класса (цифры)
            meterRegistry.counter("predictions.class.distribution.total", "digit", String.valueOf(prediction)).increment();
            return ResponseEntity.ok(new PredictionResponse(prediction));
        } finally {
            // Завершаем измерение времени и записываем результат в таймер
            sample.stop(predictionLatencyTimer);
        }
    }

    /**
     * Обрабатывает POST-запрос для пакетного предсказания по нескольким изображениям.
     *
     * @param imageFiles Массив файлов изображений, переданный в multipart/form-data с именем 'images'.
     * @return {@link ResponseEntity} со списком объектов {@link PredictionResponse} и статусом 200 (OK).
     * @throws InvalidInputDataException если массив файлов пуст, превышает лимит или содержит невалидные файлы.
     */
    @PostMapping("/predict-batch")
    public ResponseEntity<List<PredictionResponse>> predictBatch(@RequestParam("images") MultipartFile[] imageFiles) {
        validateImageBatch(imageFiles);
        totalPredictionsCounter.increment(imageFiles.length); // Инкрементируем на размер батча
        Timer.Sample sample = Timer.start(meterRegistry);

        try {
            log.info("Получен батч из {} изображений для предсказания.", imageFiles.length);
            List<PredictionResponse> predictions = Arrays.stream(imageFiles)
                    .parallel() // Используем параллельный стрим для ускорения обработки батча
                    .map(file -> {
                        validateImageFile(file);
                        int prediction = modelService.predict(file);
                        log.debug("Файл '{}' из батча -> Предсказание: {}", file.getOriginalFilename(), prediction);
                        meterRegistry.counter("predictions.class.distribution.total", "digit", String.valueOf(prediction)).increment();
                        return new PredictionResponse(prediction);
                    })
                    .collect(Collectors.toList());

            return ResponseEntity.ok(predictions);
        } finally {
            sample.stop(predictionLatencyTimer);
        }
    }

    /**
     * Проверяет один файл изображения на корректность.
     *
     * @param imageFile Файл для проверки.
     * @throws InvalidInputDataException если файл null, пустой или его Content-Type не является изображением.
     */
    private void validateImageFile(MultipartFile imageFile) {
        if (imageFile == null || imageFile.isEmpty()) {
            log.warn("Попытка предсказания с пустым или отсутствующим файлом.");
            throw new InvalidInputDataException("Файл изображения не предоставлен или пуст.");
        }
        String contentType = imageFile.getContentType();
        if (contentType == null || !contentType.startsWith("image/")) {
            log.warn("Попытка загрузки файла с неверным Content-Type: {}", contentType);
            throw new InvalidInputDataException("Недопустимый тип файла. Требуется изображение.");
        }
    }

    /**
     * Проверяет массив файлов для пакетной обработки.
     *
     * @param imageFiles Массив файлов.
     * @throws InvalidInputDataException если массив null, пустой или его размер превышает лимит.
     */
    private void validateImageBatch(MultipartFile[] imageFiles) {
        if (imageFiles == null || imageFiles.length == 0) {
            throw new InvalidInputDataException("Не предоставлено ни одного файла для пакетной обработки.");
        }
        final int maxBatchSize = 50;
        if (imageFiles.length > maxBatchSize) {
            log.warn("Превышен максимальный размер батча. Получено: {}, лимит: {}", imageFiles.length, maxBatchSize);
            throw new InvalidInputDataException("Максимальный размер батча - " + maxBatchSize + " изображений.");
        }
    }
}

package com.mlops.example.config;

import com.mlops.example.controller.dto.ApiErrorResponse;
import com.mlops.example.exception.InvalidInputDataException;
import com.mlops.example.exception.ModelProcessingException;
import io.micrometer.core.instrument.Counter;
import io.micrometer.core.instrument.MeterRegistry;
import lombok.extern.slf4j.Slf4j;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.ControllerAdvice;
import org.springframework.web.bind.annotation.ExceptionHandler;
import org.springframework.web.context.request.WebRequest;
import org.springframework.web.multipart.MaxUploadSizeExceededException;
import org.springframework.web.multipart.support.MissingServletRequestPartException;
import org.springframework.web.servlet.resource.NoResourceFoundException;

import java.time.LocalDateTime;
import java.time.ZoneOffset;

/**
 * Глобальный обработчик исключений для всего приложения.
 * <p>
 * Этот класс, аннотированный {@link ControllerAdvice}, перехватывает исключения,
 * возникающие в контроллерах, и преобразует их в стандартизированные
 * HTTP-ответы с телом {@link ApiErrorResponse}. Это позволяет централизовать
 * логику обработки ошибок, избежать дублирования кода и обеспечить
 * консистентность API.
 * <p>
 * Также интегрирован с Micrometer для подсчета ошибок, связанных с работой ML-модели.
 */
@Slf4j
@ControllerAdvice
public class GlobalExceptionHandler {

    /**
     * Счетчик ошибок, связанных непосредственно с предсказанием модели.
     * Инкрементируется, когда возникают исключения типа {@link ModelProcessingException}
     * или другие непредвиденные серверные ошибки во время обработки запроса.
     * Метрика в Prometheus: {@code predictions_errors_total}.
     */
    private final Counter predictionErrorCounter;

    /**
     * Конструктор для внедрения зависимости {@link MeterRegistry}.
     *
     * @param meterRegistry реестр метрик Micrometer, предоставляемый Spring Boot.
     *                      Используется для создания и регистрации счетчиков.
     */
    public GlobalExceptionHandler(MeterRegistry meterRegistry) {
        this.predictionErrorCounter = meterRegistry.counter("predictions.errors.total", "reason", "processing_failure");
    }

    /**
     * Обрабатывает исключение, когда статический ресурс (CSS, JS, и т.д.) или эндпоинт не найден.
     * Возвращает корректный HTTP-статус 404 Not Found.
     *
     * @param exception перехваченное исключение {@link NoResourceFoundException}.
     * @param request   текущий веб-запрос.
     * @return {@link ResponseEntity} со статусом 404 (Not Found).
     */
    @ExceptionHandler(NoResourceFoundException.class)
    public ResponseEntity<ApiErrorResponse> handleNoResourceFound(NoResourceFoundException exception, WebRequest request) {
        log.warn("Запрошен несуществующий ресурс: {}. Путь: {}", exception.getResourcePath(), request.getDescription(false));
        return buildErrorResponse(exception, HttpStatus.NOT_FOUND, request, "Запрошенный ресурс не найден.", false);
    }

    @ExceptionHandler(MissingServletRequestPartException.class)
    public ResponseEntity<ApiErrorResponse> handleMissingPart(MissingServletRequestPartException exception, WebRequest request) {
        String message = "Обязательная часть запроса '" + exception.getRequestPartName() + "' отсутствует.";
        return buildErrorResponse(exception, HttpStatus.BAD_REQUEST, request, message, false);
    }

    /**
     * Обрабатывает исключение, связанное с невалидными входными данными от клиента.
     *
     * @param exception перехваченное исключение {@link InvalidInputDataException}.
     * @param request   текущий веб-запрос.
     * @return {@link ResponseEntity} со статусом 400 (Bad Request).
     */
    @ExceptionHandler(InvalidInputDataException.class)
    public ResponseEntity<ApiErrorResponse> handleInvalidInput(InvalidInputDataException exception, WebRequest request) {
        return buildErrorResponse(exception, HttpStatus.BAD_REQUEST, request, exception.getMessage(), false);
    }

    /**
     * Обрабатывает исключение, возникающее в процессе обработки данных ML-моделью.
     *
     * @param exception перехваченное исключение {@link ModelProcessingException}.
     * @param request   текущий веб-запрос.
     * @return {@link ResponseEntity} со статусом 500 (Internal Server Error).
     */
    @ExceptionHandler(ModelProcessingException.class)
    public ResponseEntity<ApiErrorResponse> handleModelProcessingException(ModelProcessingException exception, WebRequest request) {
        return buildErrorResponse(exception, HttpStatus.INTERNAL_SERVER_ERROR, request, "Ошибка при обработке запроса моделью.", true);
    }

    /**
     * Обрабатывает исключение, возникающее при превышении максимального размера загружаемого файла.
     *
     * @param exception перехваченное исключение {@link MaxUploadSizeExceededException}.
     * @param request   текущий веб-запрос.
     * @return {@link ResponseEntity} со статусом 413 (Payload Too Large).
     */
    @ExceptionHandler(MaxUploadSizeExceededException.class)
    public ResponseEntity<ApiErrorResponse> handleMaxSizeException(MaxUploadSizeExceededException exception, WebRequest request) {
        return buildErrorResponse(exception, HttpStatus.PAYLOAD_TOO_LARGE, request, "Размер файла превышает установленный лимит.", false);
    }

    /**
     * Обрабатывает все остальные непредвиденные исключения как общую серверную ошибку.
     *
     * @param exception перехваченное исключение {@link Exception}.
     * @param request   текущий веб-запрос.
     * @return {@link ResponseEntity} со статусом 500 (Internal Server Error).
     */
    @ExceptionHandler(Exception.class)
    public ResponseEntity<ApiErrorResponse> handleGlobalException(Exception exception, WebRequest request) {
        return buildErrorResponse(exception, HttpStatus.INTERNAL_SERVER_ERROR, request, "Произошла непредвиденная внутренняя ошибка сервера.", true);
    }

    /**
     * Вспомогательный метод для построения стандартизированного ответа об ошибке.
     * Логирует полную информацию об ошибке и инкрементирует счетчик, если ошибка
     * связана с работой модели.
     *
     * @param exception     Перехваченное исключение.
     * @param status        HTTP-статус, который будет установлен в ответе.
     * @param request       Текущий веб-запрос для получения информации о пути.
     * @param clientMessage Сообщение об ошибке, предназначенное для клиента.
     * @param isModelError  Флаг, указывающий, связана ли ошибка с основной бизнес-логикой (моделью).
     * @return Готовый {@link ResponseEntity} с телом {@link ApiErrorResponse}.
     */
    private ResponseEntity<ApiErrorResponse> buildErrorResponse(Exception exception, HttpStatus status, WebRequest request, String clientMessage, boolean isModelError) {
        if (isModelError) {
            predictionErrorCounter.increment();
        }
        ApiErrorResponse errorDetails = new ApiErrorResponse(
                LocalDateTime.now(ZoneOffset.UTC),
                status.value(),
                status.getReasonPhrase(),
                clientMessage,
                request.getDescription(false)
        );

        // Логируем ошибки уровня 5xx как ERROR, а клиентские 4xx как WARN для чистоты логов
        if (status.is5xxServerError()) {
            log.error("Перехвачено исключение уровня 5xx: {}. Путь: {}. Сообщение для клиента: {}", exception.getClass().getSimpleName(), request.getDescription(false), clientMessage, exception);
        } else {
            log.warn("Перехвачено исключение уровня 4xx: {}. Путь: {}. Сообщение для клиента: {}", exception.getClass().getSimpleName(), request.getDescription(false), clientMessage);
        }

        return new ResponseEntity<>(errorDetails, status);
    }
}

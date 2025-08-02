package com.mlops.example.controller.dto;

import com.mlops.example.config.GlobalExceptionHandler;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

import java.time.LocalDateTime;

/**
 * Data Transfer Object (DTO) для стандартизированного представления информации об ошибке в API.
 * <p>
 * Используется в {@link GlobalExceptionHandler} для формирования тела HTTP-ответа
 * при возникновении исключений, обеспечивая консистентный формат ошибок для клиентов API.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class ApiErrorResponse {
    /**
     * Временная метка возникновения ошибки в формате UTC.
     * Помогает при отладке и сопоставлении логов.
     */
    private LocalDateTime timestamp;
    /**
     * HTTP-статус код ошибки (например, 400, 404, 500).
     */
    private int status;
    /**
     * Краткое наименование HTTP-статуса (например, "Bad Request", "Internal Server Error").
     */
    private String error;
    /**
     * Подробное, человекочитаемое сообщение об ошибке, предназначенное для клиента API.
     */
    private String message;
    /**
     * URI запроса, на котором произошла ошибка (например, "uri=/api/v1/predict").
     */
    private String path;
}

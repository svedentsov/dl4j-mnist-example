package com.mlops.example.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

/**
 * Пользовательское исключение, выбрасываемое при возникновении ошибки
 * в процессе обработки данных нейронной сетью (например, ошибка чтения файла, сбой при инференсе).
 * <p>
 * Аннотация {@link ResponseStatus} указывает Spring MVC автоматически возвращать
 * HTTP-статус 500 (Internal Server Error), сигнализируя о внутренней проблеме на сервере.
 */
@ResponseStatus(HttpStatus.INTERNAL_SERVER_ERROR)
public class ModelProcessingException extends RuntimeException {
    /**
     * Конструктор с сообщением об ошибке и исходной причиной.
     *
     * @param message детальное описание причины ошибки.
     * @param cause   исходное исключение (например, {@link java.io.IOException}), которое привело к данной ошибке.
     */
    public ModelProcessingException(String message, Throwable cause) {
        super(message, cause);
    }
}

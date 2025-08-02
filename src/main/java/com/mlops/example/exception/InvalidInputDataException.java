package com.mlops.example.exception;

import org.springframework.http.HttpStatus;
import org.springframework.web.bind.annotation.ResponseStatus;

/**
 * Пользовательское исключение, выбрасываемое, когда клиент передает некорректные
 * или неполные данные (например, пустой файл, неверный формат).
 * <p>
 * Аннотация {@link ResponseStatus} указывает Spring MVC автоматически возвращать
 * HTTP-статус 400 (Bad Request), когда это исключение не перехвачено явным обработчиком.
 */
@ResponseStatus(HttpStatus.BAD_REQUEST)
public class InvalidInputDataException extends RuntimeException {
    /**
     * Конструктор с сообщением об ошибке.
     *
     * @param message детальное описание причины ошибки, которое может быть показано клиенту.
     */
    public InvalidInputDataException(String message) {
        super(message);
    }
}

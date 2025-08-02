package com.mlops.example.controller.dto;

import com.mlops.example.controller.PredictionController;
import lombok.AllArgsConstructor;
import lombok.Data;
import lombok.NoArgsConstructor;

/**
 * Data Transfer Object (DTO) для представления результата успешного предсказания.
 * <p>
 * Является телом ответа для эндпоинтов в {@link PredictionController}.
 */
@Data
@AllArgsConstructor
@NoArgsConstructor
public class PredictionResponse {
    /**
     * Целочисленное значение (0-9), предсказанное нейронной сетью
     * на основе входного изображения.
     */
    private int predictedDigit;
}

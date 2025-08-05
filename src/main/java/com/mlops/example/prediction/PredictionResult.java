package com.mlops.example.prediction;

import lombok.Getter;

import java.text.DecimalFormat;
import java.util.LinkedHashMap;
import java.util.Map;
import java.util.stream.Collectors;

/**
 * Структурированный, иммутабельный объект для хранения результата предсказания для одного экземпляра.
 * <p>
 * Этот класс является Data Transfer Object (DTO), который инкапсулирует два ключевых аспекта предсказания:
 * <ol>
 *     <li><b>Что предсказала модель:</b> итоговая метка класса (`predictedLabel`).</li>
 *     <li><b>Насколько модель уверена:</b> полное распределение вероятностей по всем классам (`probabilities`).</li>
 * </ol>
 * Наличие полного распределения вероятностей крайне важно для инженера по качеству,
 * так как позволяет анализировать уверенность модели, выявлять неоднозначные случаи (например, когда
 * две метки имеют почти одинаковую вероятность) и устанавливать пороговые значения для принятия решений в реальных системах.
 */
@Getter
public class PredictionResult {

    /**
     * Итоговая метка класса, предсказанная моделью. Это класс, получивший наивысшую вероятность.
     */
    private final int predictedLabel;

    /**
     * Карта, содержащая полное распределение вероятностей по всем классам.
     * <ul>
     *     <li><b>Ключ (Integer):</b> Метка класса (например, цифра от 0 до 9).</li>
     *     <li><b>Значение (Double):</b> Вычисленная вероятность принадлежности к этому классу (от 0.0 до 1.0).</li>
     * </ul>
     * Внутри объекта эта карта отсортирована по убыванию вероятностей для удобства анализа.
     */
    private final Map<Integer, Double> probabilities;

    /**
     * Конструктор для создания объекта результата.
     *
     * @param predictedLabel Метка класса с наивысшей вероятностью.
     * @param probabilities  Сырая карта вероятностей, полученная от модели.
     */
    public PredictionResult(int predictedLabel, Map<Integer, Double> probabilities) {
        this.predictedLabel = predictedLabel;
        // Сортируем вероятности по убыванию для удобного отображения
        this.probabilities = probabilities.entrySet().stream()
                .sorted(Map.Entry.<Integer, Double>comparingByValue().reversed())
                .collect(Collectors.toMap(
                        Map.Entry::getKey,
                        Map.Entry::getValue,
                        (e1, e2) -> e1,
                        LinkedHashMap::new));
    }

    /**
     * Возвращает человеко-читаемое строковое представление результата.
     * Идеально подходит для вывода в лог или консоль.
     * <p>
     * Формат вывода включает в себя итоговый предсказанный класс и 5 наиболее
     * вероятных классов с их вероятностями в процентном формате.
     *
     * @return Форматированная строка с результатами предсказания.
     */
    @Override
    public String toString() {
        DecimalFormat df = new DecimalFormat("#.##%");
        StringBuilder sb = new StringBuilder();
        sb.append("ИТОГОВОЕ ПРЕДСКАЗАНИЕ: ").append(predictedLabel).append("\n");
        sb.append("------------------------------------------\n");
        sb.append("Распределение вероятностей (топ-5):\n");
        probabilities.entrySet().stream()
                .limit(5)
                .forEach(entry -> sb.append(String.format("    Цифра %d: %-10s%n", entry.getKey(), df.format(entry.getValue()))));
        sb.append("------------------------------------------");
        return sb.toString();
    }
}

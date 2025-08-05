package com.mlops.example.prediction;

import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;

import java.io.File;
import java.io.IOException;
import java.util.HashMap;
import java.util.Map;

/**
 * Класс, инкапсулирующий логику использования обученной модели (инференса).
 * Предоставляет методы для валидации на тестовом наборе и для предсказания на одном изображении.
 * Отделен от основной логики сервиса для переиспользования в CLI-утилитах.
 */
@Slf4j
public class ModelPredictor {

    private final MultiLayerNetwork model;
    private final int height = 28;
    private final int width = 28;
    private final int channels = 1;

    public ModelPredictor(MultiLayerNetwork model) {
        this.model = model;
    }

    /**
     * Выполняет валидацию модели на всем тестовом наборе данных MNIST.
     * Этот метод является ключевым для регрессионного тестирования модели.
     *
     * @return Объект {@link Evaluation} с результатами оценки (accuracy, F1, etc.).
     * @throws IOException если возникает ошибка при загрузке данных MNIST.
     */
    public Evaluation validateOnTestSet() throws IOException {
        log.info("Режим валидации: Проверка модели на тестовом наборе MNIST.");
        log.info("Загрузка тестового набора данных...");
        // Используем batch size 128 для быстрой оценки. Seed важен для воспроизводимости.
        DataSetIterator mnistTest = new MnistDataSetIterator(128, false, 12345);
        log.info("Выполнение оценки...");
        Evaluation eval = model.evaluate(mnistTest);
        log.info("Оценка завершена.");
        return eval;
    }

    /**
     * Выполняет предсказание для одного файла изображения.
     *
     * @param imageFile Файл изображения для предсказания.
     * @return Объект {@link PredictionResult}, содержащий результат.
     * @throws IOException при ошибке чтения файла.
     */
    public PredictionResult predictSingleImage(File imageFile) throws IOException {
        log.info("Режим предсказания: Обработка файла: {}", imageFile.getAbsolutePath());
        // 1. Загрузка и предобработка изображения
        INDArray imageMatrix = preprocessImage(imageFile);
        // 2. Выполнение предсказания
        INDArray output = model.output(imageMatrix);
        // 3. Формирование структурированного результата
        int predictedLabel = output.argMax(1).getInt(0);
        Map<Integer, Double> probabilities = new HashMap<>();
        for (int i = 0; i < output.columns(); i++) {
            probabilities.put(i, output.getDouble(i));
        }
        return new PredictionResult(predictedLabel, probabilities);
    }
    
    /**
     * Приводит изображение к формату, который "понимает" нейронная сеть.
     * @param imageFile Файл для обработки.
     * @return Матрица INDArray, готовая для подачи в модель.
     * @throws IOException при ошибке чтения файла.
     */
    private INDArray preprocessImage(File imageFile) throws IOException {
        // Загрузчик, который автоматически меняет размер и количество каналов
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray imageMatrix = loader.asMatrix(imageFile);
        // Важно: используем тот же скейлер, что и при обучении (данные MNIST нормированы на [0,1])
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(imageMatrix);
        return imageMatrix;
    }
}
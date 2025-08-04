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
 * Класс, инкапсулирующий логику использования обученной модели.
 * Предоставляет методы для валидации на тестовом наборе и для предсказания на одном изображении.
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
     *
     * @return Объект {@link Evaluation} с результатами оценки.
     * @throws IOException если возникает ошибка при загрузке данных MNIST.
     */
    public Evaluation validateOnTestSet() throws IOException {
        log.info("Режим валидации: Проверка модели на тестовом наборе MNIST.");
        log.info("Загрузка тестового набора данных...");
        DataSetIterator mnistTest = new MnistDataSetIterator(128, false, 123);
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
        // 3. Формирование результата
        int predictedLabel = output.argMax(1).getInt(0);
        Map<Integer, Double> probabilities = new HashMap<>();
        for (int i = 0; i < output.columns(); i++) {
            probabilities.put(i, output.getDouble(i));
        }
        return new PredictionResult(predictedLabel, probabilities);
    }
    
    private INDArray preprocessImage(File imageFile) throws IOException {
        NativeImageLoader loader = new NativeImageLoader(height, width, channels);
        INDArray imageMatrix = loader.asMatrix(imageFile);
        // Важно: используем тот же скейлер, что и при обучении (MNIST данные нормированы на [0,1])
        DataNormalization scaler = new ImagePreProcessingScaler(0, 1);
        scaler.transform(imageMatrix);
        return imageMatrix;
    }
}
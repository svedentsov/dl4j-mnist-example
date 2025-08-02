package com.mlops.example.training;

import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.DataSet;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;

/**
 * Исполняемый класс для демонстрации использования предварительно обученной модели.
 * <p>
 * Этот скрипт выполняет следующие действия:
 * <ol>
 *     <li>Проверяет наличие файла с обученной моделью.</li>
 *     <li>Загружает (десериализует) модель из файла.</li>
 *     <li>Берет один пример из тестового набора данных MNIST для симуляции нового входа.</li>
 *     <li>Выполняет предсказание на этом примере.</li>
 *     <li>Сравнивает предсказанную метку с реальной и выводит результат.</li>
 * </ol>
 * Служит для быстрой проверки работоспособности сохраненной модели локально,
 * без необходимости запускать веб-сервис.
 */
public class UsePretrainedModel {

    /**
     * Главный метод, запускающий процесс инференса на одном примере.
     *
     * @param args Аргументы командной строки (не используются).
     * @throws Exception если возникают ошибки при загрузке модели или данных.
     */
    public static void main(String[] args) throws Exception {

        File modelFile = new File("pretrained_mnist_model.zip");

        if (!modelFile.exists()) {
            System.err.println("Файл модели 'pretrained_mnist_model.zip' не найден!");
            System.err.println("Пожалуйста, сначала запустите TrainMnistModel для обучения и сохранения модели.");
            return;
        }

        // 1. Загрузка предварительно обученной модели
        System.out.println("Загрузка модели из файла...");
        MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);
        System.out.println("Модель успешно загружена.");

        // 2. Получение тестовых данных для предсказания
        // Мы возьмем один пример из тестового набора данных, чтобы симулировать новые данные.
        // В реальном приложении здесь будет код для загрузки и обработки вашего изображения.
        System.out.println("Загрузка тестового примера из MNIST...");
        // Загружаем только 1 пример (batchSize = 1), не для обучения (false), с фиксированным seed
        DataSetIterator mnistTest = new MnistDataSetIterator(1, false, 123);
        DataSet testData = mnistTest.next();

        // Получаем изображение (признаки)
        INDArray features = testData.getFeatures();
        // Получаем реальную метку (для проверки)
        INDArray labels = testData.getLabels();
        int actualLabel = labels.argMax(1).getInt(0);

        // 3. Выполнение предсказания
        System.out.println("Выполнение предсказания...");
        INDArray predictedProbabilities = model.output(features);
        int predictedLabel = predictedProbabilities.argMax(1).getInt(0);

        System.out.println("------------------------------------");
        System.out.println("Предсказание для одного примера:");
        System.out.println("Реальная цифра: " + actualLabel);
        System.out.println("Предсказанная цифра: " + predictedLabel);
        System.out.println("------------------------------------");

        if (actualLabel == predictedLabel) {
            System.out.println("Результат верный!");
        } else {
            System.out.println("Модель ошиблась.");
        }
    }
}

package com.mlops.example.training;

import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.nn.weights.WeightInit;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

import java.io.File;

/**
 * Исполняемый класс для обучения нейронной сети на датасете MNIST.
 * <p>
 * Этот скрипт выполняет полный цикл обучения модели:
 * <ol>
 *     <li>Определение гиперпараметров.</li>
 *     <li>Загрузка и подготовка данных.</li>
 *     <li>Конфигурация архитектуры сверточной нейронной сети (CNN).</li>
 *     <li>Процесс обучения модели на тренировочных данных.</li>
 *     <li>Оценка качества модели на тестовых данных.</li>
 *     <li>Сериализация (сохранение) обученной модели в файл для последующего использования.</li>
 * </ol>
 * Этот класс не является частью веб-сервиса, а представляет собой отдельный MLOps-шаг,
 * который может быть запущен в CI/CD пайплайне (как показано в `.gitlab-ci.yml`).
 */
@Slf4j
public class TrainMnistModel {

    /**
     * Главный метод, запускающий процесс обучения.
     *
     * @param args Аргументы командной строки (не используются).
     * @throws Exception если возникают ошибки при загрузке данных или обучении.
     */
    public static void main(String[] args) throws Exception {

        // 1. Определение гиперпараметров обучения
        final int batchSize = 64; // Количество примеров, обрабатываемых за одну итерацию (шаг градиентного спуска).
        final int nEpochs = 2; // Количество полных проходов по всему набору данных.
        final int seed = 123; // Зерно для генератора случайных чисел для воспроизводимости результатов обучения.

        // Инициализация хранилища статистики для UI
        // UI сервер (http://localhost:9000) должен быть запущен отдельно.
        // Этот код будет отправлять статистику обучения на сервер UI для визуального мониторинга.
        InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
        log.info("UI для мониторинга должен быть запущен отдельно (deeplearning4j-ui).");
        log.info("Статистика обучения будет отправляться на http://localhost:9000");

        // 2. Загрузка данных MNIST
        log.info("Загрузка и подготовка данных...");
        DataSetIterator mnistTrain = new MnistDataSetIterator(batchSize, true, seed);
        DataSetIterator mnistTest = new MnistDataSetIterator(batchSize, false, seed);

        // 3. Создание архитектуры нейронной сети (подобно LeNet)
        log.info("Создание архитектуры модели...");
        MultiLayerConfiguration conf = new NeuralNetConfiguration.Builder()
                .seed(seed)
                .updater(new Adam(1e-3)) // Оптимизатор Adam с шагом обучения 0.001
                .weightInit(WeightInit.XAVIER) // Метод инициализации весов, хорошо подходит для глубоких сетей.
                .list()
                .layer(new ConvolutionLayer.Builder(5, 5).nIn(1).stride(1, 1).nOut(20).activation(Activation.IDENTITY).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
                .layer(new ConvolutionLayer.Builder(5, 5).stride(1, 1).nOut(50).activation(Activation.IDENTITY).build())
                .layer(new SubsamplingLayer.Builder(PoolingType.MAX).kernelSize(2, 2).stride(2, 2).build())
                .layer(new DenseLayer.Builder().activation(Activation.RELU).nOut(500).build())
                .layer(new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD).nOut(10).activation(Activation.SOFTMAX).build())
                .setInputType(InputType.convolutionalFlat(28, 28, 1)) // Указываем формат входных данных: 28x28 пикселей, 1 канал (ч/б).
                .build();

        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init();

        // Подключение слушателей к модели
        // 1. StatsListener для отправки данных в UI
        // 2. ScoreIterationListener для логирования значения функции потерь в консоль каждые 100 итераций
        model.setListeners(new StatsListener(statsStorage), new ScoreIterationListener(100));

        // 4. Обучение модели
        log.info("Начало обучения модели на {} эпох...", nEpochs);
        model.fit(mnistTrain, nEpochs);
        log.info("Обучение завершено.");

        // 5. Оценка модели
        log.info("Оценка точности модели на тестовых данных...");
        Evaluation eval = model.evaluate(mnistTest);
        log.info(eval.stats()); // Вывод метрик: accuracy, precision, recall, F1-score.

        // 6. Сохранение предварительно обученной модели
        File modelFile = new File("pretrained_mnist_model.zip");
        // Сохраняем состояние оптимизатора (updater) для возможного дообучения в будущем.
        boolean saveUpdater = true;
        ModelSerializer.writeModel(model, modelFile, saveUpdater);
        log.info("Модель успешно сохранена в файл: {}", modelFile.getAbsolutePath());
    }
}

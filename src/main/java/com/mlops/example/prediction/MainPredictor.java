package com.mlops.example.prediction;

import com.mlops.example.training.MainTrainer;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;
import picocli.CommandLine.Parameters;

import java.io.File;
import java.util.concurrent.Callable;

/**
 * Точка входа для использования обученной модели.
 * Использует picocli для создания CLI с двумя подкомандами:
 * 1. `validate`: для оценки модели на полном тестовом наборе MNIST.
 * 2. `predict`: для предсказания класса одного изображения.
 * <p>
 * ОСОБЕННОСТИ ЗАПУСКА
 * 1. Запуск по умолчанию (без аргументов): автоматически выполняет команду `validate`,
 * используя модель `pretrained_mnist_model.zip` из корневой директории проекта.
 * 2. Запуск с аргументами: работает как стандартный CLI-инструмент.
 * <p>
 * ПРИМЕРЫ ЗАПУСКА
 * <pre>
 * {@code
 * # 1. Запуск по умолчанию (валидация модели из корня проекта)
 * mvn compile exec:java -Dexec.mainClass="com.mlops.example.prediction.MainPredictor"
 *
 * # 2. Явный запуск валидации
 * mvn compile exec:java -Dexec.mainClass="com.mlops.example.prediction.MainPredictor" -Dexec.args="validate -m path/to/model.zip"
 *
 * # 3. Запуск предсказания для одного файла изображения
 * mvn compile exec:java -Dexec.mainClass="com.mlops.example.prediction.MainPredictor" -Dexec.args="predict path/to/my-digit-image.png"
 * }
 * </pre>
 */
@Slf4j
@Command(
        name = "use-model",
        mixinStandardHelpOptions = true,
        version = "1.0",
        description = "Утилита для валидации и предсказания с использованием обученной модели MNIST.",
        subcommands = {MainPredictor.ValidateCommand.class, MainPredictor.PredictCommand.class}
)
public class MainPredictor implements Callable<Integer> {

    // Имя файла модели по умолчанию, которое будет искаться в корне проекта.
    // Это значение используется как в логике `main` метода, так и в аннотациях picocli.
    private static final String DEFAULT_MODEL_FILENAME = "pretrained_mnist_model.zip";

    @Override
    public Integer call() {
        // Если не указана подкоманда, выводим справку.
        // Это сработает, только если пользователь введет "use-model" без ничего дальше.
        // Наш `main` метод перехватит случай полного отсутствия аргументов.
        CommandLine.usage(this, System.out);
        return 0;
    }

    public static void main(String[] args) {
        // НОВАЯ ЛОГИКА ДЛЯ ЗАПУСКА ПО УМОЛЧАНИЮ
        if (args.length == 0) {
            log.info("Аргументы не указаны. Запуск валидации по умолчанию...");
            File defaultModel = new File(DEFAULT_MODEL_FILENAME);
            if (!defaultModel.exists()) {
                log.error("ОШИБКА: Файл модели по умолчанию '{}' не найден в корне проекта.", DEFAULT_MODEL_FILENAME);
                log.error("Пожалуйста, сначала запустите обучение с помощью класса '{}'", MainTrainer.class.getName());
                log.error("Или скопируйте файл .zip с моделью в корень проекта.");
                System.exit(1); // Завершаем с кодом ошибки
            }
            // Формируем аргументы для запуска команды 'validate' с моделью по умолчанию
            args = new String[]{"validate", "-m", DEFAULT_MODEL_FILENAME};
        }
        // Передаем управление picocli с оригинальными или сгенерированными по умолчанию аргументами
        int exitCode = new CommandLine(new MainPredictor()).execute(args);
        System.exit(exitCode);
    }

    // Подкоманда `validate`
    @Command(name = "validate", description = "Прогоняет модель через тестовый набор MNIST и выводит метрики качества.")
    static class ValidateCommand implements Callable<Integer> {
        @Option(names = {"-m", "--model-path"},
                description = "Путь к файлу модели (.zip). По умолчанию ищется в корне проекта.",
                defaultValue = DEFAULT_MODEL_FILENAME)
        private File modelFile;

        @Override
        public Integer call() throws Exception {
            if (!modelFile.exists()) {
                log.error("Файл модели не найден: {}", modelFile.getAbsolutePath());
                return 1;
            }
            log.info("Загрузка модели из файла: {}", modelFile.getAbsolutePath());
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            ModelPredictor predictor = new ModelPredictor(model);
            Evaluation evaluation = predictor.validateOnTestSet();

            log.info("РЕЗУЛЬТАТЫ ВАЛИДАЦИИ");
            log.info("\n{}", evaluation.stats(true)); // `true` для вывода матрицы ошибок
            return 0;
        }
    }

    // Подкоманда `predict`
    @Command(name = "predict", description = "Делает предсказание для одного файла изображения.")
    static class PredictCommand implements Callable<Integer> {
        @Option(names = {"-m", "--model-path"},
                description = "Путь к файлу модели (.zip). По умолчанию ищется в корне проекта.",
                defaultValue = DEFAULT_MODEL_FILENAME)
        private File modelFile;

        @Parameters(index = "0", description = "Путь к файлу изображения (например, PNG) для предсказания.")
        private File imageFile;

        @Override
        public Integer call() throws Exception {
            if (!modelFile.exists()) {
                log.error("Файл модели не найден: {}", modelFile.getAbsolutePath());
                return 1;
            }
            if (!imageFile.exists()) {
                log.error("Файл изображения не найден: {}", imageFile.getAbsolutePath());
                return 1;
            }

            log.info("Загрузка модели из файла: {}", modelFile.getAbsolutePath());
            MultiLayerNetwork model = ModelSerializer.restoreMultiLayerNetwork(modelFile);

            ModelPredictor predictor = new ModelPredictor(model);
            PredictionResult result = predictor.predictSingleImage(imageFile);

            log.info("Результат предсказания");
            log.info("\n{}", result);
            return 0;
        }
    }
}

package com.mlops.example.training;

import com.mlops.example.config.TrainingConfig;
import lombok.extern.slf4j.Slf4j;
import picocli.CommandLine;
import picocli.CommandLine.Command;
import picocli.CommandLine.Option;

import java.nio.file.Paths;
import java.util.concurrent.Callable;

/**
 * Точка входа (Entry Point) для запуска процесса обучения модели.
 * <p>
 * Этот класс спроектирован как утилита командной строки (CLI), используя библиотеку {@link picocli.CommandLine}.
 * Его основная задача — разобрать аргументы, переданные пользователем, создать на их основе
 * строго типизированный объект конфигурации {@link TrainingConfig} и передать его в {@link ModelTrainer}
 * для выполнения основной работы.
 * <p>
 * Такой подход (разделение парсинга CLI и основной логики) является лучшей практикой, так как:
 * <ul>
 *     <li>Делает основную логику (ModelTrainer) независимой от способа ее вызова и легко тестируемой.</li>
 *     <li>Предоставляет пользователю мощный и удобный интерфейс с автогенерацией справки (help-меню).</li>
 * </ul>
 */
@Slf4j
@Command(
        name = "train-model",
        mixinStandardHelpOptions = true, // Автоматически добавляет опции --help и --version
        version = "1.0",
        description = "Запускает процесс обучения, оценки и сохранения нейронной сети MNIST."
)
public class MainTrainer implements Callable<Integer> {

    /**
     * Определяет, сколько раз модель "увидит" весь тренировочный набор данных.
     */
    @Option(names = {"-e", "--epochs"}, description = "Количество эпох обучения. По умолчанию: ${DEFAULT-VALUE}.")
    private int epochs = TrainingConfig.defaultConfig().getEpochs();

    /**
     * Определяет, сколько примеров будет обработано за одну итерацию обновления весов.
     */
    @Option(names = {"-b", "--batch-size"}, description = "Размер батча. По умолчанию: ${DEFAULT-VALUE}.")
    private int batchSize = TrainingConfig.defaultConfig().getBatchSize();

    /**
     * Контролирует скорость, с которой модель адаптируется к данным.
     */
    @Option(names = {"-lr", "--learning-rate"}, description = "Скорость обучения. По умолчанию: ${DEFAULT-VALUE}.")
    private double learningRate = TrainingConfig.defaultConfig().getLearningRate();

    /**
     * Ключевой параметр для обеспечения воспроизводимости экспериментов.
     */
    @Option(names = {"-s", "--seed"}, description = "Seed для ГСЧ для воспроизводимости. По умолчанию: ${DEFAULT-VALUE}.")
    private int seed = TrainingConfig.defaultConfig().getRandomSeed();

    /**
     * Позволяет пользователю указать, где будут храниться результаты обучения (модели, отчеты).
     */
    @Option(names = {"-o", "--output-dir"}, description = "Базовая директория для сохранения результатов. По умолчанию: ${DEFAULT-VALUE}.")
    private String outputDir = TrainingConfig.defaultConfig().getBaseOutputDir().toString();

    /**
     * Основной метод, который будет выполнен picocli после успешного парсинга аргументов.
     * Реализует интерфейс {@link Callable}, что позволяет возвращать код завершения.
     *
     * @return 0 при успехе, 1 при ошибке.
     */
    @Override
    public Integer call() {
        log.info("Аргументы командной строки успешно разобраны. Создание конфигурации запуска...");

        // 1. Создаем иммутабельный объект конфигурации на основе значений,
        // полученных из командной строки (или значений по умолчанию).
        TrainingConfig config = TrainingConfig.builder()
                .epochs(epochs)
                .batchSize(batchSize)
                .learningRate(learningRate)
                .randomSeed(seed)
                .baseOutputDir(Paths.get(outputDir))
                .saveUpdater(true) // В данном случае, всегда сохраняем состояние оптимизатора.
                .build();

        // 2. Создаем экземпляр "работника" (ModelTrainer) и передаем ему конфигурацию,
        // после чего запускаем основной процесс.
        try {
            ModelTrainer trainer = new ModelTrainer(config);
            trainer.run();
            log.info("Задание успешно выполнено.");
            return 0; // Возвращаем код успешного завершения (exit code 0).
        } catch (Exception e) {
            log.error("Критическая ошибка во время выполнения процесса обучения!", e);
            return 1; // Возвращаем код ошибки (exit code 1).
        }
    }

    /**
     * Главный метод (main), который является точкой входа для JVM.
     * <p>
     * Он инициализирует {@link CommandLine}, передает ему этот класс как команду
     * и запускает парсинг аргументов (`execute(args)`).
     * <p>
     * <b>Примеры запуска из командной строки:</b>
     * <pre>
     * {@code
     * # Запуск с параметрами по умолчанию (используются значения из TrainingConfig.defaultConfig())
     * mvn compile exec:java -Dexec.mainClass="com.mlops.example.training.MainTrainer"
     *
     * # Запуск с кастомными параметрами для эксперимента
     * mvn compile exec:java -Dexec.mainClass="com.mlops.example.training.MainTrainer" -Dexec.args="--epochs 5 --batch-size 128 -o ./my_models"
     *
     * # Просмотр справки
     * mvn compile exec:java -Dexec.mainClass="com.mlops.example.training.MainTrainer" -Dexec.args="--help"
     * }
     * </pre>
     */
    public static void main(String[] args) {
        // `execute` возвращает код завершения, который мы передаем в `System.exit`.
        // Это стандартная практика для CLI-приложений, позволяющая интегрировать их
        // в CI/CD пайплайны и скрипты.
        int exitCode = new CommandLine(new MainTrainer()).execute(args);
        System.exit(exitCode);
    }
}

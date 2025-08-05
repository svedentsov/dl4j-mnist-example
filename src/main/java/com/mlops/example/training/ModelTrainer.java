package com.mlops.example.training;

import com.fasterxml.jackson.databind.ObjectMapper;
import com.fasterxml.jackson.databind.SerializationFeature;
import com.mlops.example.config.TrainingConfig;
import com.mlops.example.model.LeNetModelFactory;
import lombok.extern.slf4j.Slf4j;
import org.deeplearning4j.datasets.iterator.impl.MnistDataSetIterator;
import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.optimize.listeners.ScoreIterationListener;
import org.deeplearning4j.ui.model.stats.StatsListener;
import org.deeplearning4j.ui.model.storage.InMemoryStatsStorage;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.evaluation.classification.Evaluation;
import org.nd4j.linalg.dataset.api.iterator.DataSetIterator;

import java.io.File;
import java.io.FileWriter;
import java.io.IOException;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.StandardCopyOption;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;
import java.util.Properties;

/**
 * Оркестратор процесса обучения модели.
 * Этот класс отвечает за полный, последовательный и воспроизводимый цикл:
 * загрузка данных, создание модели по спецификации, обучение, оценка и сохранение
 * всех результирующих артефактов. Он является основной исполняемой логикой MLOps-пайплайна.
 */
@Slf4j
public class ModelTrainer {

    private final TrainingConfig config;
    private final Path runOutputDir;

    // Имена файлов для артефактов. Использование констант обеспечивает консистентность.
    public static final String MODEL_FILENAME = "model.zip";
    public static final String BEST_MODEL_FILENAME = "pretrained_mnist_model.zip";
    public static final String METADATA_FILENAME = "model_metadata.properties";
    public static final String CONFIG_FILENAME = "training_config.json";
    public static final String EVALUATION_FILENAME = "evaluation_report.txt";

    /**
     * Конструктор тренера.
     *
     * @param config Объект конфигурации, содержащий все параметры для этого запуска.
     */
    public ModelTrainer(TrainingConfig config) {
        this.config = config;
        this.runOutputDir = config.getRunOutputDir();
    }

    /**
     * Запускает полный пайплайн обучения модели. Метод последовательно выполняет все
     * необходимые шаги, логируя каждый этап.
     *
     * @throws IOException в случае ошибок ввода-вывода при работе с файлами.
     */
    public void run() throws IOException {
        long startTime = System.currentTimeMillis();
        log.info("=== НАЧАЛО ПРОЦЕССА ОБУЧЕНИЯ МОДЕЛИ ===");
        log.info("Используемая конфигурация: {}", config);

        prepareOutputDirectory();
        DataSetIterator mnistTrain = loadDataSet(true);
        DataSetIterator mnistTest = loadDataSet(false);
        MultiLayerNetwork model = createAndInitModel();

        log.info("Шаг 4/6: Начало обучения на {} эпох...", config.getEpochs());
        trainModel(model, mnistTrain);
        log.info("Обучение завершено.");

        log.info("Шаг 5/6: Начало оценки модели на тестовой выборке...");
        Evaluation evaluation = model.evaluate(mnistTest);
        log.info("Оценка завершена.");

        log.info("Шаг 6/6: Сохранение артефактов...");
        saveArtifacts(model, evaluation);

        long endTime = System.currentTimeMillis();
        log.info("=== ПРОЦЕСС ОБУЧЕНИЯ УСПЕШНО ЗАВЕРШЕН за {} мс ===", (endTime - startTime));
        log.info("Все артефакты сохранены в директории: {}", runOutputDir.toAbsolutePath());
        log.info("Файл для использования сервисом скопирован в: {}", config.getBaseOutputDir().resolve(BEST_MODEL_FILENAME));
    }

    /**
     * Создает уникальную выходную директорию для текущего запуска.
     *
     * @throws IOException если не удается создать директорию.
     */
    private void prepareOutputDirectory() throws IOException {
        log.info("Шаг 1/6: Подготовка директории для результатов...");
        Files.createDirectories(runOutputDir);
        log.info("Создана директория для результатов: {}", runOutputDir.toAbsolutePath());
    }

    /**
     * Загружает набор данных MNIST (тренировочный или тестовый).
     *
     * @param isTrain {@code true} для тренировочного набора, {@code false} для тестового.
     * @return Итератор по набору данных.
     * @throws IOException если данные не могут быть загружены/найдены.
     */
    private DataSetIterator loadDataSet(boolean isTrain) throws IOException {
        String type = isTrain ? "тренировочных" : "тестовых";
        log.info("Шаг {}/6: Загрузка {} данных MNIST...", isTrain ? 2 : 3, type);
        return new MnistDataSetIterator(config.getBatchSize(), isTrain, config.getRandomSeed());
    }

    /**
     * Создает экземпляр нейронной сети с помощью {@link LeNetModelFactory} и инициализирует ее.
     *
     * @return Готовый к обучению объект {@link MultiLayerNetwork}.
     */
    private MultiLayerNetwork createAndInitModel() {
        log.info("Шаг 3/6: Создание архитектуры и инициализация модели...");
        LeNetModelFactory modelFactory = new LeNetModelFactory(config.getRandomSeed(), config.getLearningRate());
        MultiLayerConfiguration conf = modelFactory.build();
        MultiLayerNetwork model = new MultiLayerNetwork(conf);
        model.init(); // Инициализация весов модели согласно конфигурации.
        log.info("Модель успешно инициализирована.");
        return model;
    }

    /**
     * Выполняет обучение модели и настраивает слушателей для мониторинга.
     *
     * @param model     модель для обучения.
     * @param trainData итератор с тренировочными данными.
     */
    private void trainModel(MultiLayerNetwork model, DataSetIterator trainData) {
        // Настройка слушателей (listeners) для мониторинга процесса обучения.
        // Это важный инструмент для отладки и анализа поведения модели.
        InMemoryStatsStorage statsStorage = new InMemoryStatsStorage();
        model.setListeners(
                // StatsListener собирает подробную статистику (значения потерь, параметры слоев)
                // для отображения в веб-интерфейсе DL4J.
                new StatsListener(statsStorage),
                // ScoreIterationListener просто выводит значение функции потерь в лог каждые N итераций.
                // Полезно для быстрого контроля в консоли.
                new ScoreIterationListener(100)
        );
        log.info("Для визуального мониторинга запустите deeplearning4j-ui и откройте http://localhost:9000");
        model.fit(trainData, config.getEpochs());
    }

    /**
     * Сохраняет все артефакты, связанные с этим запуском обучения, в выходную директорию.
     *
     * @param model      обученная модель.
     * @param evaluation объект с результатами оценки.
     * @throws IOException при ошибках записи файлов.
     */
    private void saveArtifacts(MultiLayerNetwork model, Evaluation evaluation) throws IOException {
        // 1. Сохранение модели в уникальную директорию запуска
        Path modelPath = runOutputDir.resolve(MODEL_FILENAME);
        ModelSerializer.writeModel(model, modelPath.toFile(), config.isSaveUpdater());
        log.info("-> Артефакт модели сохранен в: {}", modelPath);

        // 2. Сохранение метаданных
        Path metadataPath = runOutputDir.resolve(METADATA_FILENAME);
        Properties metadata = createMetadata(evaluation);
        try (FileWriter writer = new FileWriter(metadataPath.toFile())) {
            metadata.store(writer, "ML Model Training Metadata");
        }
        log.info("-> Файл метаданных сохранен в: {}", metadataPath);

        // 3. Сохранение конфигурации
        Path configPath = runOutputDir.resolve(CONFIG_FILENAME);
        ObjectMapper mapper = new ObjectMapper().enable(SerializationFeature.INDENT_OUTPUT);
        mapper.writeValue(configPath.toFile(), config);
        log.info("-> Полная конфигурация запуска сохранена в: {}", configPath);

        // 4. Сохранение отчета об оценке
        Path reportPath = runOutputDir.resolve(EVALUATION_FILENAME);
        try (FileWriter writer = new FileWriter(reportPath.toFile())) {
            writer.write("Evaluation Report\n\n");
            writer.write(evaluation.stats(true)); // true для вывода матрицы ошибок
        }
        log.info("-> Отчет об оценке сохранен в: {}", reportPath);

        // 5. Копирование модели в корневую директорию для использования сервисом
        // В реальном CI/CD этот шаг был бы сложнее (например, загрузка в артефакт-репозиторий)
        Path targetModelPath = new File(BEST_MODEL_FILENAME).toPath();
        Files.copy(modelPath, targetModelPath, StandardCopyOption.REPLACE_EXISTING);
        log.info("-> Модель скопирована для использования сервисом.");
    }

    /**
     * Создает объект {@link Properties} с ключевыми метаданными для быстрого доступа.
     *
     * @param eval результаты оценки.
     * @return объект Properties с метаданными.
     */
    private Properties createMetadata(Evaluation eval) {
        Properties metadata = new Properties();
        metadata.setProperty("model.artifact.name", MODEL_FILENAME);
        metadata.setProperty("model.training.timestamp.utc", LocalDateTime.now().format(DateTimeFormatter.ISO_DATE_TIME));
        metadata.setProperty("model.training.epochs", String.valueOf(config.getEpochs()));
        metadata.setProperty("model.training.batchSize", String.valueOf(config.getBatchSize()));
        metadata.setProperty("model.training.learningRate", String.valueOf(config.getLearningRate()));
        metadata.setProperty("model.validation.accuracy", String.format("%.5f", eval.accuracy()));
        metadata.setProperty("model.validation.f1", String.format("%.5f", eval.f1()));
        metadata.setProperty("model.validation.precision", String.format("%.5f", eval.precision()));
        metadata.setProperty("model.validation.recall", String.format("%.5f", eval.recall()));
        return metadata;
    }
}

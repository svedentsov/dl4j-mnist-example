package com.mlops.example.service;

import com.mlops.example.exception.ModelProcessingException;
import jakarta.annotation.PostConstruct;
import lombok.extern.slf4j.Slf4j;
import org.datavec.image.loader.NativeImageLoader;
import org.deeplearning4j.nn.multilayer.MultiLayerNetwork;
import org.deeplearning4j.util.ModelSerializer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.dataset.api.preprocessor.DataNormalization;
import org.nd4j.linalg.dataset.api.preprocessor.ImagePreProcessingScaler;
import org.springframework.beans.factory.annotation.Value;
import org.springframework.stereotype.Service;
import org.springframework.web.multipart.MultipartFile;

import java.io.IOException;
import java.io.InputStream;
import java.nio.file.Files;
import java.nio.file.Path;

/**
 * Сервис, инкапсулирующий логику работы с ML-моделью. Отвечает за:
 * <ul>
 *     <li>Загрузку предварительно обученной модели из файла при старте приложения.</li>
 *     <li>Предобработку входных изображений (изменение размера, нормализация).</li>
 *     <li>Выполнение инференса (предсказания) на модели.</li>
 * </ul>
 * Является ядром бизнес-логики приложения.
 */
@Slf4j
@Service
public class ModelServingService {

    /**
     * Путь к файлу с сериализованной моделью. Инжектируется из `application.properties`.
     */
    @Value("${ml.model.path}")
    private String modelPath;

    /**
     * Загруженная модель нейронной сети. Поле объявлено как {@code volatile} для обеспечения
     * корректной "публикации" объекта модели в памяти после его инициализации в
     * методе {@code loadModel()}, что гарантирует его видимость для всех потоков
     * в многопоточной среде веб-сервера.
     */
    private volatile MultiLayerNetwork model;
    private final NativeImageLoader imageLoader;
    private final DataNormalization scaler;

    /**
     * Конструктор сервиса. Инициализирует компоненты для предобработки изображений.
     * Параметры (высота, ширина, каналы) жестко заданы в соответствии с архитектурой
     * сети, обученной на датасете MNIST.
     */
    public ModelServingService() {
        final int height = 28;
        final int width = 28;
        final int channels = 1; // MNIST - черно-белые изображения
        this.imageLoader = new NativeImageLoader(height, width, channels);
        // Нормализатор, приводящий значения пикселей из диапазона [0, 255] в [0, 1]
        this.scaler = new ImagePreProcessingScaler(0, 1);
    }

    /**
     * Метод, выполняющий загрузку и десериализацию ML-модели после инициализации бина.
     * Вызывается один раз при старте приложения благодаря аннотации {@link PostConstruct}.
     *
     * @throws RuntimeException если файл модели не найден или не может быть десериализован,
     *                          что является критической ошибкой, препятствующей работе сервиса.
     */
    @PostConstruct
    public void loadModel() {
        log.info("Загрузка ML-модели из пути: '{}'...", modelPath);
        Path path = Path.of(modelPath);

        if (!Files.exists(path)) {
            log.error("Критическая ошибка: Файл модели не найден по пути: {}", path.toAbsolutePath());
            throw new RuntimeException("Файл модели не найден: " + path.toAbsolutePath());
        }
        try {
            this.model = ModelSerializer.restoreMultiLayerNetwork(path.toFile(), false);
            log.info("ML-модель успешно загружена. Архитектура: {} слоев.", model.getnLayers());
        } catch (IOException e) {
            log.error("Критическая ошибка: Не удалось загрузить ML-модель из файла '{}'.", path.toAbsolutePath(), e);
            throw new RuntimeException("Не удалось десериализовать модель. Убедитесь, что файл не поврежден.", e);
        }
    }

    /**
     * Выполняет предсказание для одного изображения.
     *
     * @param imageFile Файл изображения, полученный от контроллера.
     * @return Предсказанная цифра (0-9).
     * @throws ModelProcessingException если возникает ошибка на любом этапе обработки изображения
     *                                  или выполнения инференса.
     */
    public int predict(MultipartFile imageFile) {
        // Использование try-with-resources для автоматического закрытия InputStream
        try (InputStream imageStream = imageFile.getInputStream()) {
            // 1. Преобразование изображения в матрицу нужного размера (28x28x1)
            INDArray imageMatrix = imageLoader.asMatrix(imageStream);
            if (imageMatrix == null) {
                // Это может произойти, если файл поврежден или не является изображением
                throw new IOException("Не удалось преобразовать изображение в матрицу.");
            }
            // 2. Нормализация данных
            scaler.transform(imageMatrix);
            // 3. Выполнение инференса (прямого прохода по сети)
            INDArray output = model.output(imageMatrix);
            // 4. Нахождение индекса с максимальной вероятностью (это и есть предсказанная цифра)
            return output.argMax(1).getInt(0);
        } catch (IOException e) {
            log.error("Ошибка ввода-вывода при обработке файла: {}", imageFile.getOriginalFilename(), e);
            throw new ModelProcessingException("Не удалось прочитать или обработать файл изображения.", e);
        } catch (Exception e) {
            // Перехват любых других неожиданных ошибок во время работы DL4J/ND4J
            log.error("Непредвиденная ошибка во время инференса модели для файла: {}", imageFile.getOriginalFilename(), e);
            throw new ModelProcessingException("Внутренняя ошибка при выполнении предсказания.", e);
        }
    }
}

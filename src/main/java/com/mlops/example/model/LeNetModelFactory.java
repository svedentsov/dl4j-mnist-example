package com.mlops.example.model;

import org.deeplearning4j.nn.conf.MultiLayerConfiguration;
import org.deeplearning4j.nn.conf.NeuralNetConfiguration;
import org.deeplearning4j.nn.conf.inputs.InputType;
import org.deeplearning4j.nn.conf.layers.*;
import org.deeplearning4j.nn.weights.WeightInit;
import org.nd4j.linalg.activations.Activation;
import org.nd4j.linalg.learning.config.Adam;
import org.nd4j.linalg.lossfunctions.LossFunctions;

/**
 * Фабрика для создания архитектуры нейронной сети в стиле LeNet.
 * Инкапсулирует детали конфигурации слоев, отделяя "что" (архитектура) от "как" (процесс обучения).
 * <p>
 * Архитектура LeNet-5 является одной из первых и самых известных сверточных
 * нейронных сетей (CNN) и хорошо подходит для задач распознавания изображений, таких как датасет MNIST.
 */
public class LeNetModelFactory {

    // Параметры входных данных
    private final int height = 28; // Высота изображения MNIST в пикселях
    private final int width = 28; // Ширина изображения MNIST в пикселях
    private final int channels = 1; // Количество цветовых каналов. 1 для черно-белых изображений.
    private final int numClasses = 10; // Количество выходных классов (цифры от 0 до 9).

    // Гиперпараметры модели
    private final int seed;
    private final double learningRate;

    /**
     * Конструктор фабрики.
     *
     * @param seed         Начальное значение для генератора случайных чисел. Крайне важно для
     *                     воспроизводимости результатов обучения. При одинаковом seed
     *                     начальные веса модели и другие случайные процессы будут идентичны.
     * @param learningRate Скорость обучения для оптимизатора Adam. Определяет, насколько
     *                     сильно модель будет корректировать свои веса на каждом шаге обучения.
     */
    public LeNetModelFactory(int seed, double learningRate) {
        this.seed = seed;
        this.learningRate = learningRate;
    }

    /**
     * Создает и возвращает конфигурацию нейронной сети.
     *
     * @return Готовый объект {@link MultiLayerConfiguration}.
     */
    public MultiLayerConfiguration build() {
        return new NeuralNetConfiguration.Builder()
                // Глобальные настройки сети

                // Устанавливает seed для ГСЧ, чтобы инициализация весов была воспроизводимой.
                // Это критично для отладки и сравнения моделей.
                .seed(seed)

                // Определяет алгоритм оптимизации, который будет использоваться для обновления весов модели
                // во время обучения. Adam - это эффективный и широко используемый адаптивный оптимизатор.
                .updater(new Adam(learningRate))

                // Задает метод инициализации весов. 'XAVIER' — это популярный метод, который помогает
                // предотвратить "затухание" или "взрыв" градиентов в глубоких сетях, что ускоряет обучение.
                .weightInit(WeightInit.XAVIER)

                // Начало определения списка слоев сети.
                .list()

                // Слой 0: Сверточный (Convolutional)
                // Этот слой ищет локальные признаки в изображении (например, края, углы).
                .layer(0, new ConvolutionLayer.Builder(5, 5)
                        // .kernelSize(5, 5): Размер "окна" или фильтра, который скользит по изображению. 5x5 пикселей.
                        // .nIn(channels): Количество входных каналов. Для первого слоя это глубина входного изображения (1 для ч/б).
                        .nIn(channels)
                        // .stride(1, 1): Шаг, с которым фильтр перемещается по изображению (1 пиксель по горизонтали, 1 по вертикали).
                        .stride(1, 1)
                        // .nOut(20): Количество фильтров, которые будут применены. Каждый фильтр учится распознавать свой признак.
                        // Это значение определяет "глубину" выхода слоя. Здесь мы ищем 20 разных признаков.
                        .nOut(20)
                        // .activation(Activation.IDENTITY): Функция активации. IDENTITY (f(x)=x) означает, что активация не применяется
                        // на этом этапе. В классической LeNet активация (ReLU) часто применяется после пулинга.
                        .activation(Activation.IDENTITY)
                        .build())

                // Слой 1: Подвыборочный (Subsampling/Pooling)
                // Этот слой уменьшает размерность карты признаков, делая представление более компактным
                // и устойчивым к небольшим сдвигам признаков на изображении.
                .layer(1, new SubsamplingLayer.Builder(PoolingType.MAX)
                        // .poolingType(PoolingType.MAX): MAX-пулинг. Из каждого региона выбирается максимальное значение.
                        // Это помогает сохранить самые яркие проявления найденных признаков.
                        // .kernelSize(2, 2): Размер региона, в котором происходит уплотнение (2x2 пикселя).
                        .kernelSize(2, 2)
                        // .stride(2, 2): Шаг смещения. При совпадении с kernelSize регионы не пересекаются.
                        // Это эффективно уменьшает высоту и ширину карты признаков в 2 раза.
                        .stride(2, 2)
                        .build())

                // Слой 2: Сверточный
                // Второй сверточный слой ищет более сложные признаки, комбинируя признаки, найденные на предыдущем слое.
                .layer(2, new ConvolutionLayer.Builder(5, 5)
                        .stride(1, 1)
                        // .nOut(50): Увеличиваем количество фильтров до 50. Обычная практика в CNN - увеличивать
                        // глубину (количество признаков) по мере уменьшения пространственного размера.
                        .nOut(50)
                        .activation(Activation.IDENTITY)
                        .build())

                // Слой 3: Подвыборочный
                // Снова уменьшаем размерность.
                .layer(3, new SubsamplingLayer.Builder(PoolingType.MAX)
                        .kernelSize(2, 2)
                        .stride(2, 2)
                        .build())

                // Слой 4: Полносвязный (Dense/Fully Connected)
                // Этот слой "сплющивает" 2D карты признаков в 1D вектор и соединяет каждый нейрон
                // с каждым нейроном предыдущего слоя. Его задача — провести классификацию на основе
                // высокоуровневых признаков, обнаруженных сверточными слоями.
                .layer(4, new DenseLayer.Builder()
                        // .activation(Activation.RELU): ReLU (Rectified Linear Unit) - популярная, эффективная
                        // функция активации (f(x) = max(0, x)). Она вносит нелинейность в модель.
                        .activation(Activation.RELU)
                        // .nOut(500): Количество нейронов в этом скрытом слое.
                        .nOut(500)
                        .build())

                // Слой 5: Выходной (Output)
                // Финальный слой, который выдает итоговый результат.
                .layer(5, new OutputLayer.Builder(LossFunctions.LossFunction.NEGATIVELOGLIKELIHOOD)
                        // .lossFunction(...): Функция потерь. NEGATIVELOGLIKELIHOOD - стандартная функция для задач
                        // мультиклассовой классификации, используется в паре с активацией SOFTMAX. Она измеряет,
                        // насколько предсказания модели далеки от истинных меток.
                        // .nOut(numClasses): Количество нейронов на выходе. Должно быть равно количеству классов (10 для MNIST).
                        .nOut(numClasses)
                        // .activation(Activation.SOFTMAX): Функция активации, которая преобразует сырые значения (логиты)
                        // в распределение вероятностей по всем классам. Сумма всех вероятностей равна 1.
                        .activation(Activation.SOFTMAX)
                        .build())

                // Определение типа входных данных
                // Это важный шаг, который сообщает DL4J, какую форму данных ожидать на входе.
                // Это позволяет фреймворку автоматически вычислять размеры входов (`.nIn()`) для слоев.
                // `convolutionalFlat` используется, когда данные (как в MNIST) поставляются в виде "плоского"
                // вектора, но должны быть интерпретированы как 2D-изображение.
                .setInputType(InputType.convolutionalFlat(height, width, channels))
                .build();
    }
}

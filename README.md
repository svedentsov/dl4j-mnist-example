# MLOps-демо: Сервис распознавания цифр MNIST

![Java](https://img.shields.io/badge/Java-17-blue.svg)
![Spring Boot](https://img.shields.io/badge/Spring_Boot-3.3.1-brightgreen.svg)
![Maven](https://img.shields.io/badge/Maven-3.9-orange.svg)
![Docker](https://img.shields.io/badge/Docker-Compose-blue.svg)
![Prometheus](https://img.shields.io/badge/Prometheus-monitoring-orange.svg)
![Grafana](https://img.shields.io/badge/Grafana-dashboard-green.svg)

Этот проект представляет собой полноценное MLOps-решение для развертывания, мониторинга и обслуживания модели машинного обучения. В качестве примера используется Java-сервис на базе **Spring Boot** и **Deeplearning4j (DL4J)**, который распознает рукописные цифры из датасета MNIST.

Проект демонстрирует ключевые практики MLOps:
*   **Контейнеризация:** Приложение и все сервисы мониторинга упакованы в Docker-контейнеры.
*   **Инфраструктура как код (IaC):** Весь стек разворачивается одной командой с помощью `docker-compose`.
*   **Мониторинг:** Сбор и визуализация метрик производительности и бизнес-метрик с помощью **Prometheus** и **Grafana**.
*   **Алертинг:** Настройка правил для автоматических уведомлений о проблемах с помощью **Alertmanager**.
*   **CI/CD Ready:** Проект включает статический анализ кода, проверку уязвимостей и разделение на unit/integration тесты, что позволяет легко встроить его в любой CI/CD пайплайн.

## Архитектура системы

Система состоит из нескольких взаимосвязанных сервисов, работающих в единой Docker-сети:

| Сервис | Порт | Описание |
| :--- | :--- | :--- |
| **ML Service** | `8080` | Основное приложение на Spring Boot, которое предоставляет REST API для распознавания цифр. Экспортирует метрики для Prometheus. |
| **Prometheus** | `9090` | Собирает и хранит метрики с ML-сервиса. Оценивает правила алертинга. |
| **Alertmanager** | `9093` | Получает алерты от Prometheus, группирует их и отправляет уведомления (в данном примере - на webhook). |
| **Grafana** | `3000` | Визуализирует метрики из Prometheus на готовом дашборде. |

**Схема взаимодействия:**
1.  Пользователь отправляет изображение на **ML Service**.
2.  **ML Service** обрабатывает запрос, генерируя метрики (время ответа, количество запросов, результат предсказания).
3.  **Prometheus** периодически опрашивает (scrape) эндпоинт `/actuator/prometheus` на **ML Service** и сохраняет метрики.
4.  Если метрики соответствуют правилам в `prometheus/rules.yml` (например, высокая задержка), **Prometheus** отправляет алерт в **Alertmanager**.
5.  **Alertmanager** обрабатывает алерт согласно своей конфигурации и отправляет уведомление.
6.  **Grafana** запрашивает данные у **Prometheus** и отображает их в виде графиков на дашборде.

## Ключевые особенности

*   **REST API:** Два эндпоинта для одиночного и пакетного распознавания изображений.
*   **Оптимизированный Docker-образ:** Многоэтапная сборка (`Dockerfile`) для уменьшения размера и повышения безопасности (запуск от non-root пользователя).
*   **Обучение модели:** Включает отдельный скрипт (`TrainMnistModel.java`) для обучения сверточной нейронной сети на датасете MNIST.
*   **Автоматический провижининг Grafana:** Источник данных Prometheus и дашборд создаются автоматически при первом запуске Grafana.
*   **Кастомные метрики:**
    *   `predictions_requests_total`: Общее количество запросов.
    *   `predictions_latency_seconds`: Латентность запросов (с перцентилями P50, P95, P99).
    *   `predictions_errors_total`: Количество ошибок при обработке моделью.
    *   `predictions_class_distribution_total`: Распределение предсказанных цифр.
*   **Готовые правила алертинга:**
    *   `HighRequestLatency`: Срабатывает при высокой задержке ответа.
    *   `HighPredictionErrorRate`: Срабатывает при высоком проценте ошибок модели.
    *   `ServiceDown`: Срабатывает, если сервис становится недоступным.

## Стек технологий

*   **Бэкенд:** Java 17, Spring Boot 3.3, Deeplearning4j (DL4J)
*   **Сборка:** Maven
*   **Контейнеризация:** Docker, Docker Compose
*   **Мониторинг:** Prometheus, Micrometer
*   **Визуализация:** Grafana
*   **Алертинг:** Alertmanager
*   **Статический анализ:** SpotBugs
*   **Проверка уязвимостей:** OWASP Dependency-Check

## Начало работы

### Предварительные требования
*   [Docker](https://www.docker.com/get-started) и [Docker Compose](https://docs.docker.com/compose/install/)
*   JDK 17
*   Maven 3.6+

### Запуск проекта

**1. Клонирование репозитория**
```bash
git clone <URL_репозитория>
cd <папка_проекта>
```

**2. Обучение и сохранение модели**
Перед первым запуском необходимо обучить модель. Скрипт `TrainMnistModel` создаст файл `pretrained_mnist_model.zip`, который используется сервисом.

```bash
# Этот шаг может занять несколько минут
mvn compile exec:java -Dexec.mainClass="com.mlops.example.training.TrainMnistModel"
```
После выполнения в корне проекта появится файл `pretrained_mnist_model.zip`.

**3. Сборка Java-приложения**
Соберем JAR-файл, который будет скопирован в Docker-образ.

```bash
mvn clean package
```

**4. Запуск всего стека**
Используйте Docker Compose для запуска всех сервисов в фоновом режиме.

```bash
docker-compose up -d
```

Система полностью готова к работе!

## Использование

### Доступ к сервисам

*   **Grafana:** [http://localhost:3000](http://localhost:3000) (логин: `admin`, пароль: `admin`)
    *   Дашборд "MNIST Service Dashboard" уже будет доступен в папке "ML Services".
*   **Prometheus:** [http://localhost:9090](http://localhost:9090)
    *   Проверить статус целей можно во вкладке `Status -> Targets`.
*   **Alertmanager:** [http://localhost:9093](http://localhost:9093)
    *   Здесь будут отображаться активные алерты.

### Отправка запросов на предсказание

Вы можете использовать `curl` или любой другой HTTP-клиент для отправки изображений.

**Одиночное предсказание:**
```bash
curl -X POST -F "image=@/путь/к/вашему/изображению.png" http://localhost:8080/api/v1/predict
```
*Ожидаемый ответ:*
```json
{
  "predictedDigit": 7
}
```

**Пакетное предсказание:**
```bash
curl -X POST \
  -F "images=@/путь/к/изображению1.png" \
  -F "images=@/путь/к/изображению2.png" \
  http://localhost:8080/api/v1/predict-batch
```
*Ожидаемый ответ:*
```json
[
  { "predictedDigit": 1 },
  { "predictedDigit": 9 }
]
```

## Структура проекта
```
.
├── alertmanager/
│   └── config.yml          # Конфигурация Alertmanager (маршрутизация, ресиверы)
├── grafana/
│   ├── provisioning/       # Автоматическая настройка Grafana
│   │   ├── dashboards/     # Директория с JSON-файлами дашбордов
│   │   └── datasources/    # Конфигурация источников данных (Prometheus)
├── prometheus/
│   └── rules.yml           # Правила алертинга для Prometheus
├── src/
│   ├── main/
│   │   ├── java/           # Исходный код Java-приложения
│   │   │   ├── config/     # Конфигурационные классы (обработчик ошибок)
│   │   │   ├── controller/ # REST-контроллеры
│   │   │   ├── exception/  # Пользовательские исключения
│   │   │   ├── service/    # Бизнес-логика (загрузка и использование модели)
│   │   │   └── training/   # Скрипты для обучения и тестирования модели
│   │   └── resources/
│   │       └── application.properties # Настройки Spring Boot
│   └── test/               # Тесты
├── .gitignore              # Файлы, исключенные из Git
├── .gitlab-ci.yml          # Пример конфигурации Alertmanager (может быть адаптирован для CI)
├── docker-compose.yml      # Определение и запуск всего стека сервисов
├── Dockerfile              # Инструкции для сборки Docker-образа ML-сервиса
└── pom.xml                 # Конфигурация проекта Maven (зависимости, плагины)
```

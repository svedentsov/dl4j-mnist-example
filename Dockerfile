# === Этап 1: Сборка проекта с использованием Maven (AS builder) ===
# На этом этапе используется образ Maven для компиляции исходного кода и сборки JAR-файла.
# Результаты этого этапа (только JAR-файл) будут скопированы в финальный образ.
FROM maven:3.9.6-eclipse-temurin-17 AS builder

# Установка рабочей директории внутри контейнера.
WORKDIR /build

# 1. Копируем только pom.xml. Этот шаг вынесен отдельно для эффективного кэширования слоев Docker.
# Docker пересобирает слой только если его исходные файлы изменились.
# Зависимости меняются реже, чем исходный код, поэтому этот слой будет кэшироваться чаще.
COPY pom.xml .

# 2. Используем Docker BuildKit cache mount для кэширования зависимостей Maven.
# Это современный и наиболее эффективный способ кэширования, который значительно ускоряет сборки.
# --mount=type=cache,target=/root/.m2/repository монтирует кэшируемый volume в директорию
# локального репозитория Maven на время выполнения команды RUN.
# `dependency:go-offline` загружает все зависимости проекта, но не компилирует код.
RUN --mount=type=cache,target=/root/.m2/repository \
    mvn -Dmaven.repo.local=/root/.m2/repository dependency:go-offline --fail-at-end

# 3. Копируем весь остальной исходный код.
# Этот слой будет пересобираться только при изменении файлов в директории `src`.
COPY src ./src

# 4. Собираем приложение.
# - Пропускаем тесты (`-Dmaven.test.skip=true`), так как они уже выполнены в CI/CD пайплайне.
# - Снова монтируем кэш на случай, если для сборки потребуются дополнительные плагины.
RUN --mount=type=cache,target=/root/.m2/repository \
    mvn -Dmaven.repo.local=/root/.m2/repository clean package -Dmaven.test.skip=true


# === Этап 2: Создание минимального runtime-образа ===
# На этом этапе используется легковесный образ только с Java Runtime Environment (JRE),
# так как для запуска приложения не требуется полный JDK.
FROM eclipse-temurin:17-jre-jammy

WORKDIR /app

# Создаем специального пользователя и группу с ограниченными правами для запуска приложения.
# Запуск от non-root пользователя является ключевой практикой безопасности контейнеров.
RUN groupadd --system appuser && useradd --system --gid appuser appuser
# Переключаемся на созданного пользователя.
USER appuser

# Копируем артефакт с обученной моделью из контекста сборки.
# Устанавливаем владельца файла, чтобы у пользователя 'appuser' были права на чтение.
COPY --chown=appuser:appuser ./pretrained_mnist_model.zip ./pretrained_mnist_model.zip

# Копируем собранный JAR-файл из предыдущего этапа (builder).
# Указываем владельца файла.
COPY --from=builder --chown=appuser:appuser /build/target/*.jar app.jar

# Указываем, что приложение будет слушать порт 8080.
EXPOSE 8080

# Команда, которая будет выполняться при запуске контейнера.
# Используется exec form (в виде JSON-массива), что является рекомендуемой практикой.
# -Dspring.profiles.active=prod - активирует производственный профиль Spring Boot.
ENTRYPOINT ["java", "-Dspring.profiles.active=prod", "-jar", "app.jar"]

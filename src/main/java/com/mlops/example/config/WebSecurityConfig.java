package com.mlops.example.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.config.annotation.web.configuration.EnableWebSecurity;
import org.springframework.security.web.SecurityFilterChain;
import org.springframework.security.web.util.matcher.AntPathRequestMatcher;

/**
 * Конфигурация безопасности для веб-приложения.
 * <p>
 * Этот класс использует Spring Security для определения правил доступа к HTTP-эндпоинтам.
 * Аннотация {@link EnableWebSecurity} активирует поддержку веб-безопасности Spring.
 */
@Configuration
@EnableWebSecurity
public class WebSecurityConfig {

    /**
     * Определяет цепочку фильтров безопасности, которая управляет доступом ко всем запросам.
     * <p>
     * Здесь мы настраиваем следующие правила:
     * <ul>
     *   <li>Разрешаем анонимный (неаутентифицированный) доступ к статическим ресурсам
     *       (CSS, JavaScript) и главной странице.</li>
     *   <li>Разрешаем доступ ко всем эндпоинтам нашего API (`/api/v1/**`).</li>
     *   <li>Разрешаем доступ к эндпоинтам Actuator для мониторинга (`/actuator/**`).</li>
     *   <li>Все остальные запросы (если таковые появятся) будут требовать аутентификации.</li>
     * </ul>
     * Это стандартная и необходимая конфигурация для любого веб-приложения,
     * чтобы отделить общедоступные ресурсы от защищенных.
     *
     * @param http объект для построения конфигурации безопасности.
     * @return сконфигурированная цепочка фильтров.
     * @throws Exception если при конфигурации возникает ошибка.
     */
    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http.authorizeHttpRequests(authorize -> authorize
                // Разрешаем доступ к главной странице и статическим ресурсам
                .requestMatchers(
                        AntPathRequestMatcher.antMatcher("/"),
                        AntPathRequestMatcher.antMatcher("/index.html"),
                        AntPathRequestMatcher.antMatcher("/css/**"),
                        AntPathRequestMatcher.antMatcher("/js/**")
                ).permitAll()
                // Разрешаем доступ к нашему API
                .requestMatchers(AntPathRequestMatcher.antMatcher("/api/v1/**")).permitAll()
                // Разрешаем доступ к эндпоинтам Actuator для Prometheus
                .requestMatchers(AntPathRequestMatcher.antMatcher("/actuator/**")).permitAll()
                // Все остальные запросы должны быть аутентифицированы
                .anyRequest().authenticated()
        );

        // В этом демо-проекте мы не используем формы входа, поэтому можем отключить их
        // для упрощения. В реальном приложении здесь была бы настройка loginPage().
        http.formLogin(formLogin -> formLogin.disable());
        // Также отключаем CSRF, так как наше простое API не управляет состоянием сессии.
        // Для более сложных API с аутентификацией на основе сессий CSRF-защиту следует настроить.
        http.csrf(csrf -> csrf.disable());

        return http.build();
    }
}

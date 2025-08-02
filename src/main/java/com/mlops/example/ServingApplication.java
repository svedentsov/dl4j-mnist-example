package com.mlops.example;

import org.springframework.boot.SpringApplication;
import org.springframework.boot.autoconfigure.SpringBootApplication;

/**
 * Главный класс приложения, точка входа для запуска Spring Boot.
 */
@SpringBootApplication
public class ServingApplication {
    public static void main(String[] args) {
        SpringApplication.run(ServingApplication.class, args);
    }
}

package com.mlops.example.controller;

import org.junit.jupiter.api.BeforeEach;
import org.junit.jupiter.api.Test;
import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.boot.test.context.SpringBootTest;
import org.springframework.boot.test.web.client.TestRestTemplate;
import org.springframework.boot.test.web.server.LocalServerPort;
import org.springframework.core.io.ClassPathResource;
import org.springframework.http.*;
import org.springframework.util.LinkedMultiValueMap;
import org.springframework.util.MultiValueMap;

import static org.assertj.core.api.Assertions.assertThat;

@SpringBootTest(webEnvironment = SpringBootTest.WebEnvironment.RANDOM_PORT)
public class PredictionControllerTest {

    @LocalServerPort
    private int port;
    @Autowired
    private TestRestTemplate restTemplate;

    @BeforeEach
    public void setup() {
        assertThat(new ClassPathResource("test-images/digit-7.png").exists()).isTrue();
    }

    @Test
    void whenPredictSingleImage_thenReturnsSuccess() {
        // GIVEN: Подготавливаем multipart/form-data запрос с изображением
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        body.add("image", new ClassPathResource("test-images/digit-7.png"));
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        String url = "http://localhost:" + port + "/api/v1/predict";
        // WHEN: Выполняем POST-запрос
        ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);
        // THEN: Проверяем результат
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.OK);
        assertThat(response.getBody()).isNotNull();
        // Ожидаем, что модель предскажет цифру 7 для данного изображения.
        assertThat(response.getBody()).contains("\"predictedDigit\":7");
    }

    @Test
    void whenPredictWithNoFile_thenReturnsBadRequest() {
        // GIVEN: Пустой запрос
        MultiValueMap<String, Object> body = new LinkedMultiValueMap<>();
        HttpHeaders headers = new HttpHeaders();
        headers.setContentType(MediaType.MULTIPART_FORM_DATA);
        HttpEntity<MultiValueMap<String, Object>> requestEntity = new HttpEntity<>(body, headers);
        String url = "http://localhost:" + port + "/api/v1/predict";
        // WHEN: Выполняем POST-запрос
        ResponseEntity<String> response = restTemplate.postForEntity(url, requestEntity, String.class);
        // THEN: Проверяем, что сервер вернул ошибку 400
        assertThat(response.getStatusCode()).isEqualTo(HttpStatus.BAD_REQUEST);
        assertThat(response.getBody()).contains("Обязательная часть запроса 'image' отсутствует.");
    }
}

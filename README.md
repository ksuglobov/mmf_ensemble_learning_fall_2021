# Веб-сервис "Ансамбли алгоритмов для решения задачи регрессии"
 MMF practicum project "Ensemble learning on a web server"

Итоговый веб-сервис имеет простой, интуитивно понятный интерфейс:
* Есть режим работы с датасетами: можно добавить обучающую, валидационную и тестовую выборки. Формат всех датасетов: .csv. Во всех датасетах первая колонка — индексы. В качестве целевой переменной указывается либо колонка загруженного датасета, либо загружается таблица с двумя колонками, первая — индекс, вторая — значение.
* Есть режим работы с моделями: можно добавить ансамбли "Случайный лес" и "Градиентный бустинг", настроить их гиперпараметры, обучить модель на трэйне и посмотреть потери на нём и на контроле по графику, а потом скачать предсказание для тестовой выборки.

Ниже прикреплены скриншоты полного цикла работы веб-сервиса:
![demo](img/view.png)

Итак:
1. В папке data лежат исходные данные
2. В папке experiments лежат эксперименты и исходники к ним
3. В папке scripts лежат две папки: папка locally со скриптами для сборки и запуска docker-контейнера у себя на машине (сначала build, потом run) и папка online со скриптом для загрузки образа из dockerhub и запуска (сначала pull, а потом run)
4. В папке src находятся две папки: ensembles с исходниками реализации моделей ансамблей и app с исходниками сервера
5. В папке task находится условие задания
6. Также в папке app находится папка data с данными, на которых можно протестировать веб-сервис
7. Адрес проекта на dockerhub: https://hub.docker.com/repository/docker/ksuglobov/mmf_ensemble_learning_fall_2021 (стандартный тэг latest).

**Способы запуска проекта**:
1. Либо запуск scripts/locally/build.sh, scripts/locally/run.sh
2. Либо запуск scripts/online/pull.sh, scripts/online/run.sh
3. Альтернатива пункту 1: в корне проекта в командной строке: `docker build -t web_ensemble .`, затем `docker run --rm -p 5000:5000 -v-i web_ensemble`
4. Альтернатива пункту 2: в командной строке: `docker pull ksuglobov/mmf_ensemble_learning_fall_2021`, затем `docker run ksuglobov/mmf_ensemble_learning_fall_2021`

Далее требуется по инструкции в командной строке открыть нужный адрес, либо самостоятельно перейти в браузере на http://localhost:5000/

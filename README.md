# GEYE: RTSP видеоаналитика людей

MVP-сервер для:
- приема RTSP потока с IP-камеры,
- детекции и трекинга людей в кадре,
- устойчивой идентификации при выходе/возврате в кадр (ReID-lite),
- web-интерфейса с live видео,
- счетчиков уникальных людей за 1 час и за 24 часа.

## Что реализовано сейчас

- FastAPI веб-приложение.
- Поток обработки RTSP видео через OpenCV.
- Детекция + трекинг людей (`person`) через `ultralytics` (`YOLOv8` track mode).
- ReID-lite слой (appearance embedding), который назначает стабильный `global_id`,
  в том числе когда человек вышел из кадра и вернулся.
- Хранение событий в SQLite.
- API статистики:
  - `online_count` (сейчас в кадре),
  - `unique_last_hour`,
  - `unique_last_24h`,
  - список текущих `global_id`.
- `POST /api/reset` - сброс счетчиков, треков и очистка событий в БД.
- `POST /api/reset-all` - полный сброс: очистка событий/фото/регистрации и сброс счетчика `global_id` к 1.
- `GET /api/gallery?window=online|hour|day` - фото уникальных людей (приоритет на лицо анфас).
- `POST /api/people/register` и `GET /api/people/{global_id}` - ручная регистрация человека по `G`-ID.
  После регистрации формируется face-профиль для ID-lock (если доступно фото лица).

## Важно про "идентификацию каждого"

В текущем MVP идентификация реализована как ReID-lite по внешнему виду, с устойчивым `global_id` при повторном входе в кадр.

Это не биометрическое распознавание лица и не гарантирует 100% точность при похожей одежде/освещении. Для production-уровня обычно подключают отдельную ReID-модель.

## Требования

- Python 3.10+
- ffmpeg в системе (желательно для стабильности RTSP)
- GPU Nvidia опционально (Quadro T400 поддерживается; начните с `yolov8n.pt`)

## Быстрый старт

1. Установка зависимостей:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Для устойчивой идентификации при смене одежды включен эмбеддер лица `InsightFace` (если доступен).
Если `insightface` не установлен/не запустился, сервер автоматически откатится на упрощенный эмбеддер.

2. Настройка окружения:

```bash
cp .env.example .env
```

Заполните `RTSP_URL` под вашу камеру.
Ключевые настройки:
- `ENABLE_LINE_CROSSING=false` - пока отключает логику вход/выход,
- `REID_MATCH_THRESHOLD` - строгость сопоставления при повторном входе,
- `REID_WEAK_MATCH_THRESHOLD`, `REID_WEAK_MARGIN` и `REID_WEAK_MATCH_RECENCY_SEC` - мягкое сопоставление для сценария "вернулся в другой позе/одежде",
- `REID_MAX_ABSENCE_SEC` - как долго хранить профиль человека для повторной идентификации,
- `TRACKER_CONFIG_PATH` - конфиг трекера (рекомендуется ByteTrack с увеличенным `track_buffer`),
- `MIN_PERSON_BOX_HEIGHT_PX` и `MIN_PERSON_BOX_AREA_RATIO` - отсев ложных мелких детекций (например предметов на полу),
- `COUNT_CONFIRM_MIN_HITS` и `COUNT_CONFIRM_MIN_AGE_SEC` - подтверждение нового гостя перед записью в статистику,
- `UNIQUE_REQUIRE_FACE_FOR_COUNT` и `FACE_CONFIRM_MIN_HITS` - учитывать уникального только при подтверждении по лицу,
- `FACE_MIN_SCORE` - минимальное качество face-crop для сохранения фото и учета в уникальных,
- `FACE_EMBEDDER` (`auto|insightface|hist`) и `FACE_EMBEDDER_CTX_ID` - выбор движка эмбеддинга лица и режим GPU/CPU,
- `PHOTO_DIR`, `PHOTO_CAPTURE_ONCE_PER_ID`, `PHOTO_UPDATE_INTERVAL_SEC`, `GALLERY_LIMIT` - параметры фото-галереи,
- `FACE_LOCK_MATCH_THRESHOLD`, `FACE_LOCK_MARGIN` - жесткость ID-lock по зарегистрированным лицам,
- `FACE_REBIND_MATCH_THRESHOLD`, `FACE_REBIND_MARGIN` - авто-возврат к старому G-ID при очень сильном сходстве с ранее сохраненными фото,
- `FACE_PROFILES_REFRESH_SEC` - как часто обновлять кеш фото-профилей для авто-возврата G-ID,
- `FACE_PROFILE_BANK_PER_ID`, `FACE_REBIND_MIN_VOTES` - сравнение по серии фото (несколько лучших кадров на ID),
- `FACE_PREFER_LOCKED_DELTA` - при равных score отдавать приоритет зарегистрированному (locked) ID,
- `FACE_GLOBAL_DEDUP_THRESHOLD`, `FACE_GLOBAL_DEDUP_INTERVAL_SEC` - фоновое авто-слияние дублей G-ID по лицу,
- `FACE_STABLE_SIM_THRESHOLD`, `FACE_STABLE_MIN_HITS` - новый ID подтверждается только после нескольких согласованных face-кадров,
- `FACE_LOCKED_RELAXED_THRESHOLD` - мягкий порог для привязки/слияния к зарегистрированному ID,
- `RTSP_LOW_LATENCY_MODE`, `RTSP_DRAIN_GRABS` - уменьшение накопления задержки RTSP.

Рекомендуемый базовый профиль против ложных уникальных:
- `REID_MATCH_THRESHOLD=0.68`
- `COUNT_CONFIRM_MIN_AGE_SEC=4.0`
- `TRACK_TTL_SEC=15.0`
- Для защиты от "перехвата ID" новым человеком:
- `REID_WEAK_MATCH_THRESHOLD=0.50`
- `REID_WEAK_MARGIN=0.08`
- `REID_WEAK_MATCH_RECENCY_SEC=120`

3. Запуск сервера:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Тихий запуск (только warning/error, без access-логов):

```bash
./run.sh
```

4. Открыть в браузере:

- `http://localhost:8000`
- `http://10.0.0.189:8000`

## Ваши статичные параметры

- IP сервера: `10.0.0.189`
- RTSP поток камеры: `rtsp://admin:admin@10.0.0.242:554/live/main`

## Запуск в терминале (готовые команды)

```bash
cd /Users/gennadijvazmin/Documents/GEYE
source .venv/bin/activate
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

После запуска откройте:
- `http://10.0.0.189:8000`

## Структура

- `app/main.py` - FastAPI endpoints и запуск background обработки.
- `app/video_processor.py` - чтение RTSP, детекция и трекинг.
- `app/analytics.py` - агрегация статистики.
- `app/db.py` - SQLite инициализация.

## Следующие доработки (рекомендуется)

1. Заменить ReID-lite на обученную ReID-модель (OSNet/FastReID) для точности в зале.
2. Экспорт отчетов в CSV/Excel по часам/дням.
3. Роли пользователей и авторизация в веб-интерфейсе.

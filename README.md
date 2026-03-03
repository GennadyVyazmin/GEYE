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

2. Настройка окружения:

```bash
cp .env.example .env
```

Заполните `RTSP_URL` под вашу камеру.
Ключевые настройки:
- `ENABLE_LINE_CROSSING=false` - пока отключает логику вход/выход,
- `REID_MATCH_THRESHOLD` - строгость сопоставления при повторном входе,
- `REID_WEAK_MATCH_THRESHOLD` и `REID_WEAK_MATCH_RECENCY_SEC` - мягкое сопоставление для сценария "вернулся в другой позе/одежде",
- `REID_MAX_ABSENCE_SEC` - как долго хранить профиль человека для повторной идентификации,
- `COUNT_CONFIRM_MIN_HITS` и `COUNT_CONFIRM_MIN_AGE_SEC` - подтверждение нового гостя перед записью в статистику,
- `UNIQUE_REQUIRE_FACE_FOR_COUNT` и `FACE_CONFIRM_MIN_HITS` - учитывать уникального только при подтверждении по лицу,
- `PHOTO_DIR`, `PHOTO_CAPTURE_ONCE_PER_ID`, `PHOTO_UPDATE_INTERVAL_SEC`, `GALLERY_LIMIT` - параметры фото-галереи,
- `RTSP_LOW_LATENCY_MODE`, `RTSP_DRAIN_GRABS` - уменьшение накопления задержки RTSP.

Рекомендуемый базовый профиль против ложных уникальных:
- `REID_MATCH_THRESHOLD=0.68`
- `COUNT_CONFIRM_MIN_AGE_SEC=4.0`
- `TRACK_TTL_SEC=15.0`

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

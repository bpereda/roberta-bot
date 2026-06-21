import asyncio
import csv
import json
import logging
import os
from datetime import datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv
from telegram import Bot

from config import (
    SENSOR_DATASET_PATH,
    SENSOR_LATEST_PATH,
    SENSOR_SERVER_HOST,
    SENSOR_SERVER_PORT,
    SOIL_ALERT_PERCENT,
)

CSV_COLUMNS = [
    "timestamp",
    "soil_raw",
    "soil_percent",
    "air_humidity",
    "temp_c",
    "light_raw",
    "plant_condition",
    "tree_prediction",
    "logistic_prediction",
]
BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"

logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_dataset_path() -> Path:
    path = Path(SENSOR_DATASET_PATH).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def get_latest_sensor_path() -> Path:
    path = Path(SENSOR_LATEST_PATH).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def save_latest_reading(reading: dict) -> Path:
    path = get_latest_sensor_path()
    latest = {column: reading.get(column, "") for column in CSV_COLUMNS}
    latest["timestamp"] = latest["timestamp"] or datetime.now().strftime(
        "%Y-%m-%d %H:%M:%S"
    )

    temporary_path = path.with_suffix(path.suffix + ".tmp")
    temporary_path.write_text(
        json.dumps(latest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temporary_path.replace(path)
    return path


def ensure_csv_columns(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        return

    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        rows = list(reader)
        current_columns = reader.fieldnames or []

    if current_columns == CSV_COLUMNS:
        return

    with path.open("w", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        writer.writeheader()
        for row in rows:
            writer.writerow({column: row.get(column, "") for column in CSV_COLUMNS})


def append_reading_to_csv(reading: dict) -> Path:
    path = get_dataset_path()
    ensure_csv_columns(path)
    row = {column: reading.get(column, "") for column in CSV_COLUMNS}
    row["timestamp"] = row["timestamp"] or datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", encoding="utf-8", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=CSV_COLUMNS)
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    return path


def should_alert(reading: dict) -> bool:
    condition = str(reading.get("plant_condition", "")).lower()
    if condition == "marchita":
        return True

    soil_percent = reading.get("soil_percent")
    if soil_percent in ("", None):
        return False

    return float(soil_percent) <= SOIL_ALERT_PERCENT


def format_telegram_notification(reading: dict, needs_watering: bool) -> str:
    soil_percent = reading.get("soil_percent", "sin dato")
    air_humidity = reading.get("air_humidity", "sin dato")
    temp_c = reading.get("temp_c", "sin dato")
    light_raw = reading.get("light_raw", "sin dato")
    tree_prediction = reading.get("tree_prediction", "sin dato")
    logistic_prediction = reading.get("logistic_prediction", "sin dato")

    if needs_watering:
        introduction = (
            "Bestie, necesito riego: mi sustrato está demasiado seco 💅🌱"
        )
    else:
        introduction = "Reporte de Roberta: sigo bajo control, reina ✨🌱"

    return (
        f"{introduction}\n\n"
        f"Humedad del suelo: {soil_percent}%\n"
        f"Humedad ambiente: {air_humidity}%\n"
        f"Temperatura: {temp_c} °C\n"
        f"Luz: {light_raw}\n"
        f"Árbol de decisión: {tree_prediction}\n"
        f"Regresión logística: {logistic_prediction}"
    )


async def send_telegram_notification(reading: dict, needs_watering: bool) -> bool:
    token = os.getenv("TELEGRAM_BOT_TOKEN")
    chat_id = os.getenv("TELEGRAM_ALERT_CHAT_ID")

    if not token or not chat_id:
        return False

    bot = Bot(token=token)
    message = format_telegram_notification(reading, needs_watering)
    await bot.send_message(chat_id=chat_id, text=message)
    return True


def response_body(payload: dict) -> bytes:
    return json.dumps(payload, ensure_ascii=False).encode("utf-8")


class SensorRequestHandler(BaseHTTPRequestHandler):
    def send_json(self, status_code: int, payload: dict) -> None:
        body = response_body(payload)
        self.send_response(status_code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        path = urlparse(self.path).path

        if path == "/health":
            self.send_json(200, {"status": "ok"})
            return

        if path == "/latest":
            latest_path = get_latest_sensor_path()

            if not latest_path.exists():
                self.send_json(
                    404,
                    {"ok": False, "error": "Todavía no hay una medición actual"},
                )
                return

            latest = json.loads(latest_path.read_text(encoding="utf-8"))
            self.send_json(200, {"ok": True, "reading": latest})
            return

        self.send_json(404, {"ok": False, "error": "Ruta no encontrada"})

    def do_POST(self) -> None:
        path = urlparse(self.path).path

        if path != "/sensor":
            self.send_json(404, {"ok": False, "error": "Ruta no encontrada"})
            return

        try:
            content_length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(content_length)
            reading = json.loads(raw_body.decode("utf-8"))
            latest_path = save_latest_reading(reading)
            saved_path = append_reading_to_csv(reading)
            needs_watering = should_alert(reading)
            notification_sent = asyncio.run(
                send_telegram_notification(reading, needs_watering)
            )

            logger.info("Medición recibida y guardada en %s", saved_path)
            self.send_json(
                200,
                {
                    "ok": True,
                    "saved_to": str(saved_path),
                    "latest_saved_to": str(latest_path),
                    "needs_watering": needs_watering,
                    "notification_sent": notification_sent,
                },
            )
        except Exception as error:
            logger.exception("Error al procesar medición")
            self.send_json(400, {"ok": False, "error": str(error)})


def main() -> None:
    load_dotenv(ENV_PATH)
    server = ThreadingHTTPServer(
        (SENSOR_SERVER_HOST, SENSOR_SERVER_PORT), SensorRequestHandler
    )

    logger.info(
        "Servidor de sensores escuchando en http://%s:%s",
        SENSOR_SERVER_HOST,
        SENSOR_SERVER_PORT,
    )
    server.serve_forever()


if __name__ == "__main__":
    main()

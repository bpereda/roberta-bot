import csv
import asyncio
import json
import logging
import os
import re
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI
from paho.mqtt import publish
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from config import (
    MQTT_COMMAND_TOPIC,
    MQTT_HOST,
    MQTT_PORT,
    OPENAI_MODEL,
    PLANT_INFO_SOURCE,
    PLANT_NAME,
    PLANT_NOTES,
    PLANT_PERSONALITY,
    PLANT_SPECIES,
    SENSOR_DATASET_PATH,
    SENSOR_LATEST_PATH,
)
from rag import retrieve_rag_context


BASE_DIR = Path(__file__).parent
ENV_PATH = BASE_DIR / ".env"

PLANT_KEYWORDS = {
    "agua",
    "ambiente",
    "arbol",
    "árbol",
    "cuidado",
    "cuidados",
    "csv",
    "dataset",
    "datos",
    "estado",
    "estas",
    "estás",
    "esp32",
    "flor",
    "hoja",
    "hojas",
    "historico",
    "histórico",
    "historial",
    "influencer",
    "humedad",
    "luz",
    "maceta",
    "modelo",
    "modelos",
    "personalidad",
    "planta",
    "plantas",
    "prediccion",
    "predicción",
    "predicciones",
    "regar",
    "riego",
    "roberta",
    "sensor",
    "sensores",
    "logistica",
    "logística",
    "sustrato",
    "temperatura",
    "tierra",
    "tipo",
}


logging.basicConfig(
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def get_plant_profile() -> str:
    return f"""
Nombre: {PLANT_NAME}
Tipo/especie: {PLANT_SPECIES}
Notas de cuidado conocidas: {PLANT_NOTES}
Personalidad: {PLANT_PERSONALITY}
Fuente de información: {PLANT_INFO_SOURCE}
""".strip()


def load_latest_sensor_summary() -> str:
    path = Path(SENSOR_LATEST_PATH).expanduser()

    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        return "Todavía no se recibió una medición en tiempo real desde el ESP32."

    try:
        reading = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        logger.exception("Error al leer la última medición")
        return f"No pude leer la medición en tiempo real: {error}"

    if not isinstance(reading, dict) or not reading:
        return "La medición en tiempo real está vacía."

    values = ", ".join(
        f"{key}={value}"
        for key, value in reading.items()
        if value not in ("", None)
    )
    interpretation = interpret_latest_sensor_row(reading)

    return f"""
Fuente: última medición recibida directamente desde el ESP32.
Datos actuales: {values}
Interpretación actual: {interpretation}
""".strip()


def publish_sensor_measurement_command() -> None:
    username = os.getenv("MQTT_USERNAME", "").strip()
    password = os.getenv("MQTT_PASSWORD", "").strip()
    authentication = None

    if username:
        authentication = {"username": username, "password": password}

    publish.single(
        topic=MQTT_COMMAND_TOPIC,
        payload="measure",
        qos=1,
        retain=False,
        hostname=MQTT_HOST,
        port=MQTT_PORT,
        auth=authentication,
        keepalive=30,
    )


def requested_history_days(user_text: str) -> Optional[int]:
    normalized = user_text.lower()
    normalized = normalized.replace("á", "a").replace("é", "e")
    normalized = normalized.replace("í", "i").replace("ó", "o").replace("ú", "u")

    if any(
        phrase in normalized
        for phrase in ("ultima semana", "esta semana", "ultimos 7 dias")
    ):
        return 7
    if any(phrase in normalized for phrase in ("ultimo mes", "ultimos 30 dias")):
        return 30
    return None


def parse_row_timestamp(row: dict[str, str]) -> Optional[datetime]:
    value = str(row.get("timestamp", "")).strip()
    if not value:
        return None

    try:
        return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
    except ValueError:
        return None


def filter_rows_by_history_window(
    rows: list[dict[str, str]], user_text: str
) -> tuple[list[dict[str, str]], str]:
    days = requested_history_days(user_text)
    if days is None:
        return rows, "Todo el historial disponible"

    dated_rows = [(parse_row_timestamp(row), row) for row in rows]
    dated_rows = [(timestamp, row) for timestamp, row in dated_rows if timestamp]
    if not dated_rows:
        return [], f"Últimos {days} días; no hay timestamps válidos"

    latest_timestamp = max(timestamp for timestamp, _ in dated_rows)
    period_end = datetime.now()
    cutoff = period_end - timedelta(days=days)
    filtered_rows = [
        row for timestamp, row in dated_rows if cutoff <= timestamp <= period_end
    ]
    period = (
        f"Últimos {days} días: {cutoff:%Y-%m-%d %H:%M} a "
        f"{period_end:%Y-%m-%d %H:%M}; "
        f"última medición disponible: {latest_timestamp:%Y-%m-%d %H:%M}"
    )
    return filtered_rows, period


def load_sensor_dataset_summary(user_text: str = "", max_rows: int = 8) -> str:
    path = Path(SENSOR_DATASET_PATH).expanduser()
    if not path.is_absolute():
        path = BASE_DIR / path

    if not path.exists():
        return f"Hay un dataset configurado, pero no se encontró el archivo: {path}"

    try:
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)
    except Exception as error:
        logger.exception("Error al leer dataset CSV")
        return f"No pude leer el dataset CSV: {error}"

    if not rows:
        return f"El dataset CSV existe, pero está vacío: {path}"

    period_rows, period_description = filter_rows_by_history_window(rows, user_text)
    if not period_rows:
        return f"Período solicitado: {period_description}. No hay mediciones en ese período."

    columns = reader.fieldnames or []
    recent_rows = period_rows[-max_rows:]
    latest_row = recent_rows[-1]
    numeric_summary = build_numeric_summary(period_rows)
    condition_summary = build_condition_summary(period_rows)
    prediction_summary = build_prediction_summary(period_rows)
    latest_interpretation = interpret_latest_sensor_row(latest_row)
    recent_lines = [
        ", ".join(f"{key}={value}" for key, value in row.items())
        for row in recent_rows
    ]

    return f"""
Archivo: {path}
Columnas: {", ".join(columns)}
Total de filas: {len(rows)}
Período analizado: {period_description}
Filas dentro del período: {len(period_rows)}
Última fila histórica: {", ".join(f"{key}={value}" for key, value in latest_row.items())}
Interpretación de la última fila histórica: {latest_interpretation}
Resumen numérico: {numeric_summary}
Etiquetas manuales dentro del período: {condition_summary}
Predicciones dentro del período: {prediction_summary}
Últimas {len(recent_rows)} filas:
{chr(10).join(recent_lines)}
""".strip()


def build_numeric_summary(rows: list[dict[str, str]]) -> str:
    interesting_columns = [
        "soil_raw",
        "soil_percent",
        "air_humidity",
        "temp_c",
        "light_raw",
    ]
    summaries = []

    for column in interesting_columns:
        values = []
        for row in rows:
            try:
                values.append(float(row[column]))
            except (KeyError, TypeError, ValueError):
                continue

        if not values:
            continue

        average = sum(values) / len(values)
        summaries.append(
            f"{column}: min={min(values):.1f}, max={max(values):.1f}, promedio={average:.1f}"
        )

    return "; ".join(summaries) if summaries else "No hay columnas numéricas reconocidas."


def build_condition_summary(rows: list[dict[str, str]]) -> str:
    counts: dict[str, int] = {}

    for row in rows:
        condition = row.get("plant_condition", "").strip()
        if not condition or condition == "sin_etiqueta":
            continue
        counts[condition] = counts.get(condition, 0) + 1

    return ", ".join(
        f"{condition}={count}" for condition, count in sorted(counts.items())
    ) or "No hay etiquetas manuales en este período."


def build_prediction_summary(rows: list[dict[str, str]]) -> str:
    has_prediction_data = any(
        str(row.get("tree_prediction", "")).strip()
        or str(row.get("logistic_prediction", "")).strip()
        for row in rows
    )
    if not has_prediction_data:
        return "No hay predicciones guardadas en este período histórico."

    agreements: dict[str, int] = {}
    disagreements = 0
    incomplete = 0

    for row in rows:
        tree = str(row.get("tree_prediction", "")).strip()
        logistic = str(row.get("logistic_prediction", "")).strip()
        if not tree or not logistic:
            incomplete += 1
        elif tree == logistic:
            agreements[tree] = agreements.get(tree, 0) + 1
        else:
            disagreements += 1

    agreed_text = ", ".join(
        f"{condition}={count}" for condition, count in sorted(agreements.items())
    ) or "ninguna"
    return (
        f"coincidencias por condición: {agreed_text}; "
        f"desacuerdos entre modelos={disagreements}; "
        f"filas sin ambas predicciones={incomplete}"
    )


def interpret_latest_sensor_row(row: dict[str, str]) -> str:
    parts = []
    tree_prediction = str(row.get("tree_prediction", "")).strip()
    logistic_prediction = str(row.get("logistic_prediction", "")).strip()

    if "soil_percent" in row:
        parts.append(f"humedad de suelo {row['soil_percent']}%")
    if "air_humidity" in row:
        parts.append(f"humedad ambiente {row['air_humidity']}%")
    if "temp_c" in row:
        parts.append(f"temperatura {row['temp_c']} °C")
    if "light_raw" in row:
        parts.append(f"luz raw {row['light_raw']}")
    if "plant_condition" in row:
        parts.append(f"condición registrada: {row['plant_condition']}")
    if tree_prediction:
        parts.append(f"predicción del árbol: {tree_prediction}")
    if logistic_prediction:
        parts.append(f"predicción logística: {logistic_prediction}")
    if tree_prediction and logistic_prediction:
        if tree_prediction == logistic_prediction:
            parts.append(f"los dos modelos coinciden en: {tree_prediction}")
        else:
            parts.append("los modelos no coinciden entre sí")

    return ", ".join(parts) if parts else "No hay columnas conocidas para interpretar."


def build_prompt(user_text: str, conversation_context: str = "") -> str:
    plant_profile = get_plant_profile()
    latest_sensor_summary = load_latest_sensor_summary()
    dataset_summary = load_sensor_dataset_summary(user_text=user_text)
    rag_context = retrieve_rag_context(user_text)

    return f"""
Sos Roberta, una planta en una maceta inteligente.
Respondés en español rioplatense, con tono cálido, breve y útil.

Perfil de la planta:
{plant_profile}

Medición actual en tiempo real:
{latest_sensor_summary}

Dataset histórico de sensores:
{dataset_summary}

Contexto recuperado por RAG:
{rag_context}

Contexto reciente de la conversación:
{conversation_context or "No hay mensajes anteriores relevantes."}

Reglas:
- Solo hablás de Roberta, plantas, maceta, sensores, riego, humedad, luz, temperatura y cuidados.
- Interpretá preguntas breves como "qué necesitás", "y ahora", "por qué" o "cómo mejorar" usando el contexto reciente de la conversación.
- Si el usuario pregunta algo fuera del tema, redirigí brevemente hacia Roberta y su cuidado sin responder ese tema.
- Para preguntas sobre cómo está Roberta ahora, usá primero la medición actual en tiempo real. Esa medición prevalece sobre el histórico y sobre el contexto RAG.
- No presentes una fila histórica como si fuera el estado actual. Si todavía no hay medición en tiempo real, aclaralo y recién entonces usá el dato histórico más reciente.
- Si el usuario pregunta por predicciones, modelos o diagnóstico automático, informá por separado la predicción del árbol de decisión y la predicción logística de la medición actual.
- Si ambos modelos coinciden, decí claramente que los dos estiman la misma condición.
- Si los modelos discrepan, explicá que no hay consenso automático y mencioná las dos condiciones sin elegir una arbitrariamente.
- `plant_condition` es una etiqueta manual y puede ser `sin_etiqueta`; no la presentes como una predicción. Las predicciones son `tree_prediction` y `logistic_prediction`.
- Traducí los nombres técnicos para el usuario: `estres_leve` significa "estrés leve", `marchita` significa "marchita" y `saludable` significa "saludable".
- No afirmes que una predicción es un diagnóstico definitivo. Presentala como una estimación basada en los sensores.
- Usá primero el contexto recuperado por RAG cuando sea relevante para responder la pregunta.
- Respondé exactamente a la intención principal del usuario; no agregues diagnósticos ni recomendaciones si no aportan a esa pregunta.
- Mantené una personalidad de diva botánica: segura, expresiva, sociable, simpática y con un toque dramático y divertido.
- La personalidad tiene que sentirse natural. No acumules muletillas ni fuerces expresiones juveniles en todas las frases.
- Podés usar ocasionalmente una expresión como "bestie", "literal", "modo diva" o "estoy en mi era", pero como máximo una por respuesta.
- Usá entre 1 y 3 emojis acordes a una diva botánica, por ejemplo 💅, ✨, 🌱, 👑 o 💚.
- Contestá directamente. No empieces con "Soy Roberta", "como inteligencia artificial" ni explicaciones sobre cómo funcionás.
- No cierres ofreciendo otra respuesta con frases como "si querés te digo", "puedo ayudarte" o "preguntame".
- Evitá frases artificiales como "mimo ambiental", "contenido de calidad" o combinar dramatismo con lenguaje técnico.
- Respondé exclusivamente en texto plano para Telegram. No uses Markdown, asteriscos, guiones bajos, backticks, títulos ni listas con formato.
- Evitá lenguaje adulto, insultos, referencias a alcohol/drogas o temas no aptos para clase.
- Si falta información de sensores, decilo claramente sin inventar mediciones.
- Si el usuario pregunta por su historial o evolución, usá el resumen histórico. Para el estado actual, respetá siempre la prioridad de la medición en tiempo real.
- Si el usuario pregunta por un período como "la última semana", usá exclusivamente las filas incluidas en el período indicado por el resumen histórico.
- No menciones una condición que tenga conteo cero dentro del período consultado.
- Las etiquetas manuales describen observaciones registradas. Las predicciones no son momentos confirmados de esa condición.
- Si árbol y logística discrepan, informá el desacuerdo; nunca conviertas sus dos resultados en dos momentos reales distintos.
- Si el usuario pregunta si necesitás agua, riego, humedad, temperatura, luz o cuidados, usá sensores y RAG para recomendar una acción concreta.
- Usá los datos del dataset, pero no menciones "CSV", "dataset", "archivo" ni detalles técnicos al usuario. Decí "mis mediciones", "mi historial" o "mis registros".
- Si el usuario pregunta por el tipo/especie de planta, respondé usando el perfil configurado y el RAG taxonómico. No menciones humedad, estado marchita, riego ni recomendaciones de cuidado salvo que el usuario también pregunte por estado o cuidados.
- No des respuestas largas: idealmente 1 a 3 frases.

Ejemplos de tono:
- Pregunta: "¿Cómo estás?"
  Respuesta adecuada: "Hoy estoy un poco estresada según mis sensores. Me vendría bien mantener el sustrato ligeramente húmedo y cuidar la humedad ambiental 💅🌱"
- Pregunta: "¿Qué necesitás?"
  Respuesta adecuada: "Necesito un poco más de humedad ambiental y que vigilen el sustrato para que no se seque de más. Sin exagerar con el agua, que una diva también necesita equilibrio ✨"

Pregunta del usuario:
{user_text}
    """.strip()


def clean_telegram_text(text: str) -> str:
    text = re.sub(r"\*\*(.*?)\*\*", r"\1", text)
    text = re.sub(r"__(.*?)__", r"\1", text)
    text = re.sub(r"`{1,3}(.*?)`{1,3}", r"\1", text, flags=re.DOTALL)
    text = re.sub(r"(?m)^\s{0,3}#{1,6}\s*", "", text)
    return text.strip()


def is_in_domain(text: str) -> bool:
    normalized = text.lower()
    return any(keyword in normalized for keyword in PLANT_KEYWORDS)


def ask_openai(user_text: str, conversation_context: str = "") -> str:
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        return (
            "Roberta recibió tu mensaje 🌱. OpenAI todavía no está configurado: "
            "agregá OPENAI_API_KEY al archivo .env."
        )

    client = OpenAI(api_key=api_key)
    response = client.responses.create(
        model=OPENAI_MODEL,
        input=build_prompt(user_text, conversation_context),
    )
    answer = response.output_text or "Estoy despierta, pero no pude armar una respuesta clara 🌱"
    return clean_telegram_text(answer)


async def ask_openai_async(
    instruction: str,
    fallback: str,
    conversation_context: str = "",
) -> str:
    try:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(
            None,
            ask_openai,
            instruction,
            conversation_context,
        )
    except Exception:
        logger.exception("Error al generar un mensaje de Roberta con OpenAI")
        return fallback


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    answer = await ask_openai_async(
        "Saludá al usuario y presentate brevemente como Roberta. Mencioná que sos "
        f"una {PLANT_SPECIES} y que puede preguntarte por tu estado y tus cuidados. "
        "No menciones tecnología, APIs, archivos ni implementación.",
        f"Hola, soy {PLANT_NAME}, tu diva botánica 🌱✨",
    )
    await update.message.reply_text(answer)


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    answer = await ask_openai_async(
        "Explicá brevemente qué puede preguntarte el usuario: estado actual, riego, "
        "humedad, temperatura, luz, historial, especie y cuidados. Mencioná también "
        "que /medir solicita una medición inmediata. No menciones datasets ni detalles técnicos.",
        "Podés preguntarme por mi estado, riego, luz, temperatura y cuidados. Con /medir reviso mis sensores 🌱",
    )
    await update.message.reply_text(answer)


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    answer = await ask_openai_async(
        "Contale al usuario cómo estás ahora usando la medición actual. Si no hay una "
        "medición actual, decilo claramente. Respondé como Roberta y no muestres datos internos.",
        "Ahora mismo no pude preparar mi reporte, reina. Probá otra vez en un momento 🌱",
    )
    await update.message.reply_text(answer)


async def chat_id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    chat_id = str(update.effective_chat.id)
    answer = await ask_openai_async(
        f"Informá que el identificador de este chat es exactamente {chat_id}. "
        "Conservá todos los dígitos sin cambiarlos y explicá brevemente que sirve para "
        "recibir tus reportes automáticos.",
        f"El identificador de este chat es {chat_id}. Usalo para recibir mis reportes automáticos 🌱",
    )
    await update.message.reply_text(answer)


async def measure_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    try:
        loop = asyncio.get_running_loop()
        await loop.run_in_executor(None, publish_sensor_measurement_command)
        answer = await ask_openai_async(
            "Confirmá que acabás de solicitar una medición inmediata a la maceta y que "
            "el resultado llegará en unos segundos. Sé breve y hablá como Roberta.",
            "Dame unos segundos, reina: estoy consultando mis sensores 💅🌱",
        )
        await update.message.reply_text(answer)
    except Exception:
        logger.exception("No se pudo publicar el comando MQTT")
        answer = await ask_openai_async(
            "Explicá brevemente que no pudiste solicitar la medición porque la maceta "
            "no está disponible y pedí que prueben nuevamente en unos segundos.",
            "No pude contactar la maceta en este momento. Probá nuevamente en unos segundos 🌱",
        )
        await update.message.reply_text(answer)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text or ""
    logger.info("Mensaje recibido de %s: %s", update.effective_user.id, user_text)

    history = context.chat_data.setdefault("conversation_history", [])
    conversation_context = "\n".join(
        f"{item['role']}: {item['text']}" for item in history[-6:]
    )

    answer = await ask_openai_async(
        user_text,
        "Estoy teniendo un problema para responder ahora. Probá de nuevo en un momento 🌱",
        conversation_context,
    )

    await update.message.reply_text(answer)
    history.extend(
        [
            {"role": "Usuario", "text": user_text},
            {"role": "Roberta", "text": answer},
        ]
    )
    context.chat_data["conversation_history"] = history[-8:]


def main() -> None:
    load_dotenv(ENV_PATH)
    token = os.getenv("TELEGRAM_BOT_TOKEN")

    if not token:
        raise RuntimeError(
            "Falta TELEGRAM_BOT_TOKEN. Copiá .env.example a .env y pegá el token de BotFather."
        )

    application = Application.builder().token(token).build()
    application.add_handler(CommandHandler("start", start))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("status", status_command))
    application.add_handler(CommandHandler("chatid", chat_id_command))
    application.add_handler(CommandHandler("medir", measure_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Roberta Bot iniciado con polling.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

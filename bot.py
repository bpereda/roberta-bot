import csv
import json
import logging
import os
import re
from pathlib import Path

from dotenv import load_dotenv
from openai import OpenAI
from telegram import Update
from telegram.ext import Application, CommandHandler, ContextTypes, MessageHandler, filters

from rag import build_rag_result, format_rag_debug, retrieve_rag_context, write_rag_log


DEFAULT_OPENAI_MODEL = "gpt-5.5"
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
    plant_name = os.getenv("PLANT_NAME", "Roberta")
    plant_species = os.getenv("PLANT_SPECIES", "especie no configurada")
    plant_notes = os.getenv("PLANT_NOTES", "Sin notas adicionales.")
    plant_personality = os.getenv("PLANT_PERSONALITY", "Amable, clara y cuidadosa.")
    plant_source = os.getenv("PLANT_INFO_SOURCE", "Sin fuente configurada.")

    return f"""
Nombre: {plant_name}
Tipo/especie: {plant_species}
Notas de cuidado conocidas: {plant_notes}
Personalidad: {plant_personality}
Fuente de información: {plant_source}
""".strip()


def load_latest_sensor_summary() -> str:
    latest_path = os.getenv("SENSOR_LATEST_PATH", "data/latest_sensor.json")
    path = Path(latest_path).expanduser()

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


def load_sensor_dataset_summary(max_rows: int = 8) -> str:
    dataset_path = os.getenv("SENSOR_DATASET_PATH")

    if not dataset_path:
        return "No hay dataset CSV configurado todavía."

    path = Path(dataset_path).expanduser()
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

    columns = reader.fieldnames or []
    recent_rows = rows[-max_rows:]
    latest_row = recent_rows[-1]
    numeric_summary = build_numeric_summary(rows)
    condition_summary = build_condition_summary(rows)
    latest_interpretation = interpret_latest_sensor_row(latest_row)
    recent_lines = [
        ", ".join(f"{key}={value}" for key, value in row.items())
        for row in recent_rows
    ]

    return f"""
Archivo: {path}
Columnas: {", ".join(columns)}
Total de filas: {len(rows)}
Última fila histórica: {", ".join(f"{key}={value}" for key, value in latest_row.items())}
Interpretación de la última fila histórica: {latest_interpretation}
Resumen numérico: {numeric_summary}
Distribución de condición de planta: {condition_summary}
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
        condition = row.get("plant_condition", "").strip() or "sin dato"
        counts[condition] = counts.get(condition, 0) + 1

    return ", ".join(
        f"{condition}={count}" for condition, count in sorted(counts.items())
    ) or "No hay columna plant_condition."


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
    dataset_summary = load_sensor_dataset_summary()
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
        model=os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL),
        input=build_prompt(user_text, conversation_context),
    )
    answer = response.output_text or "Estoy despierta, pero no pude armar una respuesta clara 🌱"
    return clean_telegram_text(answer)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    plant_name = os.getenv("PLANT_NAME", "Roberta")
    plant_species = os.getenv("PLANT_SPECIES", "especie no configurada")
    await update.message.reply_text(
        f"Hola, soy {plant_name} 🌱. Tipo de planta: {plant_species}. "
        "Ya puedo responder con OpenAI usando mi perfil y, si está configurado, mi CSV de sensores."
    )


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "Podés preguntarme por mi tipo de planta, maceta, riego, humedad, temperatura, luz, sensores o dataset."
    )


async def status_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Perfil de planta:\n{get_plant_profile()}\n\nDataset:\n{load_sensor_dataset_summary(max_rows=3)}"
    )


async def chat_id_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        f"Este chat_id es: {update.effective_chat.id}\n"
        "Usalo como TELEGRAM_ALERT_CHAT_ID si querés recibir alertas automáticas."
    )


async def debug_csv_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(load_sensor_dataset_summary(max_rows=1))


async def debug_latest_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(load_latest_sensor_summary())


async def debug_rag_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = " ".join(context.args) if context.args else "cuidados riego humedad luz temperatura"
    rag_result = build_rag_result(query)
    write_rag_log(rag_result)
    await update.message.reply_text(format_rag_debug(rag_result))


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user_text = update.message.text or ""
    logger.info("Mensaje recibido de %s: %s", update.effective_user.id, user_text)

    history = context.chat_data.setdefault("conversation_history", [])
    conversation_context = "\n".join(
        f"{item['role']}: {item['text']}" for item in history[-6:]
    )

    try:
        answer = ask_openai(user_text, conversation_context)
    except Exception:
        logger.exception("Error al consultar OpenAI")
        answer = (
            "Estoy teniendo un problema para consultar OpenAI 🌱. "
            "Probá de nuevo en un momento."
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
    application.add_handler(CommandHandler("debugcsv", debug_csv_command))
    application.add_handler(CommandHandler("debuglatest", debug_latest_command))
    application.add_handler(CommandHandler("debugrag", debug_rag_command))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    logger.info("Roberta Bot iniciado con polling.")
    application.run_polling(allowed_updates=Update.ALL_TYPES)


if __name__ == "__main__":
    main()

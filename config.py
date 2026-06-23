PLANT_NAME = "Roberta"
PLANT_SPECIES = "Aphelandra squarrosa"
PLANT_NOTES = (
    "Planta de interior conocida como afelandra o planta cebra. Prefiere mucha "
    "luz indirecta, temperatura templada, humedad ambiental alta y sustrato "
    "ligeramente húmedo. Hay que evitar tanto el exceso de agua como la sequedad."
)
PLANT_PERSONALITY = (
    "Roberta es una diva botánica y medio influencer: sociable, segura, divertida "
    "y con un toque dramático, sin dejar de dar consejos útiles sobre su cuidado."
)
PLANT_INFO_SOURCE = (
    "knowledge/afelandra_cuidados.md y "
    "knowledge/afelandra_taxonomia_fuentes_tecnicas.md"
)

OPENAI_MODEL = "gpt-5.4-mini"

SENSOR_DATASET_PATH = "data/final_dataset.csv"
SENSOR_LATEST_PATH = "data/latest_sensor.json"
SOIL_ALERT_PERCENT = 10.0
SENSOR_SERVER_HOST = "0.0.0.0"
SENSOR_SERVER_PORT = 8000

RAG_ENABLED = True
RAG_KNOWLEDGE_DIR = "knowledge"
RAG_MAX_CHUNKS = 6
RAG_LOG_ENABLED = True
RAG_LOG_PATH = "logs/rag.log"

MQTT_HOST = "broker.hivemq.com"
MQTT_PORT = 1883
MQTT_COMMAND_TOPIC = "roberta-belupereda/commands"

# Roberta Bot - Pasos 1, 2 y 3

Backend mínimo para un bot de Telegram usando polling. Recibe mensajes por Telegram y responde con OpenAI usando el perfil de la planta y, si existe, un CSV histórico de sensores.

## 1. Crear el bot en Telegram

1. Abrí Telegram y hablale a `@BotFather`.
2. Mandá `/newbot`.
3. Elegí nombre y username.
4. Copiá el token que te da BotFather.

## 2. Preparar el proyecto

```bash
cd /Users/belupereda/Documents/Codex/2026-06-11/files-mentioned-by-the-user-copia/outputs/roberta_bot_step1
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

Editá `.env` y pegá tu token de Telegram, tu API key de OpenAI y los datos de la planta:

```env
TELEGRAM_BOT_TOKEN=123456:ABC...
OPENAI_API_KEY=sk-...
OPENAI_MODEL=gpt-5.5
PLANT_NAME=Roberta
PLANT_SPECIES=Aphelandra squarrosa
PLANT_NOTES=Planta de interior conocida como afelandra o planta cebra. Prefiere mucha luz pero sin sol directo, temperatura templada, humedad ambiental alta y sustrato ligeramente húmedo. Evitar tanto el exceso de agua como la sequedad: los desajustes de riego pueden afectar las hojas. Es sensible al frío prolongado.
PLANT_PERSONALITY=Roberta es una diva botánica y medio influencer: le gusta estar en todas, desde juntadas tranqui con amigos hasta fiestas, pero siempre en modo PG para clase. Habla con humor, seguridad y un toque dramático, sin dejar de dar consejos útiles sobre su cuidado.
PLANT_INFO_SOURCE=https://www.verdeesvida.es/fichas_de_plantas/plantas-de-interior_4/afelandra--planta-cebra_3182
SENSOR_DATASET_PATH=data/final_dataset.csv
SENSOR_LATEST_PATH=data/latest_sensor.json
RAG_ENABLED=true
RAG_KNOWLEDGE_DIR=knowledge
RAG_MAX_CHUNKS=6
RAG_LOG_ENABLED=true
RAG_LOG_PATH=logs/rag.log
TELEGRAM_ALERT_CHAT_ID=
SOIL_ALERT_PERCENT=25
```

`SENSOR_DATASET_PATH` puede ser una ruta relativa al proyecto o una ruta absoluta.
`RAG_KNOWLEDGE_DIR` apunta a la carpeta con documentos `.md` o `.txt` sobre la especie de la planta.

## 3. Agregar dataset CSV

El dataset actual está copiado en `data/final_dataset.csv`.

Si necesitás reemplazarlo, guardá el nuevo CSV en la carpeta `data`:

```bash
mkdir -p data
```

Formato esperado del CSV actual:

```csv
timestamp,soil_raw,soil_percent,air_humidity,temp_c,light_raw,plant_condition
2026-04-18 16:46:57,4021,2,87.0,23.0,534,saludable
2026-05-26 19:05:05,3150,26,38.0,22.0,2791,marchita
```

El bot interpreta especialmente:

- `soil_percent`: humedad del suelo en porcentaje.
- `air_humidity`: humedad ambiente.
- `temp_c`: temperatura en grados Celsius.
- `light_raw`: lectura cruda de luz.
- `plant_condition`: condición registrada, por ejemplo `saludable` o `marchita`.

También puede leer otras columnas, pero esas son las que resume mejor.

## 4. Ejecutar

En una terminal corré el bot de Telegram:

```bash
python bot.py
```

Después abrí tu bot en Telegram y mandale `/start`.

En otra terminal corré el servidor para el ESP32:

```bash
python sensor_server.py
```

Probá que esté vivo entrando a:

```text
http://localhost:8000/health
```

## 5. Conectar el ESP32

El ESP32 tiene que mandar un `POST` a:

```text
http://IP_DE_TU_COMPUTADORA:8000/sensor
```

Tu computadora y el ESP32 tienen que estar en la misma red WiFi. En macOS podés ver tu IP local en Configuración del Sistema > Wi-Fi > Detalles, o con:

```bash
ipconfig getifaddr en0
```

Ejemplo de JSON que espera el backend:

```json
{
  "soil_raw": 3150,
  "soil_percent": 26,
  "air_humidity": 38.0,
  "temp_c": 22.0,
  "light_raw": 2791,
  "plant_condition": "marchita"
}
```

Ejemplo básico para Arduino/ESP32:

```cpp
#include <WiFi.h>
#include <HTTPClient.h>

const char* ssid = "TU_WIFI";
const char* password = "TU_PASSWORD";
const char* serverUrl = "http://IP_DE_TU_COMPUTADORA:8000/sensor";

void setup() {
  Serial.begin(115200);
  WiFi.begin(ssid, password);

  while (WiFi.status() != WL_CONNECTED) {
    delay(500);
    Serial.print(".");
  }

  Serial.println("WiFi conectado");
}

void loop() {
  if (WiFi.status() == WL_CONNECTED) {
    HTTPClient http;
    http.begin(serverUrl);
    http.addHeader("Content-Type", "application/json");

    int soilRaw = 3150;
    int soilPercent = 26;
    float airHumidity = 38.0;
    float tempC = 22.0;
    int lightRaw = 2791;
    String plantCondition = "marchita";

    String body = "{";
    body += "\"soil_raw\":" + String(soilRaw) + ",";
    body += "\"soil_percent\":" + String(soilPercent) + ",";
    body += "\"air_humidity\":" + String(airHumidity) + ",";
    body += "\"temp_c\":" + String(tempC) + ",";
    body += "\"light_raw\":" + String(lightRaw) + ",";
    body += "\"plant_condition\":\"" + plantCondition + "\"";
    body += "}";

    int statusCode = http.POST(body);
    Serial.println(statusCode);
    Serial.println(http.getString());
    http.end();
  }

  delay(30 * 60 * 1000);
}
```

Para recibir las mediciones y alertas automáticas por Telegram:

1. Mandale `/chatid` al bot.
2. Copiá el número que responde.
3. Pegalo en `.env` como `TELEGRAM_ALERT_CHAT_ID`.
4. Reiniciá `sensor_server.py`.

Cada medición que recibe el backend se envía al chat configurado. Las mediciones automáticas del ESP32 llegan cada 30 minutos y las capturas manuales llegan inmediatamente sin modificar ese temporizador. Si `plant_condition` es `marchita` o `soil_percent` está por debajo de `SOIL_ALERT_PERCENT`, el mensaje se presenta como una solicitud de riego.

## Qué hace ahora

- Usa polling, más simple para empezar.
- Responde a `/start` y `/help`.
- Responde a `/status` mostrando perfil de planta y resumen del CSV.
- Responde a `/chatid` para configurar alertas automáticas.
- Responde a `/debugcsv` mostrando exactamente qué CSV está leyendo.
- Responde a `/debugrag consulta` mostrando los fragmentos recuperados por RAG para esa consulta.
- Para preguntas sobre Roberta, plantas, maceta, sensores o cuidados, llama a OpenAI.
- Agrega al prompt el tipo/especie de planta desde `PLANT_SPECIES`.
- Agrega la personalidad desde `PLANT_PERSONALITY`.
- Agrega al prompt las últimas mediciones del CSV definido en `SENSOR_DATASET_PATH`, pero Roberta las presenta como "mi historial", "mis mediciones" o "mis registros".
- Recupera fragmentos relevantes desde `knowledge/` y desde el histórico de sensores antes de llamar a OpenAI.
- Recibe mediciones del ESP32 por `POST /sensor` y las agrega al CSV.
- Si el mensaje parece fuera del dominio de plantas/maceta/sensores/cuidados, responde con una restricción simple.
- Si falta `OPENAI_API_KEY`, responde con un aviso de configuración.

## RAG implementado

El archivo `rag.py` construye chunks locales a partir de:

- documentos `.md` o `.txt` en `knowledge/`;
- filas recientes del CSV configurado en `SENSOR_DATASET_PATH`.

Para cada pregunta, el bot tokeniza la consulta, expande términos del dominio con sinónimos controlados, detecta la intención principal de la pregunta y puntúa los chunks con un ranking TF-IDF liviano. Además aplica boosts según intención: prioriza sensores para preguntas de estado/riego y prioriza fuentes taxonómicas para preguntas de especie/origen. En los chunks de sensores también aplica un pequeño boost por recencia. Los fragmentos más relevantes se agregan al prompt bajo la sección `Contexto recuperado por RAG`.

Cada recuperación queda registrada en `logs/rag.log` en formato JSON Lines. Cada línea incluye la consulta, intención detectada, tokens expandidos, cantidad de chunks evaluados, chunks seleccionados, puntaje, detalle del score, tokens coincidentes y un preview del texto recuperado.

Los documentos iniciales de conocimiento son:

- `knowledge/afelandra_cuidados.md`: información procesada sobre luz, riego, humedad, temperatura y problemas frecuentes.
- `knowledge/afelandra_taxonomia_fuentes_tecnicas.md`: respaldo taxonómico a partir de fuentes institucionales como Kew/POWO, GBIF e IPNI.

Para probar la recuperación sin llamar a OpenAI:

```bash
python bot.py
```

Y en Telegram:

```text
/debugrag que cuidados requiere esta especie
/debugrag necesita riego hoy
```

Ejemplo de log RAG:

```json
{"query":"necesita riego hoy","intent":"sensor","total_chunks":97,"selected_chunks":[{"source":"afelandra_cuidados.md#9","kind":"knowledge","score":0.9132,"score_details":{"tfidf":0.9132,"intent_boost":1.0,"recency_boost":1.0}}]}
```

## Próximo paso

Conectar las predicciones del árbol/logística del ESP32 al backend del bot y agregarlas al contexto de Roberta.

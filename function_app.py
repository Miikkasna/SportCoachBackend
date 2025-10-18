from datetime import timedelta
import azure.functions as func
from azure.storage.blob import BlobServiceClient
import logging
from openai import AzureOpenAI
from secret import LLM_BASE, LLM_API_KEY, OURA_TOKEN, TELEGRAM_BOT_TOKEN, TELEGRAM_WEBHOOK_SECRET, BLOB_CONNECTION_STRING, TELEGRAM_ID
import requests
from datetime import date
import json


endpoint = LLM_BASE
model_name = "gpt-4o"
deployment = "gpt-4o"

subscription_key = LLM_API_KEY
api_version = "2024-12-01-preview"

gptclient = AzureOpenAI(
    api_version=api_version,
    azure_endpoint=endpoint,
    api_key=subscription_key,
)

blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
container_client = blob_service_client.get_container_client("tmp")

app = func.FunctionApp(http_auth_level=func.AuthLevel.ANONYMOUS)

def get_history(n=5):
    blob_name = "history.json"
    blob_client = container_client.get_blob_client(blob_name)
    try:
        history_json = blob_client.download_blob().readall()
        history = json.loads(history_json)
        return history[-n:]  # return last n messages
    except Exception as e:
        logging.error(f"Error retrieving history: {e}")
        return []

def save_history(history):
    blob_name = "history.json"
    blob_client = container_client.get_blob_client(blob_name)
    history_json = json.dumps(history)
    blob_client.upload_blob(history_json, overwrite=True)

def ask_gpt(sysprompt, history, user_turn):
    # history: list[{"role":"user"/"assistant","content": str}]
    messages = [{"role": "system", "content": sysprompt}]
    messages.extend(history)
    messages.append({"role": "user", "content": user_turn})

    response = gptclient.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=1600,
        top_p=0.95,
        frequency_penalty=0.0,  # 0.9 is usually too high; harms lists/enumerations
        presence_penalty=0.0
    )
    return response.choices[0].message.content


def sport_coach_answer(user_message):
    # OURA authentication
    headers = {
        "Authorization": f"Bearer {OURA_TOKEN}"
    }

    # 1. get personal info
    url = "https://api.ouraring.com/v2/usercollection/personal_info"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        personal_info = response.text
    else:
        personal_info = "No personal info available."

    # 2. get training data for the last 14 days
    end_date = date.today()
    start_date = end_date - timedelta(days=14)

    # Format as YYYY-MM-DD
    url = f"https://api.ouraring.com/v2/usercollection/workout?start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        training_data = response.text
    else:
        training_data = "No training data available."

    # 3. get HRV data for the last 14 days
    end_date = date.today()
    start_date = end_date - timedelta(days=14)
    url = f'https://api.ouraring.com/v2/usercollection/sleep?start_date={start_date}&end_date={end_date}'
    response = requests.get(url, headers=headers)
    if response.status_code == 200:
        data = response.json()['data']
        # just take hrv from the json
        hrv_data = [{ "date": item["bedtime_end"], "hrv": item["average_hrv"] } for item in data]
        hrv_json = json.dumps(hrv_data)
    else:
        hrv_json = "No HRV data available."

    
    # construct context string
    sysprompt = f"You are a professional sport coach. Answer the user questions. Here is the context:\nPersonal Info: {personal_info}\nTraining Data: {training_data}\nHRV Data: {hrv_json}"
    history = get_history()
    answer = ask_gpt(sysprompt, history, user_message)

    # update huistory
    history.append({"role": "user", "content": user_message})
    history.append({"role": "assistant", "content": answer})
    save_history(history)


    # return as plain text
    return answer

@app.route(route="telegram/webhook", methods=["POST"], auth_level=func.AuthLevel.ANONYMOUS)
def telegram_webhook(req: func.HttpRequest) -> func.HttpResponse:
    BOT_TOKEN = TELEGRAM_BOT_TOKEN
    TELEGRAM_API = f"https://api.telegram.org/bot{BOT_TOKEN}"

    try:
        update = req.get_json()
    except ValueError:
        return func.HttpResponse("bad request", status_code=400)
    
    message = update.get("message") or update.get("edited_message")
    if not message:  # ignore non-message updates (callbacks, joins, etc.)
        return func.HttpResponse("ok")

    chat_id = message["chat"]["id"]
    msg_id = message.get("message_id")
    text = message.get("text") or ""

    # Security
    if req.headers.get("X-Telegram-Bot-Api-Secret-Token") != TELEGRAM_WEBHOOK_SECRET or str(chat_id) != TELEGRAM_ID:
        return func.HttpResponse("forbidden", status_code=403)


    # Quick command handling
    if text.startswith("/start"):
        _send_message(TELEGRAM_API, chat_id,
                      "Hi!")
        return func.HttpResponse("ok")

    if not text:
        _send_message(TELEGRAM_API, chat_id, "I currently handle text messages only.")
        return func.HttpResponse("ok")

    # Optional UX: show typing
    _send_chat_action(TELEGRAM_API, chat_id, "typing")

    try:
        answer = sport_coach_answer(text)
    except Exception as e:
        logging.exception("LLM call failed")
        _send_message(TELEGRAM_API, chat_id, "Sorry, I couldn’t reach the model right now.")
        return func.HttpResponse("ok")

    # Return the answer (replying to the original message is nice but optional)
    try:
        _send_message(TELEGRAM_API, chat_id, answer, reply_to=msg_id)
    except Exception:
        logging.exception("sendMessage failed")

    return func.HttpResponse("ok")

def _send_message(api_base: str, chat_id: int, text: str, reply_to: int | None = None):
    payload = {"chat_id": chat_id, "text": text, "parse_mode": "Markdown"}
    if reply_to:
        payload["reply_to_message_id"] = reply_to
    r = requests.post(f"{api_base}/sendMessage", json=payload, timeout=20)
    r.raise_for_status()

def _send_chat_action(api_base: str, chat_id: int, action: str):
    try:
        requests.post(f"{api_base}/sendChatAction",
                      json={"chat_id": chat_id, "action": action}, timeout=10)
    except Exception:
        pass
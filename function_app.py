from datetime import timedelta, datetime
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

def get_history(n=1000):
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

def get_user_instructions():
    blob_name = "user_instructions.txt"
    blob_client = container_client.get_blob_client(blob_name)
    try:
        instructions = blob_client.download_blob().readall().decode("utf-8")
        return instructions
    except Exception as e:
        logging.error(f"Error retrieving user instructions: {e}")
        return ""

def save_user_instructions(instructions):
    blob_name = "user_instructions.txt"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(instructions, overwrite=True)

def get_silent_notes():
    blob_name = "silent_notes.txt"
    blob_client = container_client.get_blob_client(blob_name)
    try:
        notes = blob_client.download_blob().readall().decode("utf-8")
        return notes
    except Exception as e:
        logging.error(f"Error retrieving silent notes: {e}")
        return "No notes yet."

def save_silent_notes(notes):
    blob_name = "silent_notes.txt"
    blob_client = container_client.get_blob_client(blob_name)
    blob_client.upload_blob(notes, overwrite=True)

def ask_gpt(sysprompt, user_turn, history=None):
    # history: list[{"role":"user"/"assistant","content": str}]
    messages = [{"role": "system", "content": sysprompt}]
    if history:
        messages.extend(history)
    messages.append({"role": "user", "content": user_turn})

    response = gptclient.chat.completions.create(
        model=model_name,
        messages=messages,
        temperature=0.7,
        max_tokens=3000,
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

    # -- get personal info (age, weight, height, gender)
    url = "https://api.ouraring.com/v2/usercollection/personal_info"
    response = requests.get(url, headers=headers)
    try:
        personal_info = response.json()
    except Exception as e:
        print(f"Error fetching personal info: {e}")
        personal_info = "No personal info available."

    # -- get baselines
    end_date = date.today()
    start_date = end_date - timedelta(days=60)
    url = f'https://api.ouraring.com/v2/usercollection/sleep?start_date={start_date}&end_date={end_date}'
    response = requests.get(url, headers=headers)
    try:
        data = response.json()['data']
        hrv_data = [item["average_hrv"] for item in data if item["average_hrv"] is not None and item["average_hrv"] > 5]
        hr_data = [item["average_heart_rate"] for item in data if item["average_heart_rate"] is not None and item["average_heart_rate"] > 30]
        total_sleep = [item["total_sleep_duration"] for item in data if item["total_sleep_duration"] is not None and item["total_sleep_duration"] > 1000]
        deep_sleep = [item["deep_sleep_duration"] for item in data if item["deep_sleep_duration"] is not None and item["deep_sleep_duration"] > 100]
        avg_hrv = sum(hrv_data) / len(hrv_data)
        avg_hr = sum(hr_data) / len(hr_data)
        personal_info['Average Sleeping HRV (60 days)'] = avg_hrv
        personal_info['Average Sleeping HR (60 days)'] = avg_hr
        personal_info['Average Total Sleep (60 days)'] = sum(total_sleep)/3600 / len(total_sleep)
        personal_info['Average Deep Sleep (60 days)'] = sum(deep_sleep)/3600 / len(deep_sleep)
    except Exception as e:
        print(f"Error fetching baselines: {e}")

    # get aggregated counts for every unique activity past 365 days
    end_date = date.today()
    start_date = end_date - timedelta(days=365)
    url = f'https://api.ouraring.com/v2/usercollection/workout?start_date={start_date}&end_date={end_date}'
    response = requests.get(url, headers=headers)
    activities = {'Activity counts': {}}
    try:
        activity_data = response.json()['data']
        for item in activity_data:
            activity_type = item.get("activity")
            if activity_type:
                activities['Activity counts'][activity_type] = activities['Activity counts'].get(activity_type, 0) + 1
        # change counts to persentages
        total_activities = sum(activities['Activity counts'].values())
        for activity in activities['Activity counts']:
            activities['Activity counts'][activity] = round((activities['Activity counts'][activity] / total_activities) * 100, 2)
        # sort by percentage descending and to str %
        activities['Activity counts'] = {k: f"{v}%" for k, v in sorted(activities['Activity counts'].items(), key=lambda item: item[1], reverse=True)}
        # cut to top 5 activities
        activities['Activity counts'] = dict(list(activities['Activity counts'].items())[:5])
        # add to personal info
        personal_info['Past year training splits'] = activities['Activity counts']
    except Exception as e:
        print(f"Error fetching activity data: {e}")


    # initialize daily stats for 14 days
    daily_stats = {}
    for i in range(15):
        daily_stats[str(i) + " days ago"] = {"Morning data": {}, "Training data": []}

    # -- get morning data
    end_date = date.today()
    start_date = end_date - timedelta(days=14)
    url = f'https://api.ouraring.com/v2/usercollection/sleep?start_date={start_date}&end_date={end_date}'
    response = requests.get(url, headers=headers)
    try:
        data = response.json()['data']
        for item in data:
            if item.get("total_sleep_duration") < 1500:  # skip naps
                continue
            day_diff = (end_date - date.fromisoformat(item["bedtime_end"][:10])).days
            daily_stats[str(day_diff) + " days ago"]['Morning data']['Total sleep'] = item.get("total_sleep_duration")/3600  # in hours
            daily_stats[str(day_diff) + " days ago"]['Morning data']['HRV'] = item.get("average_hrv")
            daily_stats[str(day_diff) + " days ago"]['Morning data']['HR'] = item.get("average_heart_rate")
    except Exception as e:
        print(f"Error fetching morning data: {e}")

    # -- get training data
    end_date = date.today()
    start_date = end_date - timedelta(days=14)

    url = f"https://api.ouraring.com/v2/usercollection/workout?start_date={start_date}&end_date={end_date}"
    response = requests.get(url, headers=headers)
    try:
        training_data = response.json()['data']
        for item in training_data:
            day_diff = (end_date - date.fromisoformat(item["start_datetime"][:10])).days

            training_data = {}
            training_data['Activity'] = item.get("activity")
            #training_data['Calories burned'] = item.get("calories")
            training_data['Intensity'] = item.get("intensity")
            # duration from start and end datetime
            start_dt = item.get("start_datetime")
            end_dt = item.get("end_datetime")
            if start_dt and end_dt:
                start_dt = datetime.fromisoformat(start_dt[:19])
                end_dt = datetime.fromisoformat(end_dt[:19])
                duration = (end_dt - start_dt).total_seconds() / 3600  # in hours
                training_data['Duration (h)'] = duration

            # check if training data exists for that day (training data array length > 0 for that day)
            if training_data:
                daily_stats[str(day_diff) + " days ago"]['Training data'].append(training_data)
                
    except Exception as e:
        print(f"Error fetching training data: {e}")

    # go through jsons and round floats to 2 decimals
    def round_floats(obj):
        if isinstance(obj, dict):
            return {k: round_floats(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [round_floats(elem) for elem in obj]
        elif isinstance(obj, float):
            return round(obj, 2)
        else:
            return obj
    personal_info = round_floats(personal_info)
    daily_stats = round_floats(daily_stats)
    # change "0 days ago" to "Today"
    daily_stats = {k.replace("0 days ago", "Today"): v for k, v in daily_stats.items()}
    # delete 14 days ago
    if "14 days ago" in daily_stats:
        del daily_stats["14 days ago"]

    user_instructions = get_user_instructions()

    silent_notes = get_silent_notes()

    # construct context
    context = f"Personal info and baselines: {personal_info}\n\n"
    context += f"Daily stats (last 14 days): {daily_stats}\n\n"
    context += f"User instructions: {user_instructions}\n\n"
    context += f"LLM silent notes: {silent_notes}\n\n"

    # construct context string
    sysprompt = f"You are a professional sport coach. Answer the user questions. Here is the context:\n{context}"
    history = get_history(n=5)
    print(sysprompt)
    print(user_message)
    answer = ask_gpt(sysprompt, user_message, history=history)
    print(answer)

    # update history
    history.append({"role": "user", "content": user_message})
    # cut assistant message to 1000 chars, user messages more important for history
    if answer and len(answer) > 1000:
        history.append({"role": "assistant", "content": answer[:1000] + "..."})
    else:
        history.append({"role": "assistant", "content": answer})
    save_history(history)
    # return as plain text
    return answer

# debug coach
#sport_coach_answer("How many players are there on the field during a soccer match?")

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

    if text.startswith("/start"):
        _send_message(TELEGRAM_API, chat_id,
                      "Hi! Chat ID: " + str(chat_id))
        return func.HttpResponse("ok")

    # Security
    if req.headers.get("X-Telegram-Bot-Api-Secret-Token") != TELEGRAM_WEBHOOK_SECRET or str(chat_id) != TELEGRAM_ID:
        return func.HttpResponse("forbidden", status_code=403)


    # Quick command handling
    if text.startswith("/get_instructions"):
        instructions = get_user_instructions()
        if instructions:
            _send_message(TELEGRAM_API, chat_id, f"Current user instructions:\n{instructions}")
        else:
            _send_message(TELEGRAM_API, chat_id, "No user instructions set.")
        return func.HttpResponse("ok")
    if text.startswith("/set_instructions"):
        instructions = text[len("/set_instructions"):].strip()
        if instructions:
            save_user_instructions(instructions)
            _send_message(TELEGRAM_API, chat_id, "User instructions updated.")
        else:
            _send_message(TELEGRAM_API, chat_id, "Please provide instructions after the command.")
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
        _send_message(TELEGRAM_API, chat_id, answer)
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


@app.timer_trigger(schedule="0 0 1 * * *", arg_name="myTimer", run_on_startup=False,
              use_monitor=False) 
def update_silent_notes(myTimer: func.TimerRequest) -> None:
    
    history = get_history(n=1000)
    max_len = 1500*4  # approx 4 chars per token
    # take messages from the end until max_len is reached. Only user messages
    combined_text = ""
    for msg in reversed(history):
        if msg["role"] == "user":
            if len(combined_text) + len(msg["content"]) > max_len:
                break
            combined_text = msg["content"] + "\n" + combined_text

    old_silent_notes = get_silent_notes()
    sysprompt = f"""You are a professional sport coach.
You have been assisting a user with their sport coaching needs.
Current user instructions:
{old_silent_notes}

Based on the recent conversation history below, update the silent notes.
Keep the notes short, only few bullet points summarizing what has been discussed.
Focus on stuff that should affect your future responses, this is your flexible memory.
"""
    user_turn = f"""Recent conversation history:
{combined_text}
"""
    new_silent_notes = ask_gpt(sysprompt, user_turn)
    save_silent_notes(new_silent_notes)
    logging.info("Silent notes updated.")
import os

from flask import Flask, abort, render_template, request

from agent import SummaryError, build_demo_summary, build_ai_summary


app = Flask(__name__)
app.config.update(
    MAX_CONTENT_LENGTH=64 * 1024,
    TRUSTED_HOSTS=["127.0.0.1", "localhost"],
)

MAX_FIELD_LENGTH = 800
FIELDS = {
    "symptoms": "Жалобы",
    "timeline": "Начало и динамика",
    "regular_medications": "Постоянные лекарства",
    "recent_medications": "Приём из-за этих жалоб",
    "allergies": "Аллергии",
    "chronic_conditions": "Хронические заболевания",
}
DEMO_FORM = {
    "symptoms": "Болит голова. Тошноты нет. Температуру не измерял.",
    "timeline": "Началось вчера вечером. Сегодня боль стала сильнее.",
    "regular_medications": "Постоянно ничего не принимаю.",
    "recent_medications": "Вчера выпил ибупрофен, дозу не помню.",
    "allergies": "",
    "chronic_conditions": "",
}


@app.route("/", methods=["GET", "POST"])
def index():
    api_key = os.getenv("MEDSAP_OPENAI_API_KEY", "").strip()
    values = dict.fromkeys(FIELDS, "")
    summary = None
    error = None
    error_field = None
    mode = "demo"
    status = 200

    if request.method == "GET" and request.args.get("sample") == "1":
        values = DEMO_FORM.copy()

    if request.method == "POST":
        origin = request.headers.get("Origin")
        if origin and origin != request.host_url.rstrip("/"):
            abort(403)

        values = {field: request.form.get(field, "").strip() for field in FIELDS}
        too_long = next((field for field, value in values.items()
                         if len(value) > MAX_FIELD_LENGTH), None)
        mode = request.form.get("mode", "demo")

        if not values["symptoms"]:
            error = "Опиши, что беспокоит пациента, и отправь анкету ещё раз."
            error_field, status = "symptoms", 400
        elif too_long:
            error = f"Сократи поле «{FIELDS[too_long]}» до {MAX_FIELD_LENGTH} символов."
            error_field, status = too_long, 400
        elif mode == "ai":
            if not api_key:
                error = "ИИ ещё не настроен. Сформируй сводку в демонстрационном режиме."
                status = 503
            else:
                try:
                    summary = build_ai_summary(values, api_key)
                except SummaryError:
                    error = "ИИ не вернул сводку. Попробуй ещё раз или сформируй её без ИИ."
                    status = 502
        else:
            summary = build_demo_summary(values)

    return render_template(
        "index.html",
        values=values,
        fields=FIELDS,
        max_length=MAX_FIELD_LENGTH,
        summary=summary,
        error=error,
        error_field=error_field,
        mode=mode,
        ai_available=bool(api_key),
    ), status, {"Cache-Control": "no-store"}


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000)

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from openai import OpenAI, OpenAIError
from pydantic import BaseModel, ConfigDict, ValidationError


PROMPT_PATH = Path(__file__).with_name("prompt.txt")
DEFAULT_MODEL = "gpt-5.4-mini"
QUESTIONS = {
    "timeline": "Когда начались жалобы и как они менялись?",
    "regular_medications": "Какие лекарства пациент принимает постоянно?",
    "recent_medications": "Принимал ли пациент что-нибудь из-за этих жалоб?",
    "regular_dose": "Какие дозировки у постоянно принимаемых препаратов?",
    "recent_dose": "В какой дозировке был принят препарат из-за этих жалоб?",
    "allergies": "Есть ли у пациента известные аллергии?",
    "chronic_conditions": "Есть ли у пациента хронические заболевания?",
    "inconsistency": "Уточнить противоречие в сообщённых сведениях.",
}


class SummaryError(Exception):
    """The app could not obtain a usable summary from the model."""


class ExtractedIntake(BaseModel):
    model_config = ConfigDict(extra="forbid")

    symptoms: list[str]
    timeline: list[str]
    regular_medications: list[str]
    recent_medications: list[str]
    allergies: list[str]
    chronic_conditions: list[str]
    clarify: list[Literal[
        "timeline", "regular_medications", "recent_medications", "regular_dose",
        "recent_dose", "allergies", "chronic_conditions", "inconsistency",
    ]]


@dataclass
class Summary:
    excerpts: dict[str, list[str]]
    questions: list[str]


def build_demo_summary(values: dict[str, str]) -> Summary:
    """Copy the completed fields into the summary without an API call."""
    excerpts = {name: [text] if text else [] for name, text in values.items()}
    questions = [QUESTIONS[name] for name in ("timeline", "regular_medications")
                 if not values[name]]
    return Summary(excerpts, questions)


def build_ai_summary(values: dict[str, str], api_key: str) -> Summary:
    """Ask the model to extract fragments and verify that they occur in the form."""
    model = os.getenv("MEDSAP_OPENAI_MODEL", "").strip() or DEFAULT_MODEL
    try:
        with OpenAI(api_key=api_key, timeout=20, max_retries=0) as client:
            response = client.responses.parse(
                model=model,
                instructions=PROMPT_PATH.read_text(encoding="utf-8"),
                input=json.dumps(values, ensure_ascii=False),
                text_format=ExtractedIntake,
                max_output_tokens=2400,
                store=False,
            )
    except (OpenAIError, ValidationError) as exc:
        raise SummaryError("API or response parsing error") from exc

    parsed = response.output_parsed
    if response.status != "completed" or parsed is None or not parsed.symptoms:
        raise SummaryError("Missing or incomplete response")

    excerpts = parsed.model_dump(exclude={"clarify"})
    for parts in excerpts.values():
        for part in parts:
            if not part.strip() or not any(part in source for source in values.values()):
                raise SummaryError("Response contained a fragment absent from the form")

    questions = [QUESTIONS[name] for name in dict.fromkeys(parsed.clarify)][:3]
    return Summary(excerpts, questions)

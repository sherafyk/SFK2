import openai
import json
import base64
import os
from pathlib import Path
from .models import FieldDocument
from pydantic import ValidationError

_client = None

def get_openai_client() -> openai.OpenAI:
    """Return an OpenAI client or raise a clear error if the API key is missing."""
    global _client
    if _client is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable is not set")
        _client = openai.OpenAI(api_key=api_key)
    return _client


def extract_document_data(image_path: Path, max_retries: int = 3) -> FieldDocument:
    schema = json.loads((Path(__file__).parent / "schema/field_doc_schema.json").read_text())
    schema["additionalProperties"] = False

    image_b64 = base64.b64encode(image_path.read_bytes()).decode()
    system_prompt = (
        "You are a maritime cargo document parser. Extract ALL values exactly per schema:\n"
        "1. Tank IDs, dates in ISO-8601, all tank rows,\n"
        "2. calculate summary totals,\n"
        "3. preserve decimals, etc."
    )
    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": "Extract all data from this document, incl. vessel name, tanks, timestamps."},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{image_b64}"}},
            ],
        },
    ]

    extracted = None
    client = get_openai_client()

    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model="o4-mini",
                messages=messages,
                temperature=0,
                response_format={
                    "type": "json_schema",
                    "json_schema": {"name": "field_document_extraction", "strict": True, "schema": schema},
                },
            )
            extracted = response.choices[0].message.content
            return FieldDocument.model_validate_json(extracted)
        except Exception as e:
            if attempt < max_retries - 1:
                messages.append({"role": "assistant", "content": extracted if extracted else ""})
                messages.append({"role": "user", "content": f"Please fix error and capture all data: {e}"})
            else:
                raise

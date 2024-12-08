import re
import typing as ta
import os

from anthropic import Anthropic
from openai import OpenAI

client = Anthropic(
    api_key=os.environ.get("ANTHROPIC_API_KEY", ""),
)


def convert_anthropic_tokens_to_cost(model, input_tokens, output_tokens):
    if model == "claude-3-haiku-20240307":
        output_cost_per_token = 1.25 / 1_000_000
        input_cost_per_token = 0.25 / 1_000_000
    elif model == "claude-3-sonnet-20240229" or model == "claude-3-5-sonnet-20240620":
        output_cost_per_token = 15 / 1_000_000
        input_cost_per_token = 3 / 1_000_000
    elif model == "claude-3-opus-20240229":
        output_cost_per_token = 75 / 1_000_000
        input_cost_per_token = 15 / 1_000_000
    else:
        raise ValueError("Model not supported")
    return input_tokens * input_cost_per_token + output_tokens * output_cost_per_token


def get_anthropic_response(sys_message: str, human_msg: str, model: str):
    message = client.messages.create(
        max_tokens=4000,
        system=sys_message,
        messages=[{"role": "user", "content": human_msg}],
        model=model,
    )
    return message.content[0].text, convert_anthropic_tokens_to_cost(
        model, message.usage.input_tokens, message.usage.output_tokens
    )

LLM_MODEL = ta.Literal[
    "opus",
    "haiku",
    "sonnet",
    "gpt-4-0125-preview",
    "gpt-3.5-turbo-0125",
    "gpt-4o",
    "gpt-4o-mini",
    "gpt-4-turbo-2024-04-09",
    # "google",
    # "llama-3-70b",
    # "llama-3-8b",
]


def get_ai_response(messages: list[str], model: LLM_MODEL):
    if len(messages) != 2:
        raise ValueError(f"Invalid number of messages. Currently only implemented for 2 messages. Received {len(messages)}")
    if model == "opus":
        return get_anthropic_response(
            messages[0], messages[1], "claude-3-opus-20240229"
        )
    elif model == "haiku":
        return get_anthropic_response(
            messages[0], messages[1], "claude-3-haiku-20240307"
        )
    elif model == "sonnet":
        return get_anthropic_response(
            messages[0], messages[1], "claude-3-5-sonnet-20240620"
        )
    elif (
        model == "gpt-4-0125-preview"
        or model == "gpt-3.5-turbo-0125"
        or model == "gpt-4-turbo-2024-04-09"
        or model == "gpt-4o"
        or model == "gpt-4o-mini"
    ):
        return openai_get_ai_response(messages, model)
    # elif model == "google":
    #     return get_google_response(messages), None
    # elif model == "llama-3-70b" or model == "llama-3-8b":
    #     return get_together_response(messages, model)
    else:
        raise ValueError("Invalid model:", model)
    

def openai_get_ai_response(messages: list[str], model: str):
    client = OpenAI()

    completion = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": messages[0]},
            {"role": "user", "content": messages[1]}
        ]
    )

    return completion.choices[0].message.content, -1.0


def extract_xml_key_value(xml_str: str, key: str) -> str:
    try:
        return re.findall(f'<{key}>(.*?)</{key}>', xml_str, flags=re.DOTALL)[0].strip()
    except IndexError:
        try:
            # A little bit of error handling and assumptions that the key is terribly off here
            return re.findall(f'<{key}>(.*?)</{key}', xml_str, flags=re.DOTALL)[0].strip()
        except IndexError:
            return ""
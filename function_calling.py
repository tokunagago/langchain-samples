# Function calling

import json

def get_current_weather(location, unit="fahrenheit"):
    if "tokyo" in location.lower():
        print("tokio--")
        return json.dumps({"location": location, "temperature": "10", "unit": unit, "forecast": ["sunny", "windy"]})
    else:
        return json.dumps({"loaction": location}, {"tempreature": "??"})

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_weather",
            "description": "Get the current weather in a given location",
            "parameters": {
                "type": "object",
                "properties": {
                    "location": {
                        "type": "string",
                        "description": "The city and state, e.g. San Francisco, CA",
                    },
                    "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]},
                },
                "required": ["location"],
            },
        },
    }
]


client = OpenAI()

messages = [
    {"role": "user", "content": "Tokyoの天気は？"}
]

response = client.chat.completions.create(
    model="gpt-4o-mini",
    messages=messages,
    tools=tools,
)

print(response.to_json(indent=2))

response_message = response.choices[0].message
messages.append(response_message.to_dict())

available_functions = {
    "get_current_weather": get_current_weather,
}

for tool_call in response_message.tool_calls:
    function_name = tool_call.function.name
    function_to_call = available_functions[function_name]
    function_args = json.loads(tool_call.function.arguments)

    function_response = function_to_call(
        location=function_args.get("location"),
        unit=function_args.get("unit"),
    )

    messages.append(
        {
            "tool_call_id": tool_call.id,
            "role": "tool",
            "name": function_name,
            "content": function_response,
        }
    )
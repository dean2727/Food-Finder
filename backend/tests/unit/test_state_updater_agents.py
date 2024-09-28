# This file is a bit different, in that it tests LLM's responses for consistency
# This is important for the early agents that need to accurately parse the user's query
# into the proper user preference values/weights

import pytest
from typing import Dict, Any
from datetime import datetime, timedelta
from app.graph import STATE_UPDATER_SYSTEM_PROMPT, DATETIME_EXTRACTOR_SYSTEM_PROMPT
from app.schemas import AgentState, UserPreferences, StateUpdaterOutputFormat, DateTimeExtract
from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

# Code to be moved to files (later)
model = ChatOpenAI(model="gpt-4o", temperature=0)

def extract_preferences(message: str) -> Dict[str, Any]:
    structured_model = model.with_structured_output(StateUpdaterOutputFormat)
    
    response = structured_model.invoke([
        SystemMessage(content=STATE_UPDATER_SYSTEM_PROMPT),
        HumanMessage(content=message)
    ])
    
    print(f"DEBUG: {response}")
    return response.dict()

def get_formatted_datetime():
    now = datetime.now()
    return now.strftime("It is currently %B %d, %Y. The time is %I:%M %p")

def extract_datetime(message: str) -> DateTimeExtract:
    structured_llm = model.with_structured_output(DateTimeExtract, method="json_mode")
    message = DATETIME_EXTRACTOR_SYSTEM_PROMPT.format(curr_day_time_msg=get_formatted_datetime(), user_query=message)
    return structured_llm.invoke(message)


# Using september 24, at 5:30 PM as the time the user would hypothetically ask their query
ANCHOR_TIME = datetime(2024, 9, 24, 17, 30)

# Test queries are queries that have some kind of indication of when user wants to eat (not the default of now, if user doesnt specify this)
test_queries = [
    # Check dietary_requests, distance and wants_outdoor_seating 
    "Im looking for a place to have dinner around 7 PM, preferably somewhere within 10 miles of downtown. I'd love a spot with a cozy atmosphere and a good selection of vegetarian dishes, but I'm open to other options if they have unique or standout menu items. Bonus points if they have outdoor seating and a view!",
    "I'm looking for a place to eat that serves Italian food. I'd prefer it to be within 5 miles of my current location, and I'd like it to be open for dinner. I'd prefer it to be a place that has a good atmosphere and serves good food. I'd also like it to have a good selection of vegetarian dishes, but I'm open to other options if they have unique or standout menu items. Bonus points if they have outdoor seating and a view!",
    "Tomorrow, I want to get some asian food by my house, during dinner.",
    "This Friday, im going out to eat at lunch time. What is good?",
    "I want to get some italian food during Halloween this year.",
    "Tomorrow at 8:00 AM, I want to grab some breakfast with friends."
]

def test_extract_datetime():
    expected_times = [
        datetime(2024, 9, 24, 19, 0), # 7 PM
        datetime(2024, 9, 24, 18, 0), # 6 PM
        datetime(2024, 9, 25, 18, 0), # 6 PM
        datetime(2024, 9, 27, 12, 0), # 12 PM
        datetime(2024, 10, 31, 18, 0), # Halloween, no time of day, but deduce dinner (6 PM)
        datetime(2024, 9, 25, 8, 0) # 8 AM
    ]

    test_cases = list(zip(test_queries, expected_times))

    for query, expected_time in test_cases:
        result = extract_datetime(query)
        print(f"DEBUG: {result.dt} == {expected_time}")
        assert result.dt == expected_time

def test_extract_preferences():
    expected_preferences = [
        ...

    ]

    for i in range(len(test_queries)):
        response = extract_preferences(test_queries[i])
        for p, v in expected_preferences[i].items():
            assert response[p]["value"] == v["value"]
            # We can be lenient on exact weights (margin of error of +- 0.1)
            assert abs(response[p]["weight"] - v["weight"]) <= 0.1


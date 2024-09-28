import os
from typing import Tuple, List
from datetime import datetime

from langchain_core.messages import HumanMessage, SystemMessage, ToolMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.prebuilt import ToolNode
from langgraph.graph import END, StateGraph

from app.schemas import Place, UserPreferences, AgentState, CustomAIMessage, DateTimeExtract, StateUpdaterOutputFormat
from app.graph.tools.places_search import google_maps_text_search_and_filter
from app.graph.prompts import MAPS_QUERY_FORMULATOR_SYSTEM_PROMPT, TEAM_SUPERVISOR_SYSTEM_PROMPT, DATETIME_EXTRACTOR_SYSTEM_PROMPT, STATE_UPDATER_SYSTEM_PROMPT

import logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG)

llm = ChatOpenAI(model="gpt-4o", temperature=0)

def get_formatted_datetime():
    now = datetime.now()
    return now.strftime("It is currently %B %d, %Y. The time is %I:%M %p")

def extract_datetime(message: str) -> DateTimeExtract:
    structured_llm = llm.with_structured_output(DateTimeExtract, method="json_mode")
    message = DATETIME_EXTRACTOR_SYSTEM_PROMPT.format(curr_day_time_msg=get_formatted_datetime(), user_query=message)
    return structured_llm.invoke(message)

def format_response_str_from_places(valid_places: List[Place]):
    """ Given places that conform to the user preferences, take the top n,
    and insert them into the portion of the supervisor agent's response to
    the user """
    NUM_RECS_TO_SHOW = min(5, len(valid_places))
    response_str = ""
    curr_rec = 1
    for place in valid_places[:NUM_RECS_TO_SHOW]:
        response_str += f"{curr_rec}. {place.display_name_text} - {place.primary_type_display_name_text}. Located at {place.formatted_address}. Phone number is {place.national_phone_number}. Rating is {place.rating} with {place.user_rating_count} ratings. ||"
        curr_rec += 1
    return response_str

def datetime_extractor_node(state: AgentState):
    # Grab the first human message's content
    user_query = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), None)
    structured_model = llm.with_structured_output(DateTimeExtract)
    message = DATETIME_EXTRACTOR_SYSTEM_PROMPT.format(curr_day_time_msg=get_formatted_datetime(), user_query=user_query)

    response = structured_model.invoke(message)

    # Now, return the updated desired time to eat
    new_user_pref = state['user_preferences']
    orig_stay_duration = new_user_pref.desired_time_and_stay_duration[1]
    new_user_pref.desired_time_and_stay_duration = (response.dt, orig_stay_duration)
    return {"user_preferences": new_user_pref, "datetime_extracted": True}

def state_updater_node(state: AgentState):
    # Grab the first human message's content
    user_query_with_preferences = next((msg.content for msg in state["messages"] if isinstance(msg, HumanMessage)), None)
    structured_model = llm.with_structured_output(StateUpdaterOutputFormat)

    response = structured_model.invoke([
        SystemMessage(content=STATE_UPDATER_SYSTEM_PROMPT),
        HumanMessage(content=user_query_with_preferences)
    ])
    new_preferences = response.dict()

    # After we get response preferences, we update the state
    # Note: We dont need to append an additional message from this agent to the state
    # Just need to update the other aspects of the state
    state_to_return = {}
    
    pref_restrictions = {k: new_preferences[k] for k in list(new_preferences.keys())[:6]}
    for k, v in pref_restrictions.items():
        if k == "length_of_stay":
            new_user_pref = state["user_preferences"]
            orig_user_pref_time_of_stay = state["user_preferences"].desired_time_and_stay_duration[0]
            new_user_pref.desired_time_and_stay_duration = (orig_user_pref_time_of_stay, v)
            state_to_return["user_preferences"] = new_user_pref
        else:
            state_to_return[k] = v

    pref_rest = {k: new_preferences[k] for k in list(new_preferences.keys())[6:]}
    user_preferences = UserPreferences.model_validate(pref_rest)
    state_to_return["user_preferences"] = user_preferences

    # Handling any other state variables not included in the StateUpdaterOutputFormat schema
    state_to_return["when_to_eat_specified"] = state["when_to_eat_specified"]

    return state_to_return

def maps_query_formulator_node(state: AgentState):
    messages = [SystemMessage(content=MAPS_QUERY_FORMULATOR_SYSTEM_PROMPT)] + state["messages"]
    response = llm.invoke(messages)
    # Use custom message, to inform later agent, as it searches past messages, where to retrieve the API query
    new_message = CustomAIMessage(content=response.content, originating_node="maps_query_formulator_node")
    return {"messages": [new_message]}

def team_supervisor_node(state: AgentState):
    # Grab the (latest) api query
    api_query = ""
    for message in reversed(state["messages"]):
        if isinstance(message, CustomAIMessage) and message.originating_node == "maps_query_formulator_node":
            api_query = message.content
            break
    
    messages = [SystemMessage(content=TEAM_SUPERVISOR_SYSTEM_PROMPT.format(api_query=api_query))] + state["messages"]
    response = team_supervisor.invoke(messages)

    # If we just called the tool to get back places, process the output of the tool to show user recommended places
    last_message = state['messages'][-1]
    if type(last_message) == ToolMessage and "Failed" not in last_message.content: # So far, just 1 tool is used - Google Maps search
        valid_places, invalid_places = last_message.artifact

        place_recommendations_str = format_response_str_from_places(valid_places)

        new_message = AIMessage(content=response.content + "\n\n" + place_recommendations_str)

        return {
            'valid_places': {p.display_name_text: p for p in valid_places},
            'invalid_places': {p[0].display_name_text: p for p in invalid_places},
            'messages': [new_message]
        }
    # Otherwise, just return the agent's response
    return {
        'messages': [response]
    }

# ~~~~~~~~~~~~~~~~~~~ Graph setup ~~~~~~~~~~~~~~~~~~~

workflow = StateGraph(AgentState)

# Separate tool node so we can pass in state
tool_node = ToolNode([google_maps_text_search_and_filter])
team_supervisor = llm.bind_tools([google_maps_text_search_and_filter])

workflow.add_node('state_updater_node', state_updater_node)
workflow.add_node('datetime_extractor_node', datetime_extractor_node)
workflow.add_node('maps_query_formulator_node', maps_query_formulator_node)
workflow.add_node('team_supervisor_node', team_supervisor_node)
workflow.add_node('google_maps_text_search_and_filter', tool_node)

workflow.set_entry_point('state_updater_node')

workflow.add_edge('maps_query_formulator_node', 'team_supervisor_node')

def what_to_do_next_for_supervisor(state: AgentState, config):
    messages = state['messages']
    last_message = messages[-1]

    # If there is no function call, then we finish
    if not last_message.tool_calls:
        return "end"
    # Otherwise if there is, we continue
    else:
        return "get_places"

workflow.add_conditional_edges(
    'team_supervisor_node',
    what_to_do_next_for_supervisor,
    {
        'get_places': "google_maps_text_search_and_filter",
        'end': END,
    },
)

def what_to_do_next_for_state_updater(state: AgentState, config):
    when_to_eat_specified = state["when_to_eat_specified"]
    datetime_extracted = state["datetime_extracted"]

    # If they specify when to eat, and datetime isnt yet extracted, extract datetime
    if when_to_eat_specified:
        if not datetime_extracted:
            return "extract_datetime"
        else:
            return "go_to_maps_query_formulator"
    else:
        return "go_to_maps_query_formulator"

workflow.add_conditional_edges(
    'state_updater_node',
    what_to_do_next_for_state_updater,
    {
        'extract_datetime': "datetime_extractor_node",
        'go_to_maps_query_formulator': "maps_query_formulator_node"
    }
)

workflow.add_edge('google_maps_text_search_and_filter', 'team_supervisor_node')
workflow.add_edge('datetime_extractor_node', 'state_updater_node')

food_finder_agent = workflow.compile()


DEFAULT_AGENT_STATE: AgentState = {
    "when_to_eat_specified": False,
    "datetime_extracted": False,
    "preferred_price_level": "PRICE_LEVEL_UNSPECIFIED",
    "desired_star_rating": 0.0,
    "user_coordinates": None,
    "preferred_direction": "any",
    "desired_max_distance_meters": 16093.0,  # 10 miles
    "user_preferences": UserPreferences(),
    "valid_places": {},
    "invalid_places": {},
    "found_place": False
}

def create_initial_state(user_input: str, user_coordinates: Tuple[float, float] | None = None) -> AgentState:
    return AgentState({
        **DEFAULT_AGENT_STATE,
        "messages": [HumanMessage(content=user_input)],
        "user_coordinates": user_coordinates
    })

# TODO: 
# - Implement human feedback with the team supervisor
# - Implement a review analyzer (for a place), that the supervisor can communicate with for more details reviews information

if __name__ == "__main__":
    from app.graph import set_environment_variables_langsmith
    set_environment_variables_langsmith("food_finder_test")

    user_input = """
    I am hungry and want to find somewhere to get some dinner. I want to eat at 7 for about an hour.
    I am going by myself. I am feeling like having Asian Cuisine. I dont want to drive more than 3 miles."
    """

    AUSTIN_TEST_COORDINATES = (30.2672, -97.7431)
    initial_state = create_initial_state(user_input, AUSTIN_TEST_COORDINATES)

    tool_called = False
    for chunk in food_finder_agent.stream(initial_state):
        for key, value in chunk.items():
            print(f"Output from node '{key}':")
            print("---")
            if key == 'google_maps_text_search_and_filter':
                print(value['messages'][0].content)
                tool_called = True
            elif key == 'team_supervisor_node':
                if tool_called:
                    print(value['messages'][0].content)
                else:
                    print(value['messages'])
            else:
                print(value)
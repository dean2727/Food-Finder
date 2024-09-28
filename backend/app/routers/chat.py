from fastapi import APIRouter, Depends, HTTPException
from fastapi.requests import Request
from uuid import UUID
from pydantic import BaseModel
from typing import List, Union, Dict, Any, Literal
from uuid import uuid4

from langgraph.graph.graph import CompiledGraph
from langchain_core.runnables import RunnableConfig

from app.main import app
from app.schemas import ChatMessage, UserInput, ChatRequest
from app.utils import parse_input
from app.graph.food_finder_agent import *

router = APIRouter()

def get_agent(request: Request):
    return request.app.state.agent

def _parse_input(user_input: UserInput, user_coordinates: Tuple[float, float] | None) -> Tuple[Dict[str, Any], str]:
    run_id = uuid4()
    thread_id = user_input.thread_id
    input_message = HumanMessage(content=user_input.message)

    # If this is a new thread (chat), then we want our state to have defaults
    if not thread_id:
        thread_id = str(uuid4())
        state = AgentState({
            **DEFAULT_AGENT_STATE,
            "messages": [input_message],
            "user_coordinates": user_coordinates
        })
    else:
        state = {
            "messages": [input_message],
            "user_coordinates": user_coordinates
        }

    kwargs = dict(
        input=state,
        config=RunnableConfig(
            configurable={"thread_id": thread_id, "model": user_input.model},
            run_id=run_id,
        ),
    )

    return kwargs, run_id

@router.post("/invoke-with-history")
async def invoke_with_history(chat_request: ChatRequest):
    agent: CompiledGraph = app.state.agent

    user_location = chat_request.userLocation
    user_location = (user_location.latitude, user_location.longitude)

    last_user_message = chat_request.messages[-1].content
    user_input: UserInput = UserInput(message=last_user_message, thread_id=chat_request.thread_id)
    kwargs, run_id = _parse_input(user_input, user_location)

    try:
        try:
            response = await agent.ainvoke(**kwargs)
        except Exception as e:
            logging.error(f"Error invoking agent: {e}")

        ai_last_message = response.get('messages')[-1].content
        return ai_last_message
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
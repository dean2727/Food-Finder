from contextlib import asynccontextmanager
import os
from uuid import uuid4
from typing import Dict, Any, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn

from langchain_core.messages import HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.graph.graph import CompiledGraph

from app.schemas import ChatMessage, Feedback, UserInput, StreamInput
#from app.routers import chat
from app.graph.food_finder_agent import food_finder_agent, create_initial_state, DEFAULT_AGENT_STATE
from app.schemas import ChatRequest, AgentState

import logging
logging.basicConfig(filename='debug.log', level=logging.DEBUG)

# "dev" or "prod"
ENVIRON = "dev"

#set_environment_variables(ENVIRON, "fastapi_dev")

# TODO: fix this
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Construct agent with Sqlite chectkpointer
    async with AsyncSqliteSaver.from_conn_string("checkpoints.db") as saver:
        food_finder_agent.checkpointer = saver
        app.state.agent = food_finder_agent
        yield
    # context manager will clean up the AsyncSqliteSaver on exit

app = FastAPI(lifespan=lifespan)
#app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # Allow your frontend origin
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# TODO: Hook up chatbot-ui with UVICORN_SERVER_HOST:UVICORN_SERVER_PORT/chat/invoke
#app.include_router(chat.router, prefix="/chat", tags=["chat"])

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

# TODO: Add this back to routers
# TODO: get the frontend to have persistent thread_id
@app.post("/chat/invoke-with-history")
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


if __name__ == "__main__":  # pragma: no cover
    uvicorn.run(
        "app.main:app",
        host=os.environ["UVICORN_SERVER_HOST"],
        port=os.environ["UVICORN_SERVER_PORT"],
        reload=True,
        log_level="debug"
    )
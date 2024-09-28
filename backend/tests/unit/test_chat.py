from langchain_core.messages import AIMessage
from fastapi.testclient import TestClient
from unittest.mock import patch, AsyncMock

from app.main import app
from app.schemas import ChatMessage

client = TestClient(app)

# more tests reference code: https://github.com/JakubPluta/gymhero/tree/1e57ec1c325199133d81ffb2dd840aa600903ef0/tests

@patch("app.agent.food_finder_agent")
def test_invoke(mock_agent):
    # TODO: Edit this
    QUESTION = "What is the weather in Tokyo?"
    ANSWER = "The weather in Tokyo is 70 degrees."
    agent_response = {"messages": [AIMessage(content=ANSWER)]}
    mock_agent.ainvoke = AsyncMock(return_value=agent_response)

    with client as c:
        response = c.post("/chat/invoke", json={"message": QUESTION})
        assert response.status_code == 200

    mock_agent.ainvoke.assert_awaited_once()
    input_message = mock_agent.ainvoke.await_args.kwargs["input"]["messages"][0]
    assert input_message.content == QUESTION

    output = ChatMessage.parse_obj(response.json())
    assert output.type == "ai"
    assert output.content == ANSWER
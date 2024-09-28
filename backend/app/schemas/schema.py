from typing import TypedDict, Dict, Any, List, Literal, Tuple, Annotated, Sequence, Optional
from datetime import datetime
import operator

from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    AIMessage,
    ToolMessage,
    ToolCall,
    message_to_dict,
    messages_from_dict,
)
from pydantic import BaseModel, Field

# ~~~~~~ Chat API models ~~~~~~
class UserInput(BaseModel):
    """Basic user input for the agent."""

    message: str = Field(
        description="User input to the agent.",
        examples=["What is the weather in Tokyo?"],
    )
    model: str = Field(
        description="LLM Model to use for the agent.",
        default="gpt-4o-mini",
        examples=["gpt-4o-mini", "llama-3.1-70b"],
    )
    thread_id: str | None = Field(
        description="Thread ID to persist and continue a multi-turn conversation.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )

# TODO: add this back later?
# class ChatSettings(BaseModel):
#     model: str  # Equivalent to LLMID in TypeScript
#     prompt: str
#     temperature: float
#     contextLength: int
#     includeProfileContext: bool
#     includeWorkspaceInstructions: bool
#     embeddingsProvider: Literal["openai", "local"]


class StreamInput(UserInput):
    """User input for streaming the agent's response."""

    stream_tokens: bool = Field(
        description="Whether to stream LLM tokens to the client.",
        default=True,
    )


class AgentResponse(BaseModel):
    """Response from the agent when called via /invoke."""

    message: Dict[str, Any] = Field(
        description="Final response from the agent, as a serialized LangChain message.",
        examples=[
            {
                "message": {
                    "type": "ai",
                    "data": {"content": "The weather in Tokyo is 70 degrees.", "type": "ai"},
                }
            }
        ],
    )


class ChatMessage(BaseModel):
    """Message in a chat."""

    type: Literal["human", "ai", "tool"] = Field(
        description="Role of the message.",
        examples=["human", "ai", "tool"],
    )
    content: str = Field(
        description="Content of the message.",
        examples=["Hello, world!"],
    )
    tool_calls: List[ToolCall] = Field(
        description="Tool calls in the message.",
        default=[],
    )
    tool_call_id: str | None = Field(
        description="Tool call that this message is responding to.",
        default=None,
        examples=["call_Jja7J89XsjrOLA5r!MEOW!SL"],
    )
    run_id: str | None = Field(
        description="Run ID of the message.",
        default=None,
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    original: Dict[str, Any] = Field(
        description="Original LangChain message in serialized form.",
        default={},
    )

    @classmethod
    def from_langchain(cls, message: BaseMessage) -> "ChatMessage":
        """Create a ChatMessage from a LangChain message."""
        original = message_to_dict(message)
        match message:
            case HumanMessage():
                human_message = cls(type="human", content=message.content, original=original)
                return human_message
            case AIMessage():
                ai_message = cls(type="ai", content=message.content, original=original)
                if message.tool_calls:
                    ai_message.tool_calls = message.tool_calls
                return ai_message
            case ToolMessage():
                tool_message = cls(
                    type="tool",
                    content=message.content,
                    tool_call_id=message.tool_call_id,
                    original=original,
                )
                return tool_message
            case _:
                raise ValueError(f"Unsupported message type: {message.__class__.__name__}")

    def to_langchain(self) -> BaseMessage:
        """Convert the ChatMessage to a LangChain message."""
        if self.original:
            return messages_from_dict([self.original])[0]
        match self.type:
            case "human":
                return HumanMessage(content=self.content)
            case _:
                raise NotImplementedError(f"Unsupported message type: {self.type}")

    def pretty_print(self) -> None:
        """Pretty print the ChatMessage."""
        lc_msg = self.to_langchain()
        lc_msg.pretty_print()


class Feedback(BaseModel):
    """Feedback for a run, to record to LangSmith."""

    run_id: str = Field(
        description="Run ID to record feedback for.",
        examples=["847c6285-8fc9-4560-a83f-4e6285809254"],
    )
    key: str = Field(
        description="Feedback key.",
        examples=["human-feedback-stars"],
    )
    score: float = Field(
        description="Feedback score.",
        examples=[0.8],
    )
    kwargs: Dict[str, Any] = Field(
        description="Additional feedback kwargs, passed to LangSmith.",
        default={},
        examples=[{"comment": "In-line human feedback"}],
    )

class Coordinates(BaseModel):
    latitude: float
    longitude: float

class Message(BaseModel):
    role: str
    content: str

class ChatRequest(BaseModel):
    """A request to the primary chat route (/invoke-with-history)."""
     #chatSettings: ChatSettings
    userAllowedLocation: bool
    userLocation: Coordinates
    messages: List[Message]
    thread_id: str | None = Field(default=None)
    #customModelId: str = ""

# ~~~~~~ Models for Google Places API and graph agents ~~~~~~
class TimeInfo(BaseModel):
    """Information about a time, extracted from a places's information."""
    day: int
    hour: int
    minute: int

class OpenClosePeriod(BaseModel):
    """TimeInfo for a given open/close period of a place."""
    open: TimeInfo
    close: TimeInfo

class RegularOpeningHours(BaseModel):
    """Regular opening hours for a place, which is a list of OpenClosePeriods
    throughout a typical week."""
    periods: List[OpenClosePeriod]

class ReviewText(BaseModel):
    """Text for a review (includes language indication)."""
    text: str
    language_code: str = Field(alias="languageCode")

class Review(BaseModel):
    """A review for a place (comes from Places API text searchresponse schema)."""
    name: str
    relative_publish_time_description: str = Field(alias="relativePublishTimeDescription")
    rating: int
    text: ReviewText = Field(alias="originalText")
    publish_time: str = Field(alias="publishTime")

    class Config:
        populate_by_name = True

class ParkingOptions(BaseModel):
    """Parking options for a place. Each field corresponds to a possible attribute (bool)
    of parking we would see in a place from places API text search response schema."""
    free_parking_lot: bool = Field(alias="freeParkingLot", default=False)
    paid_parking_lot: bool = Field(alias="paidParkingLot", default=False)
    free_street_parking: bool = Field(alias="freeStreetParking", default=False)
    paid_street_parking: bool = Field(alias="paidStreetParking", default=False)
    valet_parking: bool = Field(alias="valetParking", default=False)
    free_garage_parking: bool = Field(alias="freeGarageParking", default=False)
    paid_garage_parking: bool = Field(alias="paidGarageParking", default=False)

class Place(BaseModel):
    """A food place (mirrors the Places API text search response schema)."""
    name: str # ID
    types: List[str]
    national_phone_number: str = Field(alias="nationalPhoneNumber", default=None)
    formatted_address: str = Field(alias="formattedAddress")
    location: Coordinates
    rating: float
    google_maps_uri: str = Field(alias="googleMapsUri")
    website_uri: str = Field(alias="websiteUri", default=None)
    regular_opening_hours: RegularOpeningHours = Field(alias="regularOpeningHours")
    price_level: str = Field(alias="priceLevel", default="PRICE_LEVEL_UNSPECIFIED")
    user_rating_count: int = Field(alias="userRatingCount")
    display_name_text: str = Field(alias="displayName.text")
    primary_type_display_name_text: str = Field(alias="primaryTypeDisplayName.text")
    reviews: List[Review]
    dine_in: bool = Field(alias="dineIn", default=False)
    serves_lunch: bool = Field(alias="servesLunch", default=False)
    serves_dinner: bool = Field(alias="servesDinner", default=False)
    outdoor_seating: bool = Field(alias="outdoorSeating", default=False)
    live_music: bool = Field(alias="liveMusic", default=False)
    serves_dessert: bool = Field(alias="servesDessert", default=False)
    serves_beer: bool = Field(alias="servesBeer", default=False)
    serves_wine: bool = Field(alias="servesWine", default=False)
    serves_brunch: bool = Field(alias="servesBrunch", default=False)
    serves_cocktails: bool = Field(alias="servesCocktails", default=False)
    serves_coffee: bool = Field(alias="servesCoffee", default=False)
    serves_vegetarian_food: bool = Field(alias="servesVegetarianFood", default=False)
    good_for_children: bool = Field(alias="goodForChildren", default=False)
    menu_for_children: bool = Field(alias="menuForChildren", default=False)
    good_for_groups: bool = Field(alias="goodForGroups", default=False)
    parking_options: ParkingOptions = Field(alias="parkingOptions", default=None)

    class Config:
        alias_generator = lambda string: ''.join(
			word.capitalize() if i > 0 else word
            for i, word in enumerate(string.split('_'))
        )
        populate_by_name = True
    
    @classmethod
    def model_validate(cls, obj, *args, **kwargs):
        if isinstance(obj, dict) and "displayName" in obj:
            obj["displayName.text"] = obj["displayName"]["text"]
        if isinstance(obj, dict) and "primaryTypeDisplayName" in obj:
            obj["primaryTypeDisplayName.text"] = obj["primaryTypeDisplayName"]["text"]
        return super().model_validate(obj, *args, **kwargs)
    
    def __str__(self):
        s = f"{self.display_name_text} - {self.primary_type_display_name_text}\n"
        s += f"Phone: {self.national_phone_number}\n"
        s += f"Address: {self.formatted_address}\n"
        s += f"Rating: {self.rating}\n"
        s += f"Price level: {self.price_level}\n"
        s += f"User rating count: {self.user_rating_count}\n"
        s += f"Open hours: {self.regular_opening_hours}\n"
        s += f"Outdoor seating: {self.outdoor_seating}\n"
        s += f"Live music: {self.live_music}\n"
        s += f"Serves dessert: {self.serves_dessert}\n"
        s += f"Serves beer: {self.serves_beer}\n"
        s += f"Serves wine: {self.serves_wine}\n"
        s += f"Serves brunch: {self.serves_brunch}\n"
        s += f"Serves cocktails: {self.serves_cocktails}\n"
        s += f"Serves coffee: {self.serves_coffee}\n"
        s += f"Serves vegetarian food: {self.serves_vegetarian_food}\n"
        s += f"Good for children: {self.good_for_children}\n"
        s += f"Menu for children: {self.menu_for_children}\n"
        s += f"Good for groups: {self.good_for_groups}\n"
        s += f"Parking options: {self.parking_options}\n"
        return s

class PreferenceWeight(BaseModel):
    """A way to gauge both the value and importance weight of an aspect of the user's
    food/place preference. The weights (for non-default/non-null values, i.e. user-specified
    preferences) are utilized in the sorting algorithm before results are presented to the user."""
    value: Any
    '''
    Values for preference weights can be:
    - Need -> 1.0
    - Strongly want -> 0.8
    - Want -> 0.5
    - Nice to have -> 0.3
    '''
    weight: float = Field(default=1.0, ge=0.0, le=1.0)

class UserPreferences(BaseModel):
    """User preferences, set by the state updater and datetime extracotr agents after the first 
    user query in a new chat."""
    desired_cuisines: PreferenceWeight = Field(default=PreferenceWeight(value=["any"], weight=0.8))
    party_size: PreferenceWeight = Field(default=PreferenceWeight(value=1, weight=1.0))
    desired_time_and_stay_duration: Tuple[datetime, int] = Field(default=(datetime.now(), 60))
    desired_minimum_num_ratings: PreferenceWeight = Field(default=PreferenceWeight(value=0, weight=0.3))
    dietary_requests: PreferenceWeight = Field(default=PreferenceWeight(value=[], weight=1.0))
    wants_family_friendly: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.6))
    wants_childrens_menu: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.5))
    wants_free_parking: PreferenceWeight = Field(default=PreferenceWeight(value=True, weight=0.8))
    wants_outdoor_seating: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.5))
    wants_live_music: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_dessert: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_beer: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_wine: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_brunch: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.7))
    wants_cocktails: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_coffee: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.6))

class RecommendedPlaceDetails(BaseModel):
    # TODO: This would likely hold more information on certain places, after the user asks about them
    ...

# Using TypedDict instead of pydantic BaseModel, as the latter doesnt work with InjectedState (in tool nodes)
class AgentState(TypedDict, total=False):
    """The state for a given agent, passed around the graph and updated primarily after first user query."""
    messages: Annotated[Sequence[BaseMessage], operator.add]
    
    # If this is identified in users query, we go to datetime extractor node to extract datetime
    when_to_eat_specified: bool # default=False
    # We need this so we dont go to datetime extractor a second time
    datetime_extracted: bool # default=False
    # User preferences, which are also optional parameters to places API request
    '''
    Values for preferred_price_level (per person) can be:
    - PRICE_LEVEL_UNSPECIFIED (user doesnt care about price level)
    - PRICE_LEVEL_INEXPENSIVE (<=$10)
    - PRICE_LEVEL_MODERATE ($10-$30)
    - PRICE_LEVEL_EXPENSIVE ($30-$60)
    - PRICE_LEVEL_VERY_EXPENSIVE (>$60)
    This is calculated from the user interface agent
    '''
    preferred_price_level: str  # default="PRICE_LEVEL_UNSPECIFIED"
    
    # Rating preference
    desired_star_rating: float  # default=0.0
    
    '''
    Regarding the location preference, follow this logic:
    1. User must opt in to location sharing (otherwise, we ask them for their general area)
    2. (Assuming location is shared) Does the user provide a preferred direction on the map?
        2a. If yes, then draw a box, where opposite corner from their location has a distance of of desired_max_distance_miles (converted to meters)
        2b. If no, then have the search area be a circle, with center user_coordinates and radius desired_max_distance_miles (converted to meters)
    '''
    # This comes from the frontend, in the first chat request body
    user_coordinates: Optional[Tuple[float, float]]
    
    # Preferred direction from user (can be "any", "N", "NE", etc. - 4 primary and 4 intermediate directions of a compass)
    # This informs the locationBias optional parameter for places API
    preferred_direction: str  # default="any"
    
    # Max desired meters the user would like to travel to eat
    # Bot asks the user for number of miles, though, so we convert first
    # This informs the locationBias optional parameter for places API
    desired_max_distance_meters: float  # default=16093.0 (10 miles)
    
    user_preferences: UserPreferences
    
    # Filtered places from the original text search (by team supervisor)
    valid_places: Dict[str, Place]
    
    # Map of place name to reason why it was filtered out
    invalid_places: Dict[str, Tuple[Place, str]]
    
    # End goal is for this to be true (user says yes to a recommended place) (future state - not used at the moment)
    found_place: bool  # default=False

class CustomAIMessage(AIMessage):
    """A custom AIMessage, which includes the originating node of the message in the state.
    The vanilla implementation does not specify what LangGraph node the AIMessage origintes
    from. This is useful in state updating scenarios we need to check if/when a message from
    a certain agent was added to the messages list in the state."""
    def __init__(self, content: str, originating_node: str, **kwargs):
        super().__init__(content=content, **kwargs)
        self.originating_node = originating_node

class StateUpdaterOutputFormat(BaseModel):
    """The output format for the state updater agent. The first 6 fields correspond to certain
    restrictions, and the rest correspond to fields in UserPreferences."""
    when_to_eat_specified: bool = Field(default=False) # If true, then extract datetime into DateTimeExtract
    length_of_stay: int = Field(default=60)
    preferred_price_level: str = Field(default="PRICE_LEVEL_UNSPECIFIED")
    desired_star_rating: int = Field(default=0)
    preferred_direction: str = Field(default="any")
    desired_max_distance_meters: float = Field(default=16093.0)

    desired_cuisines: PreferenceWeight = Field(default=PreferenceWeight(value=["any"], weight=0.8))
    party_size: PreferenceWeight = Field(default=PreferenceWeight(value=1, weight=1.0))
    desired_minimum_num_ratings: PreferenceWeight = Field(default=PreferenceWeight(value=0, weight=0.3))
    dietary_requests: PreferenceWeight = Field(default=PreferenceWeight(value=[], weight=1.0))
    wants_family_friendly: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.6))
    wants_childrens_menu: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.5))
    wants_free_parking: PreferenceWeight = Field(default=PreferenceWeight(value=True, weight=0.8))
    wants_outdoor_seating: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.5))
    wants_live_music: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_dessert: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_beer: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_wine: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_brunch: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.7))
    wants_cocktails: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.3))
    wants_coffee: PreferenceWeight = Field(default=PreferenceWeight(value=False, weight=0.6))

class DateTimeExtract(BaseModel):
    """The output format for the datetime extractor agent."""
    dt: datetime
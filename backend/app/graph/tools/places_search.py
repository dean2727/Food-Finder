from datetime import datetime, time, timedelta
from typing import Annotated, List, Tuple, Dict, Any
import math
import os
import requests

from langchain.tools import tool
from langgraph.prebuilt import InjectedState

from app.schemas import Place, PreferenceWeight, UserPreferences, AgentState

import logging

GOOGLE_FIELD_MASK = "places.name,places.types,places.nationalPhoneNumber,places.formattedAddress,places.location,places.rating,places.googleMapsUri,places.websiteUri,places.regularOpeningHours,places.priceLevel,places.userRatingCount,places.displayName,places.primaryTypeDisplayName,places.reviews,places.dineIn,places.servesLunch,places.servesDinner,places.outdoorSeating,places.liveMusic,places.servesDessert,places.servesBeer,places.servesWine,places.servesBrunch,places.servesCocktails,places.servesCoffee,places.servesVegetarianFood,places.goodForChildren,places.menuForChildren,places.goodForGroups,places.parkingOptions"

def get_datetime_for_place_hours(day: int, hour: int, minute: int) -> datetime:
    """Get a datetime object for the start of the place's hours on a given day.
    Note: In the Google API response, the day is 0 indexed from Sunday"""
    # Adjust day to match Python's datetime (where Monday is 0)
    adjusted_day = (day - 1) % 7

    # Get the next occurrence of the specified day
    today = datetime.now()
    days_ahead = adjusted_day - today.weekday()
    if days_ahead <= 0:
        days_ahead += 7
    
    next_day = today + timedelta(days=days_ahead)

    # Combine the date with the time
    dt = datetime.combine(next_day.date(), time(hour, minute))
    return dt

def check_if_user_stay_fits_open_hours(place: Place, user_stay: Tuple[datetime, int]) -> Tuple[bool, str]:
    user_start, duration = user_stay
    user_end = user_start + timedelta(minutes=duration)
    
    if not place.regular_opening_hours or not place.regular_opening_hours.periods:
        return False, "No opening hours information available for this place."

    # Get periods for the user's start day and the next day
    day_periods = [period for period in place.regular_opening_hours.periods 
                   if period.open.day in [user_start.weekday(), (user_start.weekday() + 1) % 7]]
    
    if not day_periods:
        return False, "This place is closed on the day you want to visit."

    def get_datetime(date, time_info):
        dt = datetime.combine(date, datetime.min.time().replace(hour=time_info.hour, minute=time_info.minute))
        if time_info.day != date.weekday():
            dt += timedelta(days=1)
        return dt

    for period in day_periods:
        open_time = get_datetime(user_start.date(), period.open)
        close_time = get_datetime(user_start.date(), period.close)
        
        if close_time < open_time:  # Handle case where closing time is after midnight
            close_time += timedelta(days=1)

        if open_time <= user_start < close_time:
            if user_end <= close_time:
                return True, ""
            else:
                return False, f"This place will close at {close_time.strftime('%I:%M %p')} before you finish your stay. Consider shortening your visit or coming earlier."

    # If we've reached this point, the user's stay doesn't fit any period
    opening_hours_str = ', '.join(sorted(set([f"{get_datetime(user_start.date(), p.open).strftime('%I:%M %p')} - {get_datetime(user_start.date(), p.close).strftime('%I:%M %p')}" for p in day_periods])))
    return False, f"This place will be closed when you want to visit. The opening hours for this day are: {opening_hours_str}. Please adjust your visit time accordingly."

def calculate_rating_score(place_rating_count: int, user_preference_rating_count: int, weight_of_user_preference_rating_count: float) -> float:
    """ This function gives us a score for the discrepancy between the user's desired number of
    star ratings for this place and the actual number of ratings the place has. It uses linear
    scaling, and is limited to the range of 0-0.2, if the place does not meet the requirements.
    If the place does meet the requirements, we use logarithmic scaling of the excess ratings. 
    Score is then multipltied by the weight the user gives to this metric. """
    if place_rating_count < user_preference_rating_count:
        raw_score = (place_rating_count / user_preference_rating_count)
        adjusted_score = raw_score * 0.2
        return adjusted_score * weight_of_user_preference_rating_count
    
    excess_ratings = place_rating_count - user_preference_rating_count
    max_excess = 1000
    
    rating_score = min(1, math.log(excess_ratings + 1) / math.log(max_excess + 1))
    
    return rating_score * weight_of_user_preference_rating_count

def calculate_place_score(place: Place, user_preferences: UserPreferences) -> float:
    """ This algorithm will give us a score for a given place, using the response 
    data from the API and the weights the user has given (or the default weights,
    if user did not specify that information) """
    score = 0.0
    
    considered_preferences = set(user_preferences.model_dump().keys()) - set(['desired_time_and_stay_duration', 'party_size'])

    for pref, pref_weight in user_preferences.model_dump().items():
        # If this preference is false or unspecified, or its a preference used only for restrictions, continue
        if pref == 'desired_time_and_stay_duration' or pref_weight['value'] == False or pref not in considered_preferences:
            continue
        match pref:                
            case "desired_minimum_num_ratings":
                rating_score = calculate_rating_score(place.user_rating_count, pref_weight['value'], pref_weight['weight'])
                score += rating_score
            case "dietary_requests":
                req_score = 0
                accomodates_requests = True
                if "vegan" in pref_weight['value']:
                    if "vegan" not in place.primary_type_display_name_text.lower():
                        accomodates_requests = False
                if "vegetarian" in pref_weight['value']:
                    if not place.serves_vegetarian_food:
                        accomodates_requests = False
                if accomodates_requests:
                    req_score = 1.0 * pref_weight['weight']
                score += req_score
            case "wants_family_friendly":
                if place.good_for_children:
                    score += pref_weight['weight']
            case "wants_childrens_menu":
                if place.menu_for_children:
                    score += pref_weight['weight']
            case "wants_free_parking":
                parking_dict = {
                    attr: getattr(place.parking_options, attr)
                    for attr in place.parking_options.model_fields
                }
                num_free_options = 0
                # More free options -> higher score
                for k in parking_dict.keys():
                    if k.startswith("free"):
                        num_free_options += 1
                parking_score = 0.25 + (0.25 * num_free_options)
                parking_score += pref_weight['weight']
                score += parking_score
            case "wants_outdoor_seating":
                if place.outdoor_seating:
                    score += pref_weight['weight']
            case "wants_live_music":
                if place.live_music:
                    score += pref_weight['weight']
            case "wants_dessert":
                if place.serves_dessert:
                    score += pref_weight['weight']
            case "wants_beer":
                if place.serves_beer:
                    score += pref_weight['weight']
            case "wants_wine":
                if place.serves_wine:
                    score += pref_weight['weight']
            case "wants_brunch":
                if place.serves_brunch:
                    score += pref_weight['weight']
            case "wants_cocktails":
                if place.serves_cocktails:
                    score += pref_weight['weight']
            case "wants_coffee":
                if place.serves_coffee:
                    score += pref_weight['weight']

    # TODO: Add distances (between coordinates) later?
    return score

def filter_places(places: List[Place], user_preferences: UserPreferences) -> Tuple[List[Place], List[Tuple[Place, str]]]:
    """ 
    Given a list of places and a state containing 0 or more user preferences,
    filter and sort them to best satisfy the user's preferences.
    A restriction is defined by a preference with a weight of 1.0 (the user needs this to be true).
    A place is filtered if any of the restrictions are not met.
    Then, we compute total scores for each place, using the preference weights, and this
    is used as the sorting criteria.
    Preferences considered in filtering include:
    - Desired cuisines
    - Minimum number of ratings
    - Dietary requests (e.g. vegetarian, vegan, dairy free)
    - Family friendly
    - Childrens menu
    - Free parking
    - Party size
    - Whether the desired time of arrival and stay duration are within opening hours
    - Whether the place has outdoor seating
    - Whether the place has live music
    - Whether the place serves dessert
    - Whether the place serves beer
    - Whether the place serves wine
    - Whether the place serves brunch
    - Whether the place serves cocktails
    - Whether the place serves coffee
    Some of these are booleans, so if false, dont need to consider their weights
    """

    #logging.debug(f"DEBUG: user_preferences: {user_preferences}")

    # First, filter. Grab the preferences with weights of 1.0, which contain non-default (truthy) values
    preferences_with_weight_one = {
        attr: getattr(user_preferences, attr)
        for attr in user_preferences.model_fields
        if isinstance(getattr(user_preferences, attr), PreferenceWeight)
        and getattr(user_preferences, attr).weight == 1.0
        and getattr(user_preferences, attr).value
    }
    
    def check_place_against_restrictions(place: Place) -> Tuple[bool, str]:
        invalid_reason = ""
        for pref, pref_weight in preferences_with_weight_one.items():
            match pref:
                # For party size, check for goodForGroups (parties of 6+)
                case "party_size" if pref_weight.value >= 6:
                    if not place.good_for_groups:
                        invalid_reason += f"This place has indicates they do not accomodate large groups (6 or more).\n"
                # For minimum number of ratings, check if the place has enough ratings
                case "desired_minimum_num_ratings":
                    if place.user_rating_count < pref_weight.value:
                        invalid_reason += f"This place has less than your desired {pref_weight.value} ratings.\n"
                # Can only check with servesVegetarianFood and if "vegan" is in primary_type_display_name_text
                case "dietary_requests" if "vegetarian" in pref_weight.value or "vegan" in pref_weight.value:
                    if "vegan" in pref_weight.value:
                        if "vegan" not in place.primary_type_display_name_text.lower():
                            invalid_reason += f"This place does not serve vegan food.\n"
                    if "vegetarian" in pref_weight.value:
                        if not place.serves_vegetarian_food:
                            invalid_reason += f"This place does not serve vegetarian food.\n"
                case "wants_family_friendly":
                    if not place.good_for_children:
                        invalid_reason += f"This place has not indicated itself as family friendly.\n"
                case "wants_childrens_menu":
                    if not place.menu_for_children:
                        invalid_reason += f"This place does not have a childrens menu.\n"
                # For parking, make sure there is at least 1 parking option that is free, if they need free parking (which is default)
                case "wants_free_parking":
                    parking_dict = {
                        attr: getattr(place.parking_options, attr)
                        for attr in place.parking_options.model_fields
                    }
                    has_free_option = any(parking_dict.values())
                    if not has_free_option:
                        invalid_reason += f"This place does not have a free parking option.\n"                
                case "wants_outdoor_seating":
                    if not place.outdoor_seating:
                        invalid_reason += f"This place does not have outdoor seating.\n"
                case "wants_live_music":
                    if not place.live_music:
                        invalid_reason += f"This place does not have live music.\n"
                case "wants_dessert":
                    if not place.serves_dessert:
                        invalid_reason += f"This place does not have dessert.\n"
                case "wants_beer":
                    if not place.serves_beer:
                        invalid_reason += f"This place does not have beer.\n"
                case "wants_wine":
                    if not place.serves_wine:
                        invalid_reason += f"This place does not have wine.\n"
                case "wants_brunch":
                    if not place.serves_brunch:
                        invalid_reason += f"This place does not have brunch.\n"
                case "wants_cocktails":
                    if not place.serves_cocktails:
                        invalid_reason += f"This place does not have cocktails.\n"
                case "wants_coffee":
                    if not place.serves_coffee:
                        invalid_reason += f"This place does not have coffee.\n"
        
        # For desired time and stay duration, check if the place is open at the desired timeframe
        user_desired_time_and_stay_duration = user_preferences.desired_time_and_stay_duration
        found_valid_period, suggestion_msg = check_if_user_stay_fits_open_hours(place, user_desired_time_and_stay_duration)
        if not found_valid_period:
            invalid_reason += suggestion_msg

        if invalid_reason == "":
            return (True, "")
        else:
            return (False, invalid_reason)
    
    valid_places = [] # list of Places
    invalid_places = [] # list of tuples of [Place, str]
    for place in places:
        is_valid_place, invalid_reason = check_place_against_restrictions(place)
        if is_valid_place:
            valid_places.append(place)
        else:
            invalid_places.append((place, invalid_reason))
        
    # Filter and rank places
    ranked_places = sorted(
        [(place, calculate_place_score(place, user_preferences)) for place in valid_places],
        key=lambda x: x[1],
        reverse=True
    )
    ranked_places = [place for place, _ in ranked_places]

    return ranked_places, invalid_places

def get_location_bias(user_coords: Tuple[float, float], preferred_direction: str, desired_max_distance_meters: float) -> Dict[str, Any]:
    """Get the locationBias parameter for the Google Maps places API.
    This assumes the user has opted in to location sharing.
    There are 2 types of location bias we can use: circle and rectangle.
    Example output for circle:
    {
        "circle": {
            "center": {
                "latitude": 37.7937,
                "longitude": -122.3965
            },
            "radius": 500.0
        }
    }
    Example output for rectangle:
    {
        "rectangle": {
            "low": { # southwest corner of the rectangle
                "latitude": 40.477398,
                "longitude": -74.259087
            },
            "high": { # northeast corner of the rectangle
                "latitude": 40.91618,
                "longitude": -73.70018
            }
        }
    }
    """
    location_bias = {}
    if preferred_direction != 'any':
        # TODO: Implement logic for preferred direction
        pass
    else:
        location_bias['circle'] = {
            'center': {
                'latitude': user_coords[0],
                'longitude': user_coords[1]
            },
            'radius': desired_max_distance_meters
        }
    return location_bias

def get_maps_text_search_parameters(state: Dict[str, Any]) -> Dict[str, Any]:
    """Process the current state and perform necessary computations,
    collecting the parameters to be passed into google_maps_text_search()"""
    api_optional_parameters = {}

    # If user allowed location sharing, then we have their coordinates
    if state['user_coordinates']:
        api_optional_parameters['locationBias'] = get_location_bias(
            state['user_coordinates'], 
            state['preferred_direction'], 
            state['desired_max_distance_meters']
        )

    # If user has a preferred price level, add that
    if state['preferred_price_level'] != "PRICE_LEVEL_UNSPECIFIED":
        api_optional_parameters['priceLevels'] = state['preferred_price_level']

    # If user has a desired star rating, add that
    if state['desired_star_rating'] > 0:
        api_optional_parameters['minRating'] = state['desired_star_rating']

    return api_optional_parameters

def get_places_from_json(json_response: Dict[str, Any]) -> List[Place]:
    #logging.debug(f"DEBUG: json_response: {json_response}")
    places = json_response['places']
    places_objects = []
    for p in places:
        places_objects.append(Place.model_validate(p))
    return places_objects

@tool(response_format="content_and_artifact")
def google_maps_text_search_and_filter(api_query: str, state: Annotated[dict, InjectedState]) -> Tuple[List[Place], List[Tuple[Place, str]]]:
    """A tool which can perform a text search, using Google's Places API"""
    
    # Collect the parameters for the API request
    optional_parameters = get_maps_text_search_parameters(state)
    api_parameters = {
        'textQuery': api_query,
        **optional_parameters
    }

    # Perform API request to get the places with the user's desired preferences
    url = 'https://places.googleapis.com/v1/places:searchText'
    headers = {
        'Content-Type': 'application/json',
        'X-Goog-Api-Key': os.environ['GOOGLE_MAPS_API_KEY'],
        'X-Goog-FieldMask': GOOGLE_FIELD_MASK
    }

    response = requests.post(url, headers=headers, json=api_parameters)
    places = get_places_from_json(response.json())

    try:
        valid_places, invalid_places = filter_places(places, state["user_preferences"])
        return f"Obtained {len(valid_places)} places and {len(invalid_places)} invalid places!", (valid_places, invalid_places)
    except Exception as e:
        return f"Failed to get places: {str(e)}", ([], [])
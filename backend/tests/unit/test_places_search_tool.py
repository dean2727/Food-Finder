import pytest
import math
import json
from datetime import datetime, timedelta
from typing import List, Tuple

from app.schemas.schema import Place, OpenClosePeriod, TimeInfo, ParkingOptions, RegularOpeningHours, Coordinates, UserPreferences, PreferenceWeight
from app.graph.tools import check_if_user_stay_fits_open_hours, calculate_place_score, filter_places

# read in test_text_search_json.txt
TEST_JSON = None
TEST_FILE_PATH = "../test_data/test_2.txt"
with open(TEST_FILE_PATH, "r") as file:
    TEST_JSON = json.load(file)
places_objects = []
for p in TEST_JSON['places']:
    places_objects.append(Place.model_validate(p))


# ~~~~~~~ Testing this function to see sample outputs and get suitable math going for rating score(non-pytest) ~~~~~~~
def calculate_rating_score(place_rating_count: int, user_preference_rating_count: int, weight_of_user_preference_rating_count: float) -> float:    
    if place_rating_count < user_preference_rating_count:
        raw_score = (place_rating_count / user_preference_rating_count)
        adjusted_score = raw_score * 0.2
        return adjusted_score * weight_of_user_preference_rating_count
    
    excess_ratings = place_rating_count - user_preference_rating_count
    max_excess = 1000
    
    rating_score = min(1, math.log(excess_ratings + 1) / math.log(max_excess + 1))
    
    return rating_score * weight_of_user_preference_rating_count

# print(calculate_rating_score(939, 500, 0.4)) # This is 0.35
# print(calculate_rating_score(939, 500, 0.9)) # This is 0.79
# print(calculate_rating_score(939, 900, 0.4)) # This is 0.21...meaning, it is more important for a place to have many more ratings than the user gives less weight to ratings
# print(calculate_rating_score(939, 1000, 0.8)) # This gives 0.15
# print(calculate_rating_score(939, 2000, 0.8)) # This gives 0.075

# ~~~~~~~~ Test check_if_user_stay_fits_open_hours ~~~~~~~~
# Helper function to create a Place object with given opening hours
def create_test_place(opening_hours):
    return Place(
        name="Test Place",
        types=["restaurant"],  # Added this field
        national_phone_number="(123) 456-7890",
        formatted_address="123 Test St, Test City, TS 12345, USA",
        location=Coordinates(latitude=37.7749, longitude=-122.4194),  # Added this field
        rating=4.5,
        google_maps_uri="https://maps.google.com/?cid=12345",  # Added this field
        website_uri="https://testplace.com",  # Added this field
        price_level="PRICE_LEVEL_MODERATE",
        user_rating_count=1000,
        regular_opening_hours=RegularOpeningHours(periods=opening_hours),
        display_name_text="Test Place",  # Added this field
        primary_type_display_name_text="Restaurant",  # Added this field
        reviews=[],  # Added this field, you might want to add some test reviews
        outdoor_seating=True,
        live_music=False,
        serves_dessert=True,
        serves_beer=True,
        serves_wine=True,
        serves_brunch=False,
        serves_cocktails=True,
        serves_coffee=True,
        serves_vegetarian_food=True,
        good_for_children=True,
        menu_for_children=False,
        good_for_groups=True,
        parking_options=ParkingOptions(
            free_parking_lot=True,
            paid_parking_lot=False,
            free_street_parking=True,
            paid_street_parking=False,
            valet_parking=False,
            free_garage_parking=False,
            paid_garage_parking=False
        )
    )

# Fixture for a standard place open from 9 AM to 5 PM every day
@pytest.fixture
def standard_place():
    opening_hours = [
        OpenClosePeriod(
            open=TimeInfo(day=i, hour=9, minute=0),
            close=TimeInfo(day=i, hour=17, minute=0)
        ) for i in range(7)
    ]
    return create_test_place(opening_hours)

def test_user_stay_within_opening_hours(standard_place):
    user_start = datetime(2024, 10, 10, 10, 0)  # Tuesday at 10 AM
    duration = 120  # 2 hours
    result, message = check_if_user_stay_fits_open_hours(standard_place, (user_start, duration))
    assert result == True
    assert message == ''

def test_user_arrival_before_opening(standard_place):
    user_start = datetime(2024, 10, 10, 8, 0)  # Tuesday at 8 AM
    duration = 120  # 2 hours
    result, message = check_if_user_stay_fits_open_hours(standard_place, (user_start, duration))
    assert result == False
    assert "This place will be closed when you want to visit" in message
    assert "The opening hours for this day are: 09:00 AM - 05:00 PM" in message
    assert "Please adjust your visit time accordingly" in message

def test_user_stay_past_closing(standard_place):
    user_start = datetime(2024, 10, 10, 16, 0)  # Tuesday at 4 PM
    duration = 120  # 2 hours
    result, message = check_if_user_stay_fits_open_hours(standard_place, (user_start, duration))
    assert result == False
    assert "before you finish your stay. Consider shortening" in message

def test_user_stay_exactly_matches_opening_hours(standard_place):
    user_start = datetime(2024, 10, 10, 9, 0)  # Tuesday at 9 AM
    duration = 480  # 8 hours
    result, message = check_if_user_stay_fits_open_hours(standard_place, (user_start, duration))
    assert result == True
    assert message == ''

def test_user_stay_entirely_past_closing(standard_place):
    user_start = datetime(2024, 10, 10, 18, 0)  # Tuesday at 6 PM
    duration = 120  # 2 hours
    result, message = check_if_user_stay_fits_open_hours(standard_place, (user_start, duration))
    assert result == False
    assert "This place will be closed when you want to visit" in message
    assert "The opening hours for this day are: 09:00 AM - 05:00 PM" in message
    assert "Please adjust your visit time accordingly" in message

# Test for a place with split hours (closed for lunch - 12 PM to 1 PM)
@pytest.fixture
def split_hours_place():
    opening_hours = [
        OpenClosePeriod(open=TimeInfo(day=i, hour=9, minute=0), close=TimeInfo(day=i, hour=12, minute=0))
        for i in range(7)
    ] + [
        OpenClosePeriod(open=TimeInfo(day=i, hour=13, minute=0), close=TimeInfo(day=i, hour=17, minute=0))
        for i in range(7)
    ]
    return create_test_place(opening_hours)

def test_user_stay_within_period_one(split_hours_place):
    user_start = datetime(2024, 10, 10, 11, 00)  # Tuesday at 11:00 AM
    duration = 60  # 1 hour
    result, message = check_if_user_stay_fits_open_hours(split_hours_place, (user_start, duration))
    assert result == True
    assert message == ''

def test_user_stay_within_period_two(split_hours_place):
    user_start = datetime(2024, 10, 10, 14, 00)  # Tuesday at 2:00 PM
    duration = 60  # 1 hour
    result, message = check_if_user_stay_fits_open_hours(split_hours_place, (user_start, duration))
    assert result == True
    assert message == ''

def test_user_stay_past_period_one(split_hours_place):
    user_start = datetime(2024, 10, 10, 11, 30)  # Tuesday at 11:30 AM
    duration = 60  # 1 hour
    result, message = check_if_user_stay_fits_open_hours(split_hours_place, (user_start, duration))
    assert result == False
    assert "This place will close at 12:00 PM before you finish your stay." in message
    assert "Consider shortening your visit or coming earlier." in message

def test_user_stay_past_period_two(split_hours_place):
    user_start = datetime(2024, 10, 10, 16, 30)  # Tuesday at 4:30 PM
    duration = 60  # 1 hour
    result, message = check_if_user_stay_fits_open_hours(split_hours_place, (user_start, duration))
    assert result == False
    assert "before you finish your stay. Consider shortening" in message

def test_user_stay_entirely_during_in_between(split_hours_place):
    user_start = datetime(2024, 10, 10, 12, 15)  # Tuesday at 12:15 PM
    duration = 30  # 30 minutes
    result, message = check_if_user_stay_fits_open_hours(split_hours_place, (user_start, duration))
    assert result == False
    assert "This place will be closed when you want to visit" in message
    assert "The opening hours for this day are: 01:00 PM - 05:00 PM, 09:00 AM - 12:00 PM" in message

# Fixture for a standard place open from 9 AM to 5 PM every day
@pytest.fixture
def late_night_place():
    opening_hours = [
        OpenClosePeriod(
            open=TimeInfo(day=i, hour=22, minute=0),
            close=TimeInfo(day=(i + 1) % 7, hour=2, minute=0)
        ) for i in range(7)
    ]
    return create_test_place(opening_hours)

def test_user_stay_fits_late_night_hours(late_night_place):
    user_start = datetime(2024, 10, 10, 23, 30)  # Thursday at 11:30 PM
    duration = 120  # 2 hours
    result, message = check_if_user_stay_fits_open_hours(late_night_place, (user_start, duration))
    assert result == True
    assert message == ''

def test_user_stay_past_late_night_closing(late_night_place):
    user_start = datetime(2024, 10, 11, 1, 30)  # Friday at 1:30 AM
    duration = 60  # 1 hour
    result, message = check_if_user_stay_fits_open_hours(late_night_place, (user_start, duration))
    assert result == False
    assert "This place will be closed when you want to visit" in message
    assert "The opening hours for this day are: 10:00 PM - 02:00 AM." in message
    assert "Please adjust your visit time accordingly." in message


# ~~~~~~~~ Test filter_places ~~~~~~~~
@pytest.fixture
def coffee_preference():
    return UserPreferences(
        wants_coffee=PreferenceWeight(value=True, weight=1.0),  # Restriction
        # Restriction - 4pm, Thursday
        desired_time_and_stay_duration=(datetime(2024, 10, 10, 16, 0), 60)
    )

# Define multiple scenarios
coffee_scenarios = [
    {
        "name": "High rated coffee places",
        "min_ratings": 3000,
        # These are found out by running print_places.py
        "ranked": [
            "Loro Asian Smokehouse & Bar",
            "888 Pan Asian Restaurant",
            "1618 Asian Fusion",
            "Uchiko Austin",
            "JOI Asian Bistro",
            "The Pho Asian Fusion"
        ],
        "invalid": [
            "HAHA KITCHEN",
            "Lin Asian Bar + Dim Sum",
            "Pinch",
            "Old Thousand",
            "Shu Shu's Asian Cuisine",
            "QI Austin: Modern Asian Kitchen",
            "WU Chow - Downtown",
            "Bento Teppanyaki Asian Cuisine",
            "Bamboo House Austin",
            "Titaya's Thai Cuisine",
            "WU Chow - North Lamar",
            "P Thai's Khao Man Gai & Noodles",
            "Chi'Lantro",
            "Ling Wu Asian Restaurant at The Grove",
        ]
    },
]

@pytest.mark.parametrize("scenario", coffee_scenarios)
def test_filter_places_coffee(coffee_preference, scenario):
    # Update the preference with the scenario's minimum ratings
    coffee_preference.desired_minimum_num_ratings = PreferenceWeight(value=scenario["min_ratings"], weight=0.9)

    #qi_austin = next((p for p in places_objects if p.display_name_text == "QI Austin: Modern Asian Kitchen"), None)
    #print(f"DEBUG: confirming QI Austin: Modern Asian Kitchen doesnt serve coffee: {qi_austin.serves_coffee}")

    ranked_places, invalid_places = filter_places(places_objects, coffee_preference)
    
    ranked_places_display_names = [place.display_name_text for place in ranked_places]
    invalid_places_display_names = [place[0].display_name_text for place in invalid_places]
    
    print(f"DEBUG: Scenario: {scenario['name']}")
    # use sets to give us the difference
    #ranked_places_display_names_set = set(ranked_places_display_names)
    #expected_ranked_places_display_names_set = set(scenario["ranked"])
    #invalid_places_display_names_set = set(invalid_places_display_names)
    #expected_invalid_places_display_names_set = set(scenario["invalid"])

    # print whats in each individual set
    #print("DEBUG: ranked_places_display_names_set: ", ranked_places_display_names_set)
    #print("DEBUG: expected_ranked_places_display_names_set: ", expected_ranked_places_display_names_set)
    #print("DEBUG: invalid_places_display_names_set: ", invalid_places_display_names_set)
    #print("DEBUG: expected_invalid_places_display_names_set: ", expected_invalid_places_display_names_set)
    
    #print("DEBUG: places in expected but not in ranked: ", ranked_places_display_names_set - expected_ranked_places_display_names_set)
    #print("DEBUG: places in ranked but not in expected: ", expected_ranked_places_display_names_set - ranked_places_display_names_set)
    #print("DEBUG: places in expected but not in invalid: ", expected_invalid_places_display_names_set - invalid_places_display_names_set)
    #print("DEBUG: places in invalid but not in expected: ", invalid_places_display_names_set - expected_invalid_places_display_names_set)
    print("DEBUG: ranked_places_display_names: ", ranked_places_display_names)
    print("DEBUG scenario['ranked']: ", scenario["ranked"])
    assert ranked_places_display_names == scenario["ranked"], f"Ranked places mismatch in scenario: {scenario['name']}"
    assert invalid_places_display_names == scenario["invalid"], f"Invalid places mismatch in scenario: {scenario['name']}"
# This file is used primarily to help inform what tests to use in the unit tests (expected results)

import json
from datetime import datetime, time, timedelta
from typing import List, Tuple
from app.schemas import Place
from app.graph.tools.places_search import check_if_user_stay_fits_open_hours

# read in test_text_search_json.txt
TEST_JSON = None
TEST_FILE_PATH = "../test_data/test_2.txt"
with open(TEST_FILE_PATH, "r") as file:
    TEST_JSON = json.load(file)
places_objects = []
for p in TEST_JSON['places']:
    places_objects.append(Place.model_validate(p))

# This corresponds to the pytest test in question
RESTRICTIONS = [
    "serves_coffee"
]

# Oct 10, 2024, 5pm, 1hr (this way, we arent testing with datetime.now(), hence different places in result)
DESIRED_TIME_AND_STAY_DURATION = (datetime(datetime.now().year, 10, 10, 16, 0), 60)
#DESIRED_TIME_AND_STAY_DURATION = (datetime(datetime.now().year, 10, 10, 22, 0), 60)

valid_places = [p for p in places_objects if all(getattr(p, r) for r in RESTRICTIONS)]
valid_places = [p for p in valid_places if check_if_user_stay_fits_open_hours(p, DESIRED_TIME_AND_STAY_DURATION)[0]]
invalid_places = [p for p in places_objects if p not in valid_places]

def print_places(places_to_print: List[Place], val: str):
    print(f"Places {val} these restrictions - {', '.join(RESTRICTIONS)}:")
    for p in places_to_print:
        # print hours for day 4, in datetime
        DAY = 4
        datetimes = []
        for period in p.regular_opening_hours.periods:
            if period.open.day == DAY:
                dt_open = datetime(datetime.now().year, datetime.now().month, datetime.now().day, period.open.hour, period.open.minute)
                dt_close = datetime(datetime.now().year, datetime.now().month, datetime.now().day, period.close.hour, period.close.minute)
                datetimes.append(f"{dt_open.strftime('%I:%M %p')} - {dt_close.strftime('%I:%M %p')}")

        print(f"{p.display_name_text} - {', '.join(datetimes)} - {p.user_rating_count}")

# Assuming test just involves 1 preference
valid_places.sort(key=lambda x: x.user_rating_count, reverse=True)
print_places(valid_places, "that are valid for")
print()

print_places(invalid_places, "that are not valid for")




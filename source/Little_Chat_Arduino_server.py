"""
Little_Chat_Arduino is a simple AI Agent chatbot that lets you
chat with your LLM but also offers an Arduino agent.
With the Arduino agent you can ask your LLM to control your
Arduino device and do things in the real world.
The Arduino agent builds your sketch and runs it, and lets your
LLM know if something went wrong — e.g. device not connected or
device errors.

This module is the MCP tool server: a small FastAPI app that
exposes datetime / weather / calc / Arduino-control tools as
simple HTTP GET endpoints for the agent (little_Chat_Arduino.py)
to call.

Supports triple provider mode (handled on the agent side):
  - Local Ollama LLM   (default, no API key needed)
  - Network Ollama LLM (LAN server, e.g. 192.168.1.58:11434)
  - Anthropic Claude   (requires --api-key)
"""
VERSION = "0.1.0 triple-provider - text/graph option"

import os
import glob
import subprocess
import requests
from fastapi import FastAPI, HTTPException, Query
from datetime import datetime
import pytz
from timezonefinder import TimezoneFinder
from geopy.geocoders import Nominatim
from dotenv import load_dotenv
import uvicorn

# Load environment variables from .env file (e.g. OPENWEATHER_API_KEY)
load_dotenv()

# Initialize the FastAPI application
app = FastAPI(
    title="MCP Tool Server",
    description="Provides tools like weather and datetime as a service.",
    version="0.1"
)


# =================================================================
# Tool Functions (business logic)
# =================================================================

def get_date_time(city: str):
    """Gets the current date and time for a given city."""
    try:
        geolocator = Nominatim(user_agent="mcp_datetime_app")
        location = geolocator.geocode(city, timeout=10)
        if location is None:
            raise ValueError(f'City "{city}" not found.')

        tf = TimezoneFinder()
        timezone_str = tf.timezone_at(lat=location.latitude, lng=location.longitude)
        if timezone_str is None:
            raise ValueError(f'Could not determine timezone for {city}.')

        tz = pytz.timezone(timezone_str)
        current_time = datetime.now(tz)
        city_name = location.address.split(',')[0]

        return {
            'city': city_name,
            'timezone': timezone_str,
            'datetime': current_time.strftime('%Y-%m-%d %H:%M:%S'),
            'day_of_week': current_time.strftime('%A'),
        }
    except Exception as e:
        # Return a dictionary that can be converted to JSON, even for errors
        return {'error': str(e)}


def get_weather(city: str):
    """Gets the current weather for a given city."""
    base_url = "http://api.openweathermap.org/data/2.5/weather"
    api_key = os.getenv('OPENWEATHER_API_KEY')
    if not api_key:
        return {'error': 'OpenWeather API key is not set.'}

    params = {"q": city, "appid": api_key, "units": "metric"}

    try:
        response = requests.get(base_url, params=params)
        response.raise_for_status()
        return response.json()
    except requests.exceptions.RequestException as e:
        # Return a dictionary for errors
        return {'error': f'Failed to fetch weather: {e}'}


def get_Calc(l_operation: str):
    """Performs a basic arithmetic operation.

    Expects a string like "ADD, 2, 3" / "SUB, 2, 3" / "MUL, 2, 3" / "DIV, 2, 3".

    NOTE: unlike get_date_time()/get_weather(), this returns a plain
    string (not a dict), on both success and error (e.g. "Error: ...").
    The endpoint checks for that "Error" prefix directly rather than
    treating the return value as a dict.
    """
    OPERATION, NUM_ONE, NUM_TWO = l_operation.split(", ")
    try:
        # Convert string inputs to floating-point numbers for calculation
        num1_float = float(NUM_ONE)
        num2_float = float(NUM_TWO)
    except ValueError:
        return "Error: NUM_ONE and NUM_TWO must be valid numbers."

    result = None
    if OPERATION == "ADD":
        result = num1_float + num2_float
    elif OPERATION == "SUB":
        result = num1_float - num2_float
    elif OPERATION == "MUL":
        result = num1_float * num2_float
    elif OPERATION == "DIV":
        if num2_float == 0:
            return "Error: Division by zero is not allowed."
        result = num1_float / num2_float
    else:
        return "Error: Invalid operation. Please use 'ADD', 'SUB', 'MUL', or 'DIV'."
    # Convert the numerical result back to a string
    return str(result)


# =================================================================
# Arduino sketch build/upload pipeline
# =================================================================

def save_with_autoincrement(data: str, prefix: str = "Arduino_source_", directory: str = "./Arduino_source") -> str:
    """
    Save data to the next available numbered file.
    Returns the path of the file written.
    """
    # Find all existing files matching the pattern
    pattern = os.path.join(directory, f"{prefix}*.ino")
    existing = glob.glob(pattern)

    # Extract the numbers from existing filenames
    next_num = 1
    if existing:
        nums = []
        for path in existing:
            basename = os.path.splitext(os.path.basename(path))[0]  # e.g. "filename007"
            suffix = basename[len(prefix):]                          # e.g. "007"
            if suffix.isdigit():
                nums.append(int(suffix))
        if nums:
            next_num = max(nums) + 1

    # Build the new filename with zero-padded number
    filename = os.path.join(directory, f"{prefix}{next_num:03d}.ino")

    with open(filename, "w") as f:
        f.write(data)

    return filename


def Exec_Arduino_Code(l_code: str):
    """Saves, creates, compiles and uploads an Arduino sketch from source code.

    Runs the full arduino-cli pipeline: new sketch -> write .ino ->
    compile -> upload. Returns "Okay Completed" on success, or an
    "Error OS command: ..." string as soon as any step fails.
    """
    save_with_autoincrement(l_code)

    sk_name = "Arduino_sketch"

    cmd1 = ["rm", "-rf", sk_name]
    result = subprocess.run(cmd1, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error OS command: " + result.stderr)
        return "Error OS command: " + result.stderr
    else:
        print("Okay: " + result.stdout)

    cmd1 = ["arduino-cli", "sketch", "new", sk_name]
    result = subprocess.run(cmd1, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error OS command: " + result.stderr)
        return "Error OS command: " + result.stderr
    else:
        print("Okay: " + result.stdout)

    code_name = sk_name + "/" + sk_name + ".ino"
    with open(code_name, "w") as f:
        print(l_code, file=f)

    cmd1 = ["arduino-cli", "compile", "--fqbn", "arduino:avr:uno", sk_name]
    result = subprocess.run(cmd1, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error OS command: " + result.stderr)
        return "Error OS command: " + result.stderr
    else:
        print("Okay: " + result.stdout)

    cmd1 = ["arduino-cli", "upload", "-p", "/dev/ttyACM0", "--fqbn", "arduino:avr:uno", sk_name]
    result = subprocess.run(cmd1, capture_output=True, text=True)
    if result.returncode != 0:
        print("Error OS command: " + result.stderr)
        return "Error OS command: " + result.stderr
    else:
        print("Okay: " + result.stdout)

    return "Okay Completed"


# =================================================================
# API Endpoints
# =================================================================

@app.get("/")
def read_root():
    """A simple endpoint to check if the server is running."""
    return {"status": "MCP Server is running"}


@app.get("/get_datetime")
def api_get_datetime(
        myParam: str = Query(..., description="The city to get the date and time for, e.g., 'Paris, France'")):
    """API endpoint to get the current date and time."""
    result = get_date_time(myParam)
    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])
    return result


@app.get("/get_weather")
def api_get_weather(myParam: str = Query(..., description="The city to get the weather for, e.g., 'London, UK'")):
    """API endpoint to get the current weather."""
    result = get_weather(myParam)
    if "error" in result:
        # OpenWeatherMap itself returns a 2-layer error shape on failure
        # (top-level "cod"/"message"), which we surface here if present.
        if "cod" in result and result["cod"] != 200:
            raise HTTPException(status_code=int(result["cod"]), detail=result.get("message", "Weather API error"))
        raise HTTPException(status_code=500, detail=result["error"])
    return result


@app.get("/get_calc")
def api_get_calc(myParam: str = Query(..., description="The calc operation, e.g., 'ADD, 2, 3' 'SUB, 2, 3'")):
    """API endpoint to get the current calc."""
    result = get_Calc(myParam)
    # get_Calc() returns a plain string: either the numeric result, or
    # an "Error: ..." message. It is NOT a dict, so it must be checked
    # and passed as a string — do not index it with ["error"].
    if result.lower().startswith("error"):
        raise HTTPException(status_code=500, detail=result)
    return result


@app.get("/run_Arduino")
def api_run_Arduino(myParam: str = Query(..., description="Compile and running Aruino sketch")):
    """API endpoint to run anything you can do using an Arduino device."""

    ### DEBUG
    print(f"\n\nDEBUG: {myParam}")

    result = Exec_Arduino_Code(myParam)

    # Exec_Arduino_Code() returns a plain string: either "Okay Completed"
    # or an "Error OS command: ..." message. It is NOT a dict, so it must
    # be checked and passed as a string — do not index it with ["error"].
    if result.lower().startswith("error"):
        raise HTTPException(status_code=500, detail=result)
    return result


# --- Main entry point to run the server ---
if __name__ == "__main__":
    print("Starting MCP Server ...")
    uvicorn.run(app, host="127.0.0.1", port=8000)
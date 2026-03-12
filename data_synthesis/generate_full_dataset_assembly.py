import requests
import json
import random
import re
import time
import os
import itertools
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# ================= 1. Base Configuration =================
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if not API_KEY:
    raise ValueError("Please set GEMINI_API_KEY in environment or .env before running.")
ENDPOINT = os.environ.get("GEMINI_ENDPOINT", "https://example.googleapis.com/v1:generateContent")

OUTPUT_DIR = "assembly_synthesis"
if not os.path.exists(OUTPUT_DIR):
    os.makedirs(OUTPUT_DIR)

MAX_WORKERS = 10     
MAX_RETRIES = 3

# ================= 2. Toy Database =================
# Logic: N Toys * 10 Combinations = task count
TOY_DATABASE = {
    "Tracked Excavator (a01)": ["Excavator Arm", "Track", "Screw", "Screwdriver", "Chassis Parts", "Rear Body", "Machine", "Bucket", "Cabin", "Figurine", "Chassis"],
    "Bulldozer (a02)": ["Bulldozer Arm", "Screwdriver", "Figurine", "Wheel", "Screw", "Cabin", "Front Body", "Mechanical Arm", "Bucket", "Chassis Parts", "Chassis"],
    "Bulldozer (a03)": ["Screwdriver", "Track", "Screw", "Chassis", "Front Body", "Cabin", "Dozer Blade", "Roof", "Figurine", "Chassis Parts"],
    "Crane (a06)": ["Crane Arm", "Roof", "Engine Cover", "Screwdriver", "Nut", "Rear Base", "Mechanical Arm", "Rear Body", "Engine Chassis Cover", "Clamp", "Screw", "Track", "Wheel", "Cabin", "Clamp Arm"],
    "Bulldozer (a07)": ["Chassis", "Rear Bumper", "Front Body", "Roof", "Step", "Interior", "Cabin", "Mechanical Arm", "Wheel", "Rear Body", "Nut", "Screwdriver", "Bulldozer Arm", "Track", "Screw", "Bucket"],
    "Crane (a08)": ["Chassis", "Cabin", "Screwdriver", "Screwdriver Nut", "Wheel", "Screw", "Front Bumper", "Nut", "Roof", "Grill", "Crane Arm", "Turntable Base", "Rear Base"],
    "Crane (a09)": ["Screw", "Turntable", "Cabin", "Rear Body", "Crane Arm", "Clamp Arm", "Interior", "Wheel", "Clamp", "Arm", "Nut", "Turntable Top", "Screwdriver Nut", "Chassis", "Screwdriver", "Screwdriver Bit"],
    "Garbage Truck (a10)": ["Screwdriver", "Rear Base", "Wheel", "Garbage Container", "Connector", "Hub", "Rear Cabin", "Roof", "Nut", "Screw", "Interior", "Screwdriver Nut", "Windshield", "Chassis", "Side Door", "Step", "Cabin"],
    "Bulldozer (a11)": ["Cabin", "Exhaust Pipe", "Bucket", "Mechanical Arm", "Rear Body", "Screw", "Wheel", "Chassis Parts", "Bulldozer Arm", "Screwdriver", "Chassis"],
    "Dump Truck (a12)": ["Front Bumper", "Screw", "Wheel", "Nut", "Dump Bed", "Chassis"],
    "Dump Truck (a13)": ["Cylinder", "Rear Base", "Wheel", "Chassis", "Dump Bed", "Screw", "Cabin", "Front Bumper", "Screwdriver", "Interior", "Light", "Roof"],
    "Transport Truck (a14)": ["Cockpit", "Transport Cabin", "Rear Door", "Rear Roof", "Rear Base", "Wheel", "Screwdriver", "Screw", "Interior", "Chassis", "Front Bumper", "Roof", "Sound Module", "Light"],
    "Garbage Truck (a15)": ["Screwdriver", "Chassis", "Chassis Parts", "Wheel", "Screw", "Garbage Container", "Front Bumper", "Cabin"],
    "Aerial Platform Truck (a16)": ["Roof", "Front Bumper", "Screw", "Rear Base", "Screwdriver", "Wheel", "Chassis", "Basket", "Ladder", "Interior", "Body", "Rear Car", "Ladder Basket", "Turntable Top", "Cabin"],
    "Water Tank Fire Truck (a17)": ["Water Tank", "Front Bumper", "Fire Extinguisher", "Roof", "Screw", "Screwdriver", "Wheel", "Sound Module", "Chassis", "Cabin", "Rear Base", "Interior", "Turntable Top"],
    "Water Tank Fire Truck (a18)": ["Fire Extinguisher", "Side Door", "Screwdriver", "Chassis", "Water Tank", "Wheel", "Front Bumper", "Cabin", "Screw", "Hub"],
    "Aerial Platform Truck (a19)": ["Ladder Basket", "Chassis Parts", "Chassis", "Screwdriver", "Ladder", "Rear Body", "Wheel", "Cabin", "Basket", "Front Bumper", "Screw"],
    "Water Tank Fire Truck (a20)": ["Chassis", "Fire Extinguisher", "Rear Roof", "Rear Door", "Screwdriver", "Interior", "Figurine", "Water Tank Parts", "Fire Equipment", "Screw", "Cabin", "Water Tank", "Wheel", "Connector"],
    "Dump Truck (a21)": ["Wheel", "Chassis", "Screwdriver", "Front Bumper", "Cabin", "Screw", "Dump Bed", "Rear Base", "Chassis Parts"],
    "Sports Car (a23)": ["Engine Cover", "Roof", "Screwdriver", "Front Bumper", "Spoiler", "Rocker Panel", "Body", "Screw", "Engine", "Wheel", "Chassis"],
    "Sports Car (a24)": ["Wheel", "Screwdriver", "Roof", "Engine", "Chassis", "Front Bumper", "Rocker Panel", "Interior", "Engine Cover", "Spoiler", "Front Body", "Rear Body", "Screw"],
    "Unknown Vehicle (a26)": ["Windshield", "Wheel", "Screwdriver Nut", "Screwdriver", "Screw", "Screwdriver Bit", "Cover", "Chassis", "Body", "Interior", "Front Bumper"],
    "SUV (a27)": ["Body", "Roof", "Rear Seat", "Screw", "Interior", "Side Door", "Chassis", "Nut", "Wheel", "Dashboard", "Spare Tire", "Light", "Screwdriver"],
    "Sports Car (a28)": ["Wheel", "Screw", "Rear Roof", "Screwdriver Bit", "Nut", "Front Bumper", "Chassis", "Spare Tire", "Engine Cover", "Body", "Interior"],
    "SUV (a29)": ["Wheel", "Screwdriver", "Screw", "Rear Seat", "Roof", "Side Door", "Body", "Chassis", "Spare Tire", "Dashboard", "Interior"],
    "SUV (a30)": ["Wheel", "Screw", "Screwdriver", "Nut", "Roof", "Side Door", "Spare Tire", "Rear Seat", "Dashboard", "Interior", "Body", "Chassis"],
    "Sports Car (a31)": ["Screwdriver", "Battery", "Rear Bumper", "Screw", "Rear Body", "Front Body", "Body", "Wheel", "Chassis", "Spoiler", "Interior", "Roof"],
    "Bulldozer (b01a)": ["Screwdriver", "Nut", "Chassis", "Internal Structure", "Screw", "Wheel", "Arm Connector", "Roof", "Cabin", "Mechanical Arm", "Bulldozer Arm", "Bucket"],
    "Road Roller (b01b)": ["Mechanical Arm", "Push Frame", "Roller", "Roller Parts", "Cockpit", "Arm Connector", "Nut", "Wheel", "Screw", "Roof", "Chassis", "Interior", "Roller Arm"],
    "Dump Truck (b02a)": ["Wheel", "Screwdriver", "Cabin", "Screw", "Chassis", "Dump Bed", "Roof", "Rear Base"],
    "Sports Car (b02b)": ["Track", "Cabin", "Chassis", "Screw", "Interior", "Screwdriver", "Engine Cover", "Boom Parts", "Arm Parts", "Rear Body", "Clamp", "Clamp Arm"],
    "Aerial Platform Truck (b03a)": ["Screwdriver", "Interior", "Front Bumper", "Ladder", "Cabin", "Roof", "Turntable Top", "Screw", "Rear Base", "Chassis", "Ladder Basket", "Basket", "Wheel"],
    "Water Tank Fire Truck (b03b)": ["Fire Extinguisher", "Screwdriver", "Water Tank", "Interior", "Screw", "Cabin", "Roof", "Front Bumper", "Rear Base", "Wheel", "Chassis"],
    "Wheeled Excavator (b04a)": ["Main Arm", "Interior", "Rear Body", "Boom", "Screwdriver", "Roof", "Cabin", "Excavator Arm", "Wheel", "Chassis", "Rear Bumper", "Screw", "Arm Parts", "Bucket", "Arm Connector"],
    "Bulldozer (b04b)": ["Screw", "Screwdriver", "Wheel", "Chassis", "Rear Body", "Arm Connector", "Cabin", "Rear Bumper", "Mechanical Arm", "Bucket", "Interior", "Bulldozer Arm", "Roof"],
    "Jackhammer Truck (b04c)": ["Rear Body", "Boom", "Mechanical Arm", "Chassis", "Arm Connector", "Arm Parts", "Cockpit", "Wheel", "Interior", "Screw", "Jackhammer", "Screwdriver", "Jackhammer Arm", "Roof", "Rear Bumper"],
    "Road Roller (b04d)": ["Road Roller Parts", "Road Roller Wheel", "Screw", "Wheel", "Rear", "Chassis", "Arm Connector", "Push Frame", "Drive", "Interior", "Rear Bumper", "Screwdriver", "Roof", "Roller Arm"],
    "Wheeled Excavator (b05a)": ["Small Arm", "Main Arm", "Bucket", "Arm Parts", "Main Arm Parts", "Turntable Top Cover", "Turntable Base", "Screwdriver", "Wrench", "Interior", "Screw", "Rear Bumper", "Cabin", "Front Bumper", "Wheel", "Screwdriver Nut", "Chassis", "Nut", "Excavator Arm"],
    "Crane (b05b)": ["Wrench", "Arm Parts", "Hook", "Wheel", "Nut", "Cabin", "Front Bumper", "Lifting Arm", "Arm", "Crane Arm", "Interior", "Rear Bumper", "Chassis", "Screwdriver", "Crane Arm Parts", "Turntable Top", "Turntable Base", "Screw"],
    "Cement Mixer (b05c)": ["Cabin", "Front Bumper", "Screw", "Mixing Bucket", "Rear Bumper", "Mixing Bucket Stand", "Wheel", "Nut", "Chassis", "Interior", "Screwdriver", "Mixing Bucket Parts", "Screwdriver Nut"],
    "Dump Truck (b05d)": ["Screwdriver", "Chassis", "Rear Base", "Rear Bumper", "Nut", "Front Bumper", "Cover", "Wheel", "Dump Bed", "Screw", "Drive", "Interior", "Wrench"],
    "Dump Truck (b06a)": ["Wheel", "Screw", "Cabin", "Dump Bed", "Tilter", "Interior", "Nut", "Chassis", "Screwdriver"],
    "Crane (b06b)": ["Turntable Top", "Screwdriver", "Wheel", "Interior", "Chassis", "Cabin", "Arm", "Screw", "Hook", "Nut", "Lifting Arm"],
    "Wheeled Excavator (b06c)": ["Boom", "Screwdriver", "Bucket", "Turntable Base", "Cabin", "Nut", "Screw", "Wheel", "Interior", "Chassis", "Excavator Arm"],
    "Cement Mixer (b06d)": ["Mixer Stand", "Mixer Parts", "Screwdriver", "Cabin", "Nut", "Tilter", "Wheel", "Screw", "Chassis", "Mixing Bucket", "Interior"],
    "Sports Car (b08a)": ["Clamp", "Boom", "Mechanical Arm", "Screwdriver", "Screw", "Interior", "Track", "Rear Body", "Clamp Arm", "Engine Cover", "Chassis", "Cabin", "Roof", "Arm Parts"],
    "Bulldozer (b08b)": ["Dozer Blade", "Screwdriver", "Chassis", "Screw", "Track", "Mechanical Arm", "Rear Base", "Roof", "Cabin", "Interior", "Front Body"],
    "Bulldozer (b08c)": ["Screwdriver", "Arm Connector", "Cabin", "Chassis", "Rear Base", "Roof", "Mechanical Arm", "Bucket", "Internal Structure", "Nut", "Bulldozer Arm", "Wheel", "Screw"],
    "Dump Truck (b08d)": ["Wheel", "Screwdriver", "Front Body", "Grill", "Chassis", "Rear Base", "Cabin", "Screw", "Dump Bed"],
    "Wheeled Excavator (c01a)": ["Cabin Window", "Mechanical Arm", "Arm Parts", "Bucket", "Turntable Base", "Rear Base", "Wheel", "Chassis", "Excavator Arm", "Screwdriver", "Roof", "Interior", "Screw", "Front Bumper", "Cabin"],
    "Dump Truck (c01b)": ["Screwdriver", "Wheel", "Chassis", "Front Bumper", "Wrench", "Screw", "Dump Bed", "Roof", "Interior", "Cabin", "Rear Base"],
    "Crane (c01c)": ["Lifting Arm", "Front Bumper", "Roof", "Chassis", "Cabin Window", "Cabin", "Wheel", "Screw", "Rear Base", "Interior", "Screwdriver", "Turntable Base", "Crane Arm Parts", "Hook", "Arm Parts", "Mechanical Arm", "Crane Arm"],
    "Cement Mixer (c01d)": ["Mixing Bucket", "Rear Base", "Front Bumper", "Roof", "Screwdriver", "Mixing Bucket Stand", "Interior", "Cabin", "Screw", "Chassis", "Mixing Bucket Parts", "Wheel"],
    "Wheeled Excavator (c02a)": ["Screwdriver", "Arm Connector", "Interior", "Sound Module", "Nut", "Wheel", "Chassis", "Rear Body", "Excavator Arm", "Roof", "Screw", "Cabin", "Boom", "Stick", "Bucket", "Rear Bumper"],
    "Bulldozer (c02b)": ["Roof", "Bucket", "Arm", "Rear Bumper", "Chassis", "Wheel", "Interior", "Rear Body", "Cabin", "Bulldozer Arm", "Screwdriver", "Screw", "Arm Connector", "Sound Module"],
    "Road Roller (c02c)": ["Screwdriver", "Roof", "Rear Body", "Rear Bumper", "Cockpit", "Front Body", "Interior", "Arm Connector", "Sound Module", "Wheel", "Screw", "Chassis", "Push Frame", "Roller Arm", "Roller"],
    "Wheeled Excavator (c03a)": ["Wheel", "Screwdriver", "Screw", "Arm Connector", "Chassis Parts", "Excavator Arm", "Bucket", "Rear Body", "Cabin", "Chassis", "Cabin Window"],
    "Bulldozer (c03b)": ["Screwdriver", "Bucket", "Cabin", "Cabin Window", "Chassis", "Arm Connector", "Rear Body", "Wheel", "Screw", "Chassis Parts", "Bulldozer Arm"],
    "Cement Mixer (c03c)": ["Screwdriver", "Chassis Parts", "Mixing Bucket", "Cabin", "Grill", "Front Body", "Screw", "Chassis", "Mixing Bucket Stand", "Wheel"],
    "Dump Truck (c03d)": ["Cabin", "Grill", "Wheel", "Screw", "Dump Bed", "Front Body", "Chassis Parts", "Screwdriver", "Chassis"],
    "Jackhammer Truck (c03e)": ["Cabin Window", "Cabin", "Wheel", "Jackhammer", "Chassis Parts", "Boom", "Chassis", "Mechanical Arm", "Rear Body", "Screwdriver", "Arm Connector", "Screw", "Jackhammer Arm"],
    "Road Roller (c03f)": ["Screw", "Wheel", "Mechanical Arm", "Roller", "Arm Connector", "Cabin", "Cabin Window", "Rear", "Screwdriver", "Roller Arm", "Chassis", "Chassis Parts"],
    "Wheeled Excavator (c04a)": ["Main Arm", "Cabin", "Rear Bumper", "Interior", "Boom", "Bucket", "Roof", "Boom Connector", "Chassis", "Screw", "Screwdriver", "Nut", "Excavator Arm", "Wheel"],
    "Jackhammer Truck (c04b)": ["Wheel", "Chassis", "Arm Connector", "Rear Bumper", "Cabin", "Screw", "Jackhammer", "Nut", "Boom", "Mechanical Arm", "Interior", "Roof", "Jackhammer Arm", "Screwdriver"],
    "Road Roller (c04c)": ["Push Frame", "Wheel", "Roller", "Screwdriver", "Screw", "Cockpit", "Interior", "Rear Bumper", "Arm Connector", "Roof", "Chassis", "Roller Arm"],
    "Bulldozer (c04d)": ["Mechanical Arm Connector", "Rear Bumper", "Roof", "Mechanical Arm", "Screw", "Bucket", "Screwdriver", "Chassis", "Wheel", "Interior", "Rear Body", "Drive", "Bulldozer Arm"],
    "Wheeled Excavator (c05a)": ["Bucket", "Boom Parts", "Screw", "Screwdriver", "Chassis", "Rear Bumper", "Rear Body", "Interior", "Cabin", "Front Body", "Roof", "Boom", "Arm", "Wheel", "Excavator Arm", "Arm Parts"],
    "Bulldozer (c05b)": ["Front Body", "Screw", "Chassis", "Screwdriver", "Cabin", "Rear Bumper", "Bucket", "Roof", "Mechanical Arm", "Wheel", "Interior", "Bulldozer Arm", "Rear Body"],
    "Dump Truck (c06a)": ["Wheel", "Roof", "Rear Base", "Sound Module", "Dump Bed", "Screw", "Screwdriver", "Interior", "Chassis", "Light", "Cabin", "Cover"],
    "Cement Mixer (c06b)": ["Wheel", "Screwdriver", "Interior", "Mixing Bucket Stand", "Roof", "Mixing Bucket", "Light", "Sound Module", "Chassis", "Screw", "Cabin"],
    "Crane (c06c)": ["Wheel", "Roof", "Interior", "Cabin", "Turntable Base", "Sound Module", "Screwdriver", "Light", "Chassis", "Rear Base", "Screw", "Crane Arm", "Hook"],
    "Water Tank Fire Truck (c06d)": ["Water Tank Parts", "Rear Base", "Interior", "Wheel", "Screwdriver", "Chassis", "Drive", "Light", "Sound Module", "Roof", "Screw", "Water Tank"],
    "Wheeled Excavator (c06e)": ["Chassis", "Wheel", "Screw", "Turntable Base", "Rear Base", "Screwdriver", "Excavator Arm", "Roof", "Light", "Cabin", "Interior", "Sound Module", "Boom", "Bucket", "Main Arm"],
    "Aerial Platform Truck (c06f)": ["Ladder Basket", "Turntable Base", "Rear", "Chassis", "Interior", "Screwdriver", "Light", "Sound Module", "Screw", "Cabin", "Roof", "Wheel"],
    "Wheeled Excavator (c07a)": ["Rear Body", "Rear Bumper", "Cabin", "Roof", "Chassis", "Interior", "Excavator Arm", "Boom", "Main Arm", "Screw", "Arm Connector", "Bucket", "Screwdriver", "Wheel"],
    "Aerial Platform Truck (c07b)": ["Ladder Parts", "Rear Body", "Rear Bottom", "Chassis", "Basket", "Ladder", "Turntable Top", "Wheel", "Front Bumper", "Interior", "Screw", "Cabin", "Ladder Basket", "Screwdriver", "Roof"],
    "Garbage Truck (c07c)": ["Roof", "Screwdriver", "Screw", "Interior", "Wheel", "Chassis", "Rear Roof", "Cover", "Front Bumper", "Cabin", "Garbage Container"],
    "Crane (c08a)": ["Roof", "Arm", "Screwdriver", "Wheel", "Cabin", "Front Bumper", "Rear Body", "Turntable Top", "Chassis", "Interior", "Arm Parts", "Screw", "Crane Arm", "Light", "Rear Base", "Crane Arm Parts", "Sound Module", "Lifting Arm", "Hook"],
    "Garbage Truck (c08b)": ["Rear Roof", "Screw", "Sound Module", "Screwdriver", "Front Bumper", "Garbage Container", "Wheel", "Interior", "Chassis", "Cabin", "Garbage Container Parts", "Cover", "Roof"],
    "Transport Truck (c08c)": ["Transport Cabin", "Rear Bottom", "Rear Roof", "Wheel", "Screwdriver", "Chassis", "Cockpit", "Sound Module", "Interior", "Screw", "Front Bumper", "Roof", "Rear Door"],
    "Dump Truck (c09a)": ["Rear Base", "Front Bumper", "Screw", "Cylinder", "Dump Bed", "Sound Module", "Chassis", "Cabin", "Screwdriver", "Wheel", "Roof", "Interior", "Light"],
    "Water Tank Fire Truck (c09b)": ["Wheel", "Sound Module", "Cabin", "Screw", "Light", "Roof", "Water Tank Parts", "Rear Roof", "Rear Base", "Water Tank", "Chassis", "Front Bumper", "Interior"],
    "Transport Truck (c09c)": ["Light", "Rear Base", "Transport Cabin", "Sound Module", "Rear Door", "Chassis", "Rear Roof", "Wheel", "Front Bumper", "Screwdriver", "Cockpit", "Roof", "Screw", "Interior"],
    "Garbage Truck (c10a)": ["Rear Roof", "Garbage Container Parts", "Cover", "Connector", "Chassis", "Wheel", "Screw", "Light", "Screwdriver", "Drive", "Roof", "Front Bumper", "Interior", "Garbage Container", "Sound Module"],
    "Dump Truck (c10b)": ["Chassis", "Roof", "Cylinder", "Cabin", "Interior", "Front Bumper", "Screwdriver", "Dump Bed", "Light", "Sound Module", "Wheel", "Screw", "Rear Base", "Cover"],
    "Water Tank Fire Truck (c10c)": ["Cabin", "Interior", "Rear Roof", "Light", "Rear Base", "Screwdriver", "Sound Module", "Water Tank Parts", "Chassis", "Screw", "Front Bumper", "Wheel", "Roof", "Water Tank"],
    "Sports Car (c11a)": ["Roof", "Sound Module", "Body", "Chassis", "Nut", "Interior", "Wheel", "Screw", "Front Bumper", "Rear Bumper", "Screwdriver"],
    "Sports Car (c11b)": ["Front Bumper", "Roof", "Wheel", "Screw", "Screwdriver", "Sound Module", "Rear Door", "Body", "Chassis", "Interior"],
    "Crane (c12a)": ["Chassis", "Cabin", "Grill", "Screw", "Wheel", "Crane Arm", "Interior", "Arm", "Hook", "Light", "Rear Base", "Roof", "Boom", "Turntable Top", "Turntable Base", "Screwdriver", "Arm Parts", "Crane Arm Parts"],
    "Water Tank Fire Truck (c12b)": ["Rear Base", "Screwdriver", "Ladder", "Chassis", "Water Tank Parts", "Screw", "Wheel", "Light", "Water Tank", "Grill", "Roof", "Interior", "Cabin"],
    "Wheeled Excavator (c12c)": ["Rear Base", "Cabin", "Screwdriver", "Grill", "Turntable Base", "Turntable Top Cover", "Interior", "Arm Parts", "Boom", "Mechanical Arm", "Bucket", "Light", "Roof", "Wheel", "Screw", "Excavator Arm", "Chassis"],
    "Aerial Platform Truck (c12d)": ["Turntable Top", "Ladder", "Screwdriver", "Basket", "Wheel", "Turntable Base", "Grill", "Light", "Rear Base", "Screw", "Ladder Basket", "Chassis", "Roof", "Interior", "Cabin", "Ladder Parts"],
    "Dump Truck (c12e)": ["Dump Bed", "Rear Base", "Screwdriver", "Wheel", "Interior", "Screw", "Grill", "Chassis", "Cabin", "Roof", "Light"],
    "Road Roller (c13a)": ["Screwdriver", "Chassis", "Strap", "Cockpit", "Chassis Parts", "Wheel", "Step", "Screw", "Front Body", "Push Frame", "Roller", "Light"],
    "Jackhammer Truck (c13b)": ["Jackhammer Arm", "Wheel", "Screwdriver", "Chassis", "Jackhammer", "Nut", "Step", "Light", "Boom", "Mechanical Arm", "Front Body", "Cabin", "Strap", "Screw", "Chassis Parts"],
    "Wheeled Excavator (c13c)": ["Excavator Arm", "Wheel", "Bucket", "Nut", "Arm", "Screwdriver", "Boom", "Step", "Strap", "Chassis Parts", "Chassis", "Screw", "Front Body", "Drive", "Light"],
    "Dump Truck (c13d)": ["Chassis", "Wheel", "Screwdriver", "Screw", "Step", "Cabin", "Strap", "Front Body", "Light", "Bucket", "Cabin Parts", "Chassis Parts"],
    "Dump Truck (c13e)": ["Dump Bed", "Cabin", "Screw", "Front Bumper", "Screwdriver", "Chassis", "Wheel", "Light", "Rear Base", "Step", "Chassis Parts"],
    "Water Tank Fire Truck (c13f)": ["Chassis", "Step", "Front Bumper", "Screwdriver", "Chassis Parts", "Light", "Water Tank", "Screw", "Wheel", "Cabin", "Rear Base"],
    "Wheeled Excavator (c14a)": ["Screwdriver", "Roof", "Rear Body", "Step", "Cabin", "Rear Bumper", "Interior", "Wheel", "Chassis", "Arm", "Boom", "Front Body", "Front Base", "Excavator Arm", "Nut", "Screw", "Bucket"],
    "Jackhammer Truck (c14b)": ["Wheel", "Screwdriver", "Chassis", "Jackhammer", "Nut", "Step", "Boom", "Mechanical Arm", "Front Body", "Cabin", "Strap", "Screw", "Roof", "Jackhammer Arm", "Chassis Parts", "Light", "Clamp Arm", "Interior", "Front Base", "Rear Bumper", "Crane Arm"]
}

# ================= 3. Personas and Combination Logic =================

ERROR_TYPES = ["wrong_order", "previous_one_is_mistake", "shouldn't_have_happened", "wrong_position"]

# Assembler personas
PERSONAS = {
    "Expert": {
        "desc": "Professional, fast, efficient. Minimal hesitation.",
        "error_rate": 0.1,
        "style_note": "Movements are precise and confident."
    },
    "Novice": {
        "desc": "Hesitant, clumsy, checks manual often.",
        "error_rate": 0.8,
        "style_note": "Movements are slow, shaky, often pauses to think."
    },
    "Careless": {
        "desc": "Overconfident, fast but sloppy, ignores manual.",
        "error_rate": 0.6,
        "style_note": "Movements are rushed, forceful, often skips checks."
    }
}

# 10 fixed 3-person combinations
ASSEMBLER_COMBINATIONS = [
    ["Expert", "Expert", "Novice"],
    ["Novice", "Novice", "Expert"],
    ["Careless", "Careless", "Expert"],
    ["Expert", "Novice", "Careless"],
    ["Novice", "Expert", "Careless"],
    ["Careless", "Novice", "Expert"],
    ["Novice", "Novice", "Novice"],
    ["Careless", "Novice", "Careless"],
    ["Expert", "Expert", "Expert"],
    ["Expert", "Careless", "Expert"]
]

# ================= 4. Task Generation Matrix =================
def generate_matrix_tasks():
    tasks = []
    task_id_counter = 0
    
    viewpoint_options = [
        {"type": "EGO_BW", "desc": "First-Person (Black & White)"},
        {"type": "EXO_FRONT", "desc": "Third-Person (Front-Facing)"}
    ]
    
    for toy_id, parts in TOY_DATABASE.items():
        for combo_idx, combo_personas in enumerate(ASSEMBLER_COMBINATIONS):
            views = [random.choice(viewpoint_options) for _ in range(3)]
            if all(v['type'] == views[0]['type'] for v in views):
                views[0] = viewpoint_options[1] if views[0]['type'] == "EGO_BW" else viewpoint_options[0]
            
            videos_config = []
            for i in range(3):
                persona_name = combo_personas[i]
                p_data = PERSONAS[persona_name]
                is_error = random.random() < p_data['error_rate']
                if is_error:
                    num_errs = random.choice([1, 1, 2])
                    errors = random.sample(ERROR_TYPES, k=num_errs)
                else:
                    errors = []
                
                videos_config.append({
                    "label": f"Video {i+1}",
                    "view_type": views[i]['desc'],
                    "persona_name": persona_name,
                    "persona_desc": p_data['desc'],
                    "style_note": p_data['style_note'],
                    "errors": errors
                })
            
            tasks.append({
                "id": task_id_counter,
                "toy_id": toy_id,
                "parts": parts,
                "combo_id": combo_idx,
                "videos_config": videos_config
            })
            task_id_counter += 1
            
    return tasks

# ================= 5. Prompt Builder =================
def build_prompt(params):
    toy_id = params['toy_id']
    parts_str = ", ".join(params['parts'])
    
    v_reqs = ""
    for cfg in params['videos_config']:
        err_str = ", ".join(cfg['errors']) if cfg['errors'] else "None (Perfect Assembly)"
        v_reqs += f"""
        **{cfg['label']}**:
        - **Viewpoint**: {cfg['view_type']}
        - **Persona**: {cfg['persona_name']} ({cfg['persona_desc']})
        - **Movement Style**: {cfg['style_note']}
        - **Errors to Simulate**: [{err_str}]
        """

    prompt = f"""
    You are an expert scriptwriter for Video Understanding Benchmarks.
    **Task:** Generate detailed visual scripts for 3 videos of different people assembling the same toy.
    
    **Context:**
    - Toy: {toy_id}
    - Parts Sequence (Standard SOP): [{parts_str}]
    - **Goal:** In ALL videos, the user MUST eventually complete the FULL assembly of all parts.
    
    **SCRIPTING RULES (STRICT):**
    1. **Full Process**: Start from loose parts on the table -> Finish with the completed toy.
    2. **Granularity**: 
       - ONE clip = ONE distinct action (e.g., "Attach Wheel", "Search for Screw", "Unscrew Body"). 
       - DO NOT combine multiple assembly steps into one clip.
    3. **Allowed Actions**: 
       - `ATTACH`: Connecting two parts.
       - `DETACH`: Removing a part (crucial for fixing errors).
       - `ADJUST`: Rotating/Aligning a part without removing it.
       - `INSPECT`: Checking a part or the manual (Static action).
       - `SEARCH`: Looking for a part on the table (Static action).
       - `IDLE`: Brief pause/resting.
    4. **Visual Detail**: Describe the hands, the specific parts, and the physics (e.g., "struggles to insert", "part clicks into place"). NO Audio/Captions.
    
    **ERROR LOGIC (Apply based on Config):**
    - **wrong_order**: User attaches Part B. Realizes Part A is missing inside. **DETACHES** Part B. Attaches Part A. Re-attaches Part B.
    - **previous_one_is_mistake**: User tries to attach Part C. Fails because Part B (previous step) is loose/backward. User **DETACHES** Part B. Fixes it. Re-attaches Part B. Attaches Part C.
    - **shouldn't_have_happened**: User picks up an unnecessary object (or a part not needed yet). Brings it close. Realizes error. Puts it back without attaching.
    - **wrong_position**: User attaches Part A to the WRONG slot. Realizes misalignment. **DETACHES** Part A. Attaches to CORRECT slot.
    
    **Video Configs:**
    {v_reqs}
    
    **QA GENERATION:**
    Design a Single Choice Question (SCQ) based on these scripts.
    - **Rule**: The question must rely on observing the *process* or *errors*, not just the final result (since all finish).
    - **Focus**: Error identification, Error attribution ("Why did they detach X?") , or Persona comparison ("Which user struggled with the wheels?").
    - **Options**: Must be plausible actions or video references.
    - The design of the question should not limit the answer to being derived from just one video. Instead, it would be better to relate it to the type of error.
    - The total duration of each video is randomly allocated within the range of 100 to 300 seconds.
    
    **Output JSON Format:**
    {{
        "id": {params['id']},
        "meta": {{
            "toy_id": "{toy_id}",
            "personas": ["{params['videos_config'][0]['persona_name']}", "{params['videos_config'][1]['persona_name']}", "{params['videos_config'][2]['persona_name']}"]
        }},
        "question": "Question text...",
        "options": ["A. ...", "B. ...", "C. ...", "D. ..."],
        "correct_answer": "A",
        "videos": {{
            "video_1": {{
                "description": "A methodical assembly by an expert...",
                "clips": [
                    {{ "start": 0, "end": 10, "action_type": "INSPECT", "visual": "User spreads all parts on the table and briefly checks the manual." }},
                    {{ "start": 10, "end": 25, "action_type": "ATTACH", "visual": "User firmly snaps the [Chassis] into the [Base]." }},
                    ... 
                ]
            }},
            "video_2": {{ "description": "...", "clips": [...] }},
            "video_3": {{ "description": "...", "clips": [...] }}
        }}
    }}
    """
    return prompt

# ================= 6. API Handler =================
def call_gemini(prompt):
    headers = {
        "api-key": API_KEY,
        "Content-Type": "application/json"
    }
    payload = {
        "contents": [
            {
                "role": "user",
                "parts": [{"text": prompt}]
            }
        ],
        "generationConfig": {
            "temperature": 0.9,
            "maxOutputTokens": 8192
        }
    }
    
    try:
        response = requests.post(ENDPOINT, headers=headers, json=payload, timeout=120)
        if response.status_code != 200:
            return {"error": f"HTTP {response.status_code}", "details": response.text}
        return response.json()
    except Exception as e:
        return {"error": str(e)}

def worker(task_params):
    prompt = build_prompt(task_params)
    task_id = task_params['id']
    
    for attempt in range(MAX_RETRIES):
        res = call_gemini(prompt)
        
        if "error" in res:
            print(f"⚠️ [Task {task_id}] Retry {attempt+1}. Error: {res['error']}")
            time.sleep(2)
            continue
            
        try:
            if "candidates" not in res:
                 print(f"❌ [Task {task_id}] No candidates.")
                 return {"status": "failed"}

            raw = res['candidates'][0]['content']['parts'][0]['text']
            clean = re.sub(r"```json|```", "", raw).strip()
            data = json.loads(clean)
            
            fname = f"task_{task_id}.json"
            
            fpath = os.path.join(OUTPUT_DIR, fname)
            
            with open(fpath, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            return {"status": "success", "id": task_id}
            
        except json.JSONDecodeError:
            pass
        except Exception as e:
            print(f"❌ [Task {task_id}] Parse Error: {e}")
            
    return {"status": "failed", "id": task_id}

# ================= 7. Main =================
if __name__ == "__main__":
    print(f"🚀 Starting Matrix Generation...")
    
    all_tasks = generate_matrix_tasks()
    print(f"📋 Total Tasks Scheduled: {len(all_tasks)}")
    print(f"📂 Output: {OUTPUT_DIR}")
    
    with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
        futures = {executor.submit(worker, t): t['id'] for t in all_tasks}
        
        for future in as_completed(futures):
            res = future.result()
            if res['status'] == 'success':
                if res['id'] % 10 == 0:
                    print(f"✅ Saved Task {res['id']}")
            else:
                print(f"❌ FAILED Task {res['id']}")
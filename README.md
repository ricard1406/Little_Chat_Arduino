# Little_Chat_Arduino
Little_Chat_Arduino is a simple AI Agent chatbot that lets you
chat with your LLM but also offers an Arduino agent.
With the Arduino agent you can ask your LLM to control your
Arduino device and do things in the real world.
The Arduino agent builds your sketch and runs it, and lets your
LLM know if something went wrong — e.g. device not connected or
device errors.

Supports triple provider mode:
  - Local Ollama LLM   (default, no API key needed)
  - Network Ollama LLM (LAN server, e.g. 192.168.1.58:11434)
  - Anthropic Claude   (requires --api-key)


## 📦 Run your application.....

<img width="750" height="400" alt="Run_application" src="https://github.com/user-attachments/assets/ba192457-ae69-4405-9772-ae902b207724" />

## 📦 Open your browser.....

<img width="600" height="600" alt="Chatbot" src="https://github.com/user-attachments/assets/4e6f248e-a632-4fcb-bc39-701ae0addfa5" />

## 📦 Type your prompt .....

<img width="600" height="600" alt="Chatbot_response" src="https://github.com/user-attachments/assets/e409a699-7b09-491d-83dd-0560242bced5" />

## 📦 Your Arduino ....

<img width="2000" height="1500" alt="1000019618" src="https://github.com/user-attachments/assets/c56b9116-4c40-470a-a73f-17fea26961f2" />



## 📦 Installation

📦 Download Ollama local model : (not required when using Claude)
```bash
ollama pull qwen3:4b
```
📦 Installation
   ```bash
   wget https://github.com/ricard1406/Little_Chat_Arduino/archive/refs/heads/main.zip
   unzip main.zip
   mv Little_Chat_Arduino-main Little_MCP
   cd Little_Chat_Arduino
   ```
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   pip install requests python-dotenv pydantic langchain-core langgraph langchain-ollama gradio langchain-anthropic fastapi uvicorn pytz timezonefinder geopy
   ```
📦 Config your api_key (if any)
   ```bash
   cd source
   ```
   ```bash
   [open your fav editor and set your openweather key, claude api_key]
   [vi] .env
   OPENWEATHER_API_KEY=your_api_key_here
   ANTHROPIC_API_KEY=your_api_key_here

   Note ): key required just if you use it. otherwise not required

   ```
📦 Config your arduino_configuration
   ```bash
   [open your fav editor and set your arduino configuration]
   [vi] Arduino_configuration.txt
   Arduino UNO R3 and LCD 1602 module (connect pins are: 12, 11, 5, 4, 3, 2).

   Note ): describe your arduino device . So, your LLM will know how use it .

   ```
📦 Start app
   ```bash
   python Little_Chat_Arduino_server.py

   [leave open this terminal and open a second terminal please]
   [activate your env if not jet] : source ../.venv/bin/activate   ]

   python little_Chat_Arduino.py graph --provider anthropic  --think

   python little_Chat_Arduino.py [text/graph]              (Ollama LLM)
   python little_Chat_Arduino.py [text/graph] --think       (Ollama LLM thinking mode)
   python little_Chat_Arduino.py [text/graph] --provider anthropic               (Claude LLM)
   python little_Chat_Arduino.py [text/graph] --provider anthropic --think       (Claude LLM thinking mode)

   note: add graph parameter for graphical interface
   When use graph interface open your browser and run local URL:
   http://127.0.0.1:7860             
   ```

📦 Remember to install your Arduino CLI
   ```bash
Reference:
   https://docs.arduino.cc/arduino-cli/installation/

my simple guide:
curl -fsSL https://raw.githubusercontent.com/arduino/arduino-cli/master/install.sh | sudo BINDIR=/usr/local/bin sh
arduino-cli version
arduino-cli config init
arduino-cli core update-index
arduino-cli core install arduino:avr
arduino-cli lib install "LiquidCrystal"

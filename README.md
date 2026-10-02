# Weather Advisory Support Bot

A simple chatbot that helps users make safer decisions about outdoor activities using live weather data.

## How It Works

1. Takes a user's question and identifies the location.
2. Fetches current weather data from the Open-Meteo API.
3. Checks the weather conditions against the relevant safety guidelines.
4. Uses an LLM to generate a clear, helpful response.
5. Keeps track of the conversation so users can ask follow-up questions.

## Tech Stack

* **Python** for the application
* **Streamlit** for the chat interface
* **LangGraph** for managing the conversation flow
* **Groq LLM** for understanding questions and generating responses
* **Open-Meteo API** for live weather data

## Key Features

* Live weather updates based on the user's location
* Natural language questions and follow-up conversations
* Weather-based safety advice for outdoor activities
* Error handling when weather data is unavailable
* Conversation memory within a session

## Setup

**1. Install the dependencies**

```bash
pip install -r requirements.txt
```

**2. Add your Groq API key**

Create a `.env` file in the project folder:

```env
GROQ_API_KEY=your_groq_api_key_here
```

**3. Run the application**

```bash
streamlit run app.py
```

Open the local URL shown in your terminal to start chatting with the bot.

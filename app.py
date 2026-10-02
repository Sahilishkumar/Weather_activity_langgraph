
import streamlit as st
import json
import requests
import uuid
import os
from typing import TypedDict, List, Dict, Optional, Annotated
from langchain_core.messages import AnyMessage, HumanMessage, AIMessage, SystemMessage
from langgraph.graph.message import add_messages
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from pydantic import BaseModel, Field

from dotenv import load_dotenv
load_dotenv()


class Intent(BaseModel):
    location: Optional[str] = Field(
        description="The city mentioned by the user. If they refer to a location from earlier, extract it. If no location, return None."
    )

class SOPMatch(BaseModel):
    sop_id: str = Field(description="ID of the single most relevant SOP that applies. If none, return 'NONE'.")
    reasoning: str = Field(description="Reasoning for this SOP selection or why none applied.")


class AgentState(TypedDict):
    messages: Annotated[List[AnyMessage], add_messages]
    location: Optional[str]
    weather_data: Optional[Dict]
    weather_error: Optional[str]
    matched_sop: Optional[Dict]

try:
    with open("sops.json", "r") as f:
        SOPS = json.load(f)
except Exception:
    SOPS = []


def get_coordinates(city: str):
    url = f"https://geocoding-api.open-meteo.com/v1/search?name={city}"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        data = response.json()
        if "results" in data and len(data["results"]) > 0:
            return data["results"][0]["latitude"], data["results"][0]["longitude"]
        return None, None
    except Exception:
        return None, None


def get_weather(lat: float, lon: float):
    url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current=temperature_2m,wind_speed_10m,precipitation,precipitation_probability,uv_index"
    try:
        response = requests.get(url, timeout=5)
        response.raise_for_status()
        return response.json()
    except Exception:
        return None


def parse_intent(state: AgentState):
    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "Review the conversation. Extract the location if the user is asking about weather or outdoor activities. If it's a follow-up and the location was previously mentioned, extract the previously mentioned location."),
        ("placeholder", "{messages}")
    ])
    chain = prompt | llm.with_structured_output(Intent)
    res = chain.invoke({"messages": state["messages"]})
    return {"location": res.location}


def fetch_weather(state: AgentState):
    if not state.get("location"):
        return {
            "weather_error": "I couldn't identify a location. Which city are you asking about?",
            "weather_data": None
        }

    lat, lon = get_coordinates(state["location"])
    if lat is None:
        return {
            "weather_error": f"I couldn't resolve the location '{state['location']}'. Please check the spelling or provide a different city.",
            "weather_data": None
        }

    weather = get_weather(lat, lon)
    if weather is None:
        return {
            "weather_error": "I'm currently unable to fetch weather data from the API. Please try again later.",
            "weather_data": None
        }

    return {"weather_data": weather, "weather_error": None}


def evaluate_sops(state: AgentState):
    if state.get("weather_error"):
        return {"matched_sop": None}
    
    llm = ChatGroq(model="openai/gpt-oss-120b", temperature=0)
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a strict policy evaluator. You must evaluate the SOPs against the user query and weather data. Choose the single most relevant SOP ID that matches the conditions. If none apply, return 'NONE'.\n\nSOPs:\n{sops}\n\nWeather:\n{weather}"),
        ("placeholder", "{messages}")
    ])
    chain = prompt | llm.with_structured_output(SOPMatch)
    res = chain.invoke({
        "messages": state["messages"],
        "sops": json.dumps(SOPS, indent=2),
        "weather": json.dumps(state["weather_data"], indent=2)
    })
    
    matched_sop = next((s for s in SOPS if s["id"] == res.sop_id), None)
    return {"matched_sop": matched_sop}


def generate_response(state: AgentState):
    if state.get("weather_error"):
        return {"messages": [AIMessage(content=state["weather_error"])]}
        
    matched_sop = state.get("matched_sop")
    if not matched_sop:
        return {"messages": [AIMessage(content="I don't have guidance for that. No applicable policy was found.")]}

    llm = ChatGroq(
        model="openai/gpt-oss-120b",
        temperature=0
    )
    prompt = ChatPromptTemplate.from_messages([
        ("system", "You are a weather advisory bot. Respond strictly based on the matched SOP and provided weather data. "
                   "1. CITE the actual numbers (temperature, rain, wind, etc.) from the API data to ground your advice.\n"
                   "2. EXPLICITLY state which SOP ID applies and why.\n"
                   "3. DO NOT invent generic advice if the SOP doesn't cover it. "
                   "4. Maintain a helpful and conversational tone.\n\nApplied SOP:\n{sop}\n\nWeather Data:\n{weather}"),
        ("placeholder", "{messages}")
    ])
    chain = prompt | llm
    res = chain.invoke({
        "messages": state["messages"],
        "sop": json.dumps(matched_sop, indent=2),
        "weather": json.dumps(state["weather_data"], indent=2)
    })
    return {"messages": [res]}


# Build LangGraph
builder = StateGraph(AgentState)
builder.add_node("parse_intent", parse_intent)
builder.add_node("fetch_weather", fetch_weather)
builder.add_node("evaluate_sops", evaluate_sops)
builder.add_node("generate_response", generate_response)

builder.add_edge(START, "parse_intent")
builder.add_edge("parse_intent", "fetch_weather")
builder.add_edge("fetch_weather", "evaluate_sops")
builder.add_edge("evaluate_sops", "generate_response")
builder.add_edge("generate_response", END)

memory = MemorySaver()
graph = builder.compile(checkpointer=memory)


# Streamlit App
st.set_page_config(page_title="Weather Advisory Bot", page_icon="🌤️")
st.title("Weather Advisory Bot")

if "thread_id" not in st.session_state:
    st.session_state.thread_id = str(uuid.uuid4())

config = {"configurable": {"thread_id": st.session_state.thread_id}}

try:
    current_state = graph.get_state(config)
    messages = current_state.values.get("messages", [])
except Exception:
    messages = []

for msg in messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.write(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.write(msg.content)

user_input = st.chat_input(
    "Ask about outdoor activity safety (e.g., 'Is it safe to cycle today in Bhopal?')"
)

if user_input:
    with st.chat_message("user"):
        st.write(user_input)

    with st.chat_message("assistant"):
        with st.spinner("Analyzing..."):
            result_state = graph.invoke(
                {"messages": [HumanMessage(content=user_input)]},
                config=config
            )
            latest_msg = result_state["messages"][-1]
            st.write(latest_msg.content)
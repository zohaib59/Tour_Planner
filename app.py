# app.py
import os
from dotenv import load_dotenv
from datetime import datetime
import streamlit as st

from crewai import LLM, Agent, Task, Crew, Process
from crewai.tools import tool
from langchain_community.tools import DuckDuckGoSearchResults
from langchain.chat_models import ChatOpenAI

# Load environment variables
load_dotenv()

# Web search tool
@tool
def search_web_tool(query: str):
    """
    Searches the web using DuckDuckGo and returns results.
    """
    search_tool = DuckDuckGoSearchResults(num_results=10, verbose=True)
    return search_tool.run(query)

# Streamlit interface
st.set_page_config(page_title="AI Travel Planner", layout="centered", initial_sidebar_state="collapsed")
st.title("🌍 AI Travel Planner")

# UI controls
with st.sidebar:
    theme_toggle = st.radio("Theme Mode", ["Light", "Dark"], index=0)
    if theme_toggle == "Dark":
        st.markdown("""<style>body { background-color: #121212; color: white; }</style>""", unsafe_allow_html=True)

    if st.button("🛑 Kill Session"):
        st.session_state.clear()
        st.success("Session terminated.")
        st.stop()

    if st.button("🧹 Clear History"):
        if "chat_history" in st.session_state:
            del st.session_state["chat_history"]
        st.success("Chat history cleared.")

from_city = st.text_input("From City", value="India")
destination_city = st.text_input("Destination City", value="Rome")
date_from = st.text_input("Departure Date", value="1st March 2025")
date_to = st.text_input("Return Date", value="7th March 2025")
interests = st.text_area("Interests", value="sight seeing and good food")

run_button = st.button("Plan My Trip")

# Ensure the LLM is correctly initialized
def get_openai_llm():
    """
    Initializes the OpenAI LLM for use with CrewAI.
    """
    return LLM(
        model="gpt-4o",
        api_key=os.getenv("OPENAI_API_KEY")
    )

if run_button:
    llm = get_openai_llm()

    # Agents
    guide_expert = Agent(
        role="City Local Guide Expert",
        goal="Provides information on things to do in the city based on the user's interests.",
        backstory="A local expert with a passion for sharing the best experiences and hidden gems of their city.",
        tools=[search_web_tool],
        verbose=True,
        max_iter=5,
        llm=llm,
        allow_delegation=False,
    )

    location_expert = Agent(
        role="Travel Trip Expert",
        goal="Gather helpful information about the city during travel, in ENGLISH only.",
        backstory="A seasoned traveler who has explored various destinations and knows the ins and outs of travel logistics.",
        tools=[search_web_tool],
        verbose=True,
        max_iter=5,
        llm=llm,
        allow_delegation=False,
    )

    planner_expert = Agent(
        role="Travel Planning Expert",
        goal="Compiles all gathered information to provide a comprehensive travel plan.",
        backstory="An organizational wizard who can turn a list of possibilities into a seamless itinerary.",
        tools=[search_web_tool],
        verbose=True,
        max_iter=5,
        llm=llm,
        allow_delegation=False,
    )

    # Tasks
    location_task = Task(
        description=f"""
        Comprehensive data collection on accommodations, transportation, visa, costs, weather, and events.
        Traveling from: {from_city}, Destination city: {destination_city}
        Dates: {date_from} to {date_to}
        Respond in ENGLISH.
        """,
        expected_output=f"""
        Markdown report with places to stay, living expenses, travel tips.
        Respond in ENGLISH.
        """,
        agent=location_expert,
        output_file='city_report.md'
    )

    guide_task = Task(
        description=f"""
        Create a city guide tailored to interest: {interests}. Include attractions, food, events.
        Dates: {date_from} to {date_to}
        Respond in ENGLISH.
        """,
        expected_output=f"""
        Markdown guide with itinerary and attraction details.
        """,
        agent=guide_expert,
        output_file='guide_report.md'
    )

    planner_task = Task(
        description=f"""
        Combine all data into a detailed itinerary with a 4-paragraph city intro and daily travel plan.
        Dates: {date_from} to {date_to}
        Respond in ENGLISH.
        """,
        expected_output=f"""
        Markdown with emojis, city overview, cost, visit spots, and daily travel plan.
        """,
        context=[location_task, guide_task],
        agent=planner_expert,
        output_file='travel_plan.md'
    )

    crew = Crew(
        agents=[location_expert, guide_expert, planner_expert],
        tasks=[location_task, guide_task, planner_task],
        process=Process.sequential,
        full_output=True,
        share_crew=False,
        verbose=True
    )

    result = crew.kickoff()
    st.success("✅ Travel plan created successfully!")
    st.write(result)

import streamlit as st
import requests

# nastevni UI
st.set_page_config(page_title="Micro-EVA - Experimental Assistant", page_icon="🤖")
st.title("📱 Terminal: Mikro-EVA")
st.markdown("Connection initialized. EVA is ready ✅")

# URL pro připojení na "Mozek" 🧠 (FastAPI)
#API_URL = "http://127.0.0.1:8000/chat"
API_URL = "https://mikro-eva-cyberich-cmcjf0ardbgqdshh.swedencentral-01.azurewebsites.net/chat"

# lokální pamět, aby nemizelo okno při refreshi
if "messages" not in st.session_state:
    st.session_state.messages = []

# vykresleni historie na obrazovku
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# starting point
user_input = st.chat_input("Start your chat with EVA ⌛ (if you dare..)")

# zobrazim si dotaz uzivatele v bubline a ulozim do pameti
if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

        # odesilam do main.py backend mozku 🧠
        with st.chat_message("assistant"):
            with st.spinner("EVA is thinking 🔢"):
                try:
                    # vstup od uzivatele - testovano na backendu
                    payload = {"user_id": 1, "message": user_input}

                    # pozadavek na FastAPI pro post v mozku
                    response = requests.post(API_URL, json=payload)

                    if response.status_code == 200:
                        eva_reply = response.json()["response"]
                        st.markdown(eva_reply)
                        st.session_state.messages.append({"role": "assistant", "content": eva_reply})
                    else:
                        st.error(f"Error detected. I can see {response.status_code}")
                except Exception as e:
                    st.error("Critical error detected. Mozek 🧠 nereaguje.")

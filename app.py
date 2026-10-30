
import json

import pandas as pd
import streamlit as st
from sqlalchemy import text

from database import engine
from llm import ask_database, get_history


# 1. PAGE CONFIGURATION

st.set_page_config(
    page_title="LangChain Text-to-SQL",
    page_icon="🗄️",
    layout="wide",
)


# 2. INITIALIZE SESSION STATE

if "messages" not in st.session_state:
    st.session_state.messages = []

if "selected_history_id" not in st.session_state:
    st.session_state.selected_history_id = None


# 3. SIDEBAR AND QUERY HISTORY

with st.sidebar:
    st.title("Database Assistant")

    if st.button("➕ New conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.selected_history_id = None
        st.rerun()

    st.divider()
    st.subheader("📚 Query History")

    try:
        history_df = get_history(limit=100)

        if history_df.empty:
            st.info("No previous queries yet.")

        else:
            for _, row in history_df.iterrows():
                history_id = int(row["log_id"])
                question_text = str(row["question"])

                label = (
                    question_text[:45] + "..."
                    if len(question_text) > 45
                    else question_text
                )

                if st.button(
                    f"💬 {label}",
                    key=f"history_{history_id}",
                    use_container_width=True,
                ):
                    st.session_state.selected_history_id = history_id
                    st.rerun()

                st.caption(
                    f"ID: {history_id} | "
                    f"{row['status']} | "
                    f"{row['created_at']}"
                )

    except Exception as error:
        st.warning("Unable to load query history.")
        print("History loading error:", repr(error))


# 4. APPLICATION HEADER

st.title("LangChain Text-to-SQL")

st.write(
    "Ask questions about your PostgreSQL database and receive "
    "natural-language answers, generated SQL, and query results."
)


# 5. RESTORE SELECTED HISTORY

if st.session_state.selected_history_id is not None:
    try:
        history_query = text("""
            SELECT
                log_id,
                question,
                generated_sql,
                answer,
                result_data,
                status,
                error_message,
                created_at
            FROM query_history
            WHERE log_id = :history_id
        """)

        with engine.connect() as connection:
            record = connection.execute(
                history_query,
                {
                    "history_id": st.session_state.selected_history_id
                },
            ).mappings().first()

        if record:
            st.subheader("📖 Previous Query")

            st.markdown("**Question**")
            st.write(record["question"])

            if record["status"] == "success":
                st.markdown("**Answer**")
                st.write(record["answer"] or "No answer was saved.")

            if record["generated_sql"]:
                st.markdown("**Generated SQL**")
                st.code(record["generated_sql"], language="sql")

            if record["status"] == "success":
                saved_data = record["result_data"]

                if saved_data:
                    try:
                        records = json.loads(saved_data)
                        result_df = pd.DataFrame(records)

                        if not result_df.empty:
                            st.markdown("**Query Results**")
                            st.dataframe(
                                result_df,
                                use_container_width=True,
                                hide_index=True,
                            )
                        else:
                            st.info("The saved query returned no rows.")

                    except (ValueError, TypeError):
                        st.info(
                            "The saved result data could not be displayed."
                        )
                else:
                    st.info("No saved result data is available.")

            else:
                st.error("This historical query failed.")

                if record["error_message"]:
                    st.caption(record["error_message"])

        else:
            st.warning("This history entry could not be found.")

    except Exception as error:
        st.error("Unable to restore this history entry.")
        print("History restoration error:", repr(error))

    st.divider()


# 6. DISPLAY CURRENT CONVERSATION

st.subheader("💬 Current Conversation")

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

        if message.get("sql"):
            st.markdown("**Generated SQL**")
            st.code(message["sql"], language="sql")

        data = message.get("data")

        if isinstance(data, pd.DataFrame) and not data.empty:
            st.markdown("**Query Results**")
            st.dataframe(
                data,
                use_container_width=True,
                hide_index=True,
            )


# 7. USER INPUT

question = st.chat_input("Ask your database a question...")


# 8. PROCESS NEW QUESTIONS

if question:
    st.session_state.selected_history_id = None

    prior_messages = st.session_state.messages[-10:]

    chat_history = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in prior_messages
    )

    st.session_state.messages.append({
        "role": "user",
        "content": question,
    })

    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner(
            "Understanding your question and checking the database..."
        ):
            response = ask_database(
                question,
                chat_history=chat_history,
            )

        if response.get("error"):
            answer = response["error"]
            st.error(answer)

        else:
            answer = response.get("answer", "")
            st.markdown(answer)

        generated_sql = response.get("sql", "")
        result_df = response.get("data")

        if generated_sql:
            st.markdown("**Generated SQL**")
            st.code(generated_sql, language="sql")

        if (
            not response.get("error")
            and isinstance(result_df, pd.DataFrame)
        ):
            if not result_df.empty:
                st.markdown("**Query Results**")
                st.dataframe(
                    result_df,
                    use_container_width=True,
                    hide_index=True,
                )
            elif generated_sql:
                st.info("No matching records found.")

        st.session_state.messages.append({
            "role": "assistant",
            "content": answer,
            "sql": generated_sql,
            "data": (
                result_df.copy()
                if isinstance(result_df, pd.DataFrame)
                and not response.get("error")
                else None
            ),
        })

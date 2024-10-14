import re
import warnings
import boto3
import base64
import json
from PIL import Image
import io

import streamlit as st

from template import clarifier_prompt, SQL_prompt, eval_prompt, analysis_prompt

from pydantic import BaseModel, validator
from langchain_aws import BedrockLLM, ChatBedrock
from langchain_community.embeddings import BedrockEmbeddings
# from langchain.prompts.prompt import PromptTemplate
# from langchain.schema import format_document
# from langchain_core.messages import get_buffer_string
from langchain_core.output_parsers import StrOutputParser
# from langchain_core.runnables import RunnableParallel, RunnablePassthrough
from operator import itemgetter

import pandas as pd
from pygwalker.api.streamlit import StreamlitRenderer

from ui_utils import StreamlitUICallbackHandler, message_func
import psycopg2
from psycopg2 import OperationalError, ProgrammingError, DataError, IntegrityError

def get_database_connection(db_params):
    # Check if 'conn' exists in locals and is a valid connection
    if 'conn1' in locals() and locals()['conn1'] and not locals()['conn1'].closed:
        print("Reusing existing connection")
        return locals()['conn1']
    
    # If not, create a new connection
    try:
        print("Creating new connection")
        conn1 = psycopg2.connect(**db_params)
        return conn1
    except OperationalError as e:
        print(f"Error connecting to PostgreSQL database: {e}")
        return None

bedrock_runtime = boto3.client(
    service_name='bedrock-runtime',
    region_name='us-east-1'
)

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf8')

def get_embedding(myimage):
    body = json.dumps({
        "inputImage": myimage,
        "embeddingConfig": {
            "outputEmbeddingLength": 1024  # Can be 256, 384, or 1024
        }
    })
    response = bedrock_runtime.invoke_model(
        body=body,
        modelId="amazon.titan-embed-image-v1",
        accept="application/json",
        contentType="application/json"
    )
    response_body = json.loads(response['body'].read())
    embedding = response_body['embedding']
    return embedding

def resize_image(input_path, output_path, max_pixels=100000):
    with Image.open(input_path) as img:
        # Calculate the current number of pixels
        current_pixels = img.width * img.height
        
        # Calculate the scaling factor
        scale = (max_pixels / current_pixels) ** 0.5
        
        # Calculate new dimensions
        new_width = int(img.width * scale)
        new_height = int(img.height * scale)
        
        # Resize the image
        resized_img = img.resize((new_width, new_height), Image.LANCZOS)
        
        # Save the resized image
        resized_img.save(output_path)
        


### Configurations ###
st.set_page_config(layout="wide")

db_config = {
    'dbname': 'aderas',
    'user': 'aderas',
    'password': 'aderas123',
    'host': 'localhost',
    'port': '5432'
}

try:
    # Attempt to connect to the PostgreSQL database
    # conn = psycopg2.connect(**db_config)
    conn = get_database_connection(db_config)
    
    # Check if the connection is valid
    if conn:
        print("Connection successful!")

except OperationalError as e:
    print(f"Connection failed: {e}")

cursor = conn.cursor()
query = "SELECT count(*) FROM medals"
cursor.execute(query)
records = cursor.fetchone()

embeddings = BedrockEmbeddings(region_name="us-east-1")

# modelid = "anthropic.claude-v2:1" 
# modelid = "anthropic.claude-3-sonnet-20240229-v1:0" 
modelid = "anthropic.claude-instant-v1"

llm = BedrockLLM (
# llm = ChatBedrock( 
    credentials_profile_name="default", model_id=modelid )

clarifier_chain = (
    {
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history"),
    }
    | clarifier_prompt
    | llm
    | StrOutputParser()
)

eval_chain = (
    {
        "standalone_question": itemgetter("standalone_question"),
    }
    | eval_prompt
    | llm
    | StrOutputParser()
)

sql_chain = (
    {
        "standalone_question": itemgetter("standalone_question"),
    }
    | SQL_prompt
    | llm
    | StrOutputParser()
)

analysis_chain = (
    {
        "mydata": itemgetter("mydata"),
    }
    | analysis_prompt
    | llm
    | StrOutputParser()
)

### 

warnings.filterwarnings("ignore")
chat_history = []

gradient_text_html = """
<style>
.gradient-text {
    font-weight: bold;
    background: -webkit-linear-gradient(left, red, orange);
    background: linear-gradient(to right, red, orange);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    display: inline;
    font-size: 3em;
}
.st-emotion-cache-16txtl3 {
  padding: 3rem 1.5rem;
}
.st-emotion-cache-9tg1hl {
  padding: 2rem 1rem 1rem;
}
.st-emotion-cache-1smf172 {
  margin-top: -1rem;
}
</style>
<div class="gradient-text">DataChat</div>
"""

st.markdown(gradient_text_html, unsafe_allow_html=True)

st.session_state["model"] = modelid

INITIAL_MESSAGE = [
    {"role": "user", "content": "Hi!"},
    {
        "role": "assistant",
        "content": "Hey there, I'm your SQL-speaking sidekick, ready to help you talk with your data and retrieve answers fast. By the way, we have "+str(records[0])+" records in the modal database.",
    },
]

with open("ui/sidebar.md", "r") as sidebar_file:
    sidebar_content = sidebar_file.read()

with open("ui/styles.md", "r") as styles_file:
    styles_content = styles_file.read()

st.sidebar.markdown(sidebar_content)
st.sidebar.divider()

sbcol1, sbcol2 = st.sidebar.columns(2)

# Place objects in the first column
with sbcol2:
    use_widget = st.checkbox("Use Widget")

# Place objects in the second column
with sbcol1:
    # Add a reset button
    if st.button("Reset Chat"):
        for key in st.session_state.keys():
            del st.session_state[key]
        st.session_state["messages"] = INITIAL_MESSAGE
        st.session_state["history"] = []

st.write(styles_content, unsafe_allow_html=True)


# Initialize the chat messages history
if "messages" not in st.session_state.keys():
    st.session_state["messages"] = INITIAL_MESSAGE

if "history" not in st.session_state:
    st.session_state["history"] = []

# Prompt for user input and save
if prompt := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": prompt})

for message in st.session_state.messages:
    message_func(
        message["content"],
        True if message["role"] == "user" else False,
        True if message["role"] == "data" else False,
        modelid,
    )

callback_handler = StreamlitUICallbackHandler(modelid)

def append_chat_history(question, answer):
    st.session_state["history"].append((question, answer))

def get_sql(text):
    sql_match = re.search(r"```sql\n(.*)\n```", text, re.DOTALL)
    return sql_match.group(1) if sql_match else None

def append_message(content, role="assistant"):
    """Appends a message to the session state messages."""
    if content.strip():
        st.session_state.messages.append({"role": role, "content": content})

def handle_sql_exception(query, conn, e, retries=2):
    append_message("Uh oh, I made an error, let me try to fix it..")
    error_message = (
        "You gave me a wrong SQL. FIX The SQL query by searching the schema definition:  \n```sql\n"
        + query
        + "\n```\n Error message: \n "
        + str(e)
    )
    new_query = chain({"question": error_message, "chat_history": ""})["answer"]
    append_message(new_query)
    if get_sql(new_query) and retries > 0:
        return execute_sql(get_sql(new_query), conn, retries - 1)
    else:
        append_message("I'm sorry, I couldn't fix the error. Please try again.")
        return None

def execute_sql(query, conn, retries=2):
    if re.match(r"^\s*(drop|alter|truncate|delete|insert|update)\s", query, re.I):
        append_message("Sorry, I can't execute queries that can modify the database.")
        return None
    try:
        mydata = pd.read_sql_query(query, conn)
        return mydata
    except ProgrammingError as e:
        return handle_sql_exception(query, conn, e, retries)

def execute_knn(question, conn):
    myembeddings = embeddings.embed_query(question)
    try:
        knn_query = f"""SELECT id, doc_nbr, part_nbr, national_stock_number, niin, cage, title, file_name, embedding <-> '{myembeddings}' as distance
            FROM medals
            ORDER BY embedding <-> '{myembeddings}'
            LIMIT 5;"""
        mydata = pd.read_sql_query(knn_query, conn)
        return mydata
    except ProgrammingError as e:
        return None

def image_to_base64(image_bytes):
    temp_image = base64.b64encode(image_bytes).decode("utf-8")
    ret_image = f"data:image/jpeg;base64,{temp_image}"
    return ret_image

st.sidebar.markdown("## Image Search")
# Upload file for image similarity search
uploaded_file = st.sidebar.file_uploader("", type=["jpg", "jpeg", "png"], key="upload_file")
# if uploaded_file is not None:
if uploaded_file is not None:
    message_func("""The following is the output of your similarity search based upon an image. 
    First, I must resize your image to a standard size. Each record contains three possible images. 
    I will return the top 5 similar records with the best similarity score.""", is_user=False, is_df=False)

    myext = uploaded_file.name.split(".")[-1].upper()
    if myext == 'JPG':
        myext = 'JPEG' 
    myImg = Image.open(uploaded_file)
    current_pixels = myImg.width * myImg.height
    scale = (100000 / current_pixels) ** 0.5
    # Calculate new dimensions
    new_width = int(myImg.width * scale)
    new_height = int(myImg.height * scale)
    # Resize the image
    resized_img = myImg.resize((new_width, new_height), Image.LANCZOS)
    st.image(resized_img, caption=uploaded_file.name, use_column_width=False)
    buffer = io.BytesIO()
    # Save the image to the buffer in a specified format (e.g., JPEG or PNG)
    resized_img.save(buffer, format=myext)
    # Get the bytes-like object from the buffer
    image_bytes = buffer.getvalue()
    b64_image = base64.b64encode(image_bytes).decode('utf8')
    myembeddings = get_embedding(b64_image)
    mysql = f"""
SELECT id, doc_nbr, part_nbr, title, photo_data_1, photo_data_2, photo_data_3, MIN(distance) AS distance, file_name, national_stock_number, niin, cage
FROM (
SELECT id, doc_nbr, part_nbr, national_stock_number, niin, cage, title, file_name, photo_data_1, photo_data_2, photo_data_3, 
    photo_embedding_1 <-> '{myembeddings}' as distance
FROM medals_data
union
SELECT id, doc_nbr, part_nbr, national_stock_number, niin, cage, title, file_name, photo_data_1, photo_data_2, photo_data_3, 
    photo_embedding_2 <-> '{myembeddings}' as distance
FROM medals_data
union
SELECT id, doc_nbr, part_nbr, national_stock_number, niin, cage, title, file_name, photo_data_1, photo_data_2, photo_data_3, 
    photo_embedding_3 <-> '{myembeddings}' as distance
FROM medals_data
)
group by id, doc_nbr, part_nbr, title, photo_data_1, photo_data_2, photo_data_3, file_name, national_stock_number, niin, cage
order by distance
limit 5;
"""
    mydata = execute_sql(mysql, conn, retries=2)
    mydata["photo_data_1"] = mydata["photo_data_1"].apply(image_to_base64)
    mydata["photo_data_2"] = mydata["photo_data_2"].apply(image_to_base64)
    mydata["photo_data_3"] = mydata["photo_data_3"].apply(image_to_base64)

    mycolumns = {"photo_data_1": st.column_config.ImageColumn("photo_data_1"),
                 "photo_data_2": st.column_config.ImageColumn("photo_data_2"),
                 "photo_data_3": st.column_config.ImageColumn("photo_data_3")}
    st.data_editor(mydata, disabled=True, num_rows="Fixed", hide_index=True, column_config=mycolumns )
    upkey="upload_file"
    if upkey in st.session_state.keys():
        del st.session_state[upkey]
    uploaded_file = None


if (
    "messages" in st.session_state
    and st.session_state["messages"][-1]["role"] != "assistant"
):
    user_input_content = st.session_state["messages"][-1]["content"]

    if isinstance(user_input_content, str):
        callback_handler.start_loading_message()

        standalone_question = clarifier_chain.invoke(
            {
                "question": user_input_content, 
                "chat_history": [h for h in st.session_state["history"]]
            }
        )
        # st.write(standalone_question)
        eval_statement = eval_chain.invoke(
            {
                "standalone_question": standalone_question
            }
        )

        if eval_statement.strip() == "SQL":
            SQL_statement = sql_chain.invoke(
                {
                    "standalone_question": standalone_question
                }
            )

            append_message(SQL_statement)
            message_func(SQL_statement, is_user=False, is_df=False)

            mySQL = get_sql(SQL_statement)
            # st.write(mySQL)
            mydata = execute_sql(mySQL, conn, retries=2)
            # st.data_editor(mydata)
            analysis_data = mydata
            if "file_data" in analysis_data.columns:
                analysis_data = analysis_data.drop(columns=["file_data"])
            if "embedding" in analysis_data.columns:
                analysis_data = analysis_data.drop(columns=["embedding"])
            if "photo_data_1" in analysis_data.columns:
                analysis_data = analysis_data.drop(columns=["photo_data_1"])
            if "photo_embedding_1" in analysis_data.columns:
                analysis_data = analysis_data.drop(columns=["photo_embedding_1"])
            if "photo_data_2" in analysis_data.columns:
                analysis_data = analysis_data.drop(columns=["photo_data_2"])
            if "photo_embedding_2" in analysis_data.columns:
                analysis_data = analysis_data.drop(columns=["photo_embedding_2"])
            if "photo_data_3" in analysis_data.columns:
                analysis_data = analysis_data.drop(columns=["photo_data_3"])
            if "photo_embedding_3" in analysis_data.columns:
                analysis_data = analysis_data.drop(columns=["photo_embedding_3"])
            csv_str = analysis_data.to_csv()
            # st.write(csv_str)
            analysis_statement = analysis_chain.invoke(
                {
                    "mydata": csv_str
                }
            )
            # st.write(analysis_statement)
            append_message(analysis_statement)
            message_func(analysis_statement, is_user=False, is_df=False)

            if use_widget:
                pyg_app = StreamlitRenderer(mydata)
                pyg_app.explorer()
            else:
                if "photo_data_1" in mydata.columns:
                    mydata["photo_data_1"] = mydata["photo_data_1"].apply(image_to_base64)
                if "photo_data_2" in mydata.columns:
                    mydata["photo_data_2"] = mydata["photo_data_2"].apply(image_to_base64)
                if "photo_data_3" in mydata.columns:
                    mydata["photo_data_3"] = mydata["photo_data_3"].apply(image_to_base64)

                mycolumns = {"photo_data_1": st.column_config.ImageColumn("photo_data_1"),
                            "photo_data_2": st.column_config.ImageColumn("photo_data_2"),
                            "photo_data_3": st.column_config.ImageColumn("photo_data_3")}
                st.data_editor(mydata, disabled=True, num_rows="Fixed", hide_index=True, column_config=mycolumns)
        elif eval_statement.strip() == "KNN":
            append_message("A similarity search of the blueprint database was conducted")
            message_func("The following is the output of your similarity search. The top 5 similar records are provided.", is_user=False, is_df=False)
            mydata = execute_knn(user_input_content, conn)
            st.data_editor(mydata, disabled=True, num_rows="Fixed", hide_index=True)
        else:
            pass

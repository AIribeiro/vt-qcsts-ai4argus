import streamlit as st
import openai
import os
import requests
import azure
import random
import re
from azure.data.tables import TableServiceClient
import datetime
#now = datetime.datetime.utcnow()
from datetime import datetime
now=""
now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

import ast
from streamlit_feedback import streamlit_feedback

import random
import uuid
import json

from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import numpy as np
import matplotlib.pyplot as plt

#from openai import AzureOpenAI
from openai import AzureOpenAI


openai.api_type = "azure"
# Azure OpenAI on your own data is only supported by the 2023-08-01-preview API version
openai.api_version = "2023-08-01-preview"


client = AzureOpenAI(api_key="35fcd9150f044fdcbca33b5c3318a1f2",
azure_endpoint="https://vt-generative-ai-dev.openai.azure.com/openai/deployments/vt-gpt-35-turbo/extensions/chat/completions?api-version=2023-07-01-preview",
api_version="2023-08-01-preview")
openai.api_base = "https://vt-generative-ai-dev.openai.azure.com/"




# Deployment Configuration
deployment_id_4 = "vt-text-davinci-003"  # Set the deployment ID
deployment_id_turbo ="vt-gpt-35-turbo"
deployment_id = deployment_id_4


# Azure Cognitive Search setup

search_key = "Zqna1Alw8z4E8hwggZGfTd4prqtVW0tmhwrQlfF81EAzSeCrS0iI" 
search_region= "westeurope"
search_endpoint = "https://vt-cognitivesearch-test.search.windows.net" 
search_index_name = "vt-argus-latam-br-test-index"; # Add your Azure Cognitive Search index name here

# Storage Account
from datetime import datetime, timedelta
from azure.data.tables import TableServiceClient, generate_account_sas, ResourceTypes, AccountSasPermissions
from azure.core.credentials import AzureNamedKeyCredential, AzureSasCredential

connection_string = "SharedAccessSignature=sv=2021-10-04&ss=btqf&srt=sco&st=2023-11-01T20%3A24%3A39Z&se=2026-11-02T20%3A24%3A00Z&sp=rwdxftlacu&sig=64BwcCyMJb9JZK0dC6hNkCUzVShfq0CJkJ8MC%2FRFTIM%3D;BlobEndpoint=https://vtgenerativeaistorage.blob.core.windows.net/;FileEndpoint=https://vtgenerativeaistorage.file.core.windows.net/;QueueEndpoint=https://vtgenerativeaistorage.queue.core.windows.net/;TableEndpoint=https://vtgenerativeaistorage.table.core.windows.net/;"
azure_table_service = TableServiceClient.from_connection_string(conn_str=connection_string)

# Azure Translation and Document Translation Configuration
translator_endpoint = "https://api.cognitive.microsofttranslator.com"
translator_key = "bda7423927544a9f9a318993beae26c7"
translator_location = "westeurope"
key=translator_key
location=translator_location
endpoint=translator_endpoint

doc_translation_endpoint = "https://vt-translator-we-test.cognitiveservices.azure.com/"
doc_translation_key = "bda7423927544a9f9a318993beae26c7"

brainstorming_answer = None
chatbot_answer = None
#original_selected_language_code = "pt"
#original_user_language_selection ="pt"

def setup_byod(deployment_id: str) -> None:
    """Sets up the OpenAI Python SDK to use your own data for the chat endpoint.

    :param deployment_id: The deployment ID for the model to use with your own data.

    To remove this configuration, simply set openai.requestssession to None.
    """

    class BringYourOwnDataAdapter(requests.adapters.HTTPAdapter):

        def send(self, request, **kwargs):
            request.url = f"{openai.api_base}/openai/deployments/{deployment_id}/extensions/chat/completions?api-version={openai.api_version}"
            return super().send(request, **kwargs)

    session = requests.Session()

    # Mount a custom adapter which will use the extensions endpoint for any call using the given `deployment_id`
    session.mount(
        prefix=f"{openai.api_base}/openai/deployments/{deployment_id}",
        adapter=BringYourOwnDataAdapter()
    )

    openai.requestssession = session

setup_byod(deployment_id)

# Streamlit Page Configuration
st.set_page_config(
    page_title="Assistente de IA QCSTS para Argus América Latina",
    page_icon="https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Volvo-Iron-Mark-Black.svg/2048px-Volvo-Iron-Mark-Black.svg.png",
    layout="wide"
)
def hide_streamlit_style():
    st.markdown("""
        <style>
        .reportview-container .main footer {visibility: hidden;}    
        </style>
        """, unsafe_allow_html=True)
    
hide_streamlit_style()
# Custom Styling
st.markdown("""
<style>
    .reportview-container {
        flex-direction: column;
        background-color: #f0f0f0;
    }
    .big-font {
        font-size: 50px !important;
    }
</style>
""", unsafe_allow_html=True)


user_feedback = ""
#chat_session_id=""
chat_session_id_temp=""
chat_answer_temp=""
chat_question_temp=""
user_feedback_temp =""
chat_session_id_temp=""
chat_interactions=0
user_feedback=""


def generate_new_session_id():
    return str(uuid.uuid4())

# Initialize conversation history in session state if it does not exist
if 'conversation_history' not in st.session_state:
    st.session_state.conversation_history = []

if 'chat_question' not in st.session_state:
    st.session_state['chat_question'] = None

if 'chat_feedback' not in st.session_state:
    st.session_state['chat_feedback'] = None    

if 'chat_feedback' not in st.session_state:
    st.session_state['user_feedback'] = None 

    
# Initialize session if not already initialized
if 'chat_session_id' not in st.session_state:
    st.session_state['chat_session_id'] = generate_new_session_id()
    chat_session_id=st.session_state['chat_session_id']
    

# Generate a new session ID - second instance.
#st.session_state['chat_session_id'] = generate_new_session_id()    
chat_session_id=st.session_state['chat_session_id']    
    

# Initialize the counter for tracking chatbot responses
if 'response_counter' not in st.session_state:
    st.session_state['response_counter'] = 0

# This variable will hold the chatbot's answer once the "Get Ideas" button is pressed.
chatbot_answer = None

# Initialize session state variables if they don't exist
if 'chatbot_answer' not in st.session_state:
    st.session_state['chat_answer'] = ""
    
    
#Sidebar
# Sidebar content
# If the image is stored locally:
image_url = "https://1000logos.net/wp-content/uploads/2020/03/Volvo-Logo-2020.png"

# add sidebar buttons
with st.sidebar:
  st.image(image_url, width=250)
    
st.sidebar.header("Assistente de IA QCSTS para Argus América Latina")
st.sidebar.subheader("Using Generative AI to streamline product diagnostics")
    
    
    
    
# Initialize an empty list to store the conversation history
conversation_history = []

def capture_feedback():
    feedback = streamlit_feedback(
        feedback_type="thumbs",
        optional_text_label="[Optional] Please provide an explanation",
    )
    user_feedback = str(feedback)
    st.session_state['chat_feedback'] = feedback
    chat_answer = st.session_state['chat_answer']
    return user_feedback

def remove_suggested_question(text):
    # Split the text at the phrase "Suggested Question:", keeping only the part before it
    processed_text = text.split("Suggested conversation:")[0]
    return processed_text.strip()  # Remove any leading/trailing whitespace

chat_question=""
chat_answer=""

def translate_text(text, source_lang, target_langs):
    """
    Translates text from a source language to one or more target languages using Azure Cognitive Services Translator.

    :param text: The text to translate.
    :param source_lang: The source language code (e.g., 'en' for English).
    :param target_langs: A list of target language codes (e.g., ['fr', 'de'] for French and German).
    :param key: Your Azure Translator subscription key.
    :param location: Your Azure Translator resource location.
    :return: The translated text.
    """
    endpoint = "https://api.cognitive.microsofttranslator.com"
    path = '/translate'
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0',
        'from': source_lang,
        'to': target_langs
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text': text
    }]

    response = requests.post(constructed_url, params=params, headers=headers, json=body)
    return response.json()



languages_with_codes = {
    "English": "en", 
    "Chinese": "zh", 
    "Hindi": "hi", 
    "Spanish": "es", 
    "French": "fr", 
    "Arabic": "ar",
    "Bengali": "bn", 
    "Russian": "ru", 
    "Indonesian": "id", 
    "Urdu": "ur", 
    "German": "de",
    "Japanese": "ja", 
    "Swahili": "sw", 
    "Marathi": "mr", 
    "Telugu": "te", 
    "Turkish": "tr", 
    "Korean": "ko",
    "Vietnamese": "vi", 
    "Tamil": "ta", 
    "Italian": "it", 
    "Dutch": "nl", 
    "Polish": "pl", 
    "Portuguese": "pt", 
    "Persian": "fa",
    "Greek": "el", 
    "Hebrew": "he", 
    "Malay": "ms", 
    "Ukrainian": "uk", 
    "Romanian": "ro", 
    "Thai": "th",
    "Czech": "cs", 
    "Serbian": "sr", 
    "Danish": "da", 
    "Finnish": "fi", 
    "Slovak": "sk", 
    "Norwegian": "no",
    "Hungarian": "hu", 
    "Croatian": "hr", 
    "Bulgarian": "bg", 
    "Lithuanian": "lt", 
    "Slovenian": "sl",
    "Latvian": "lv", 
    "Estonian": "et", 
    "Georgian": "ka", 
    "Armenian": "hy", 
    "Mongolian": "mn", 
    "Kazakh": "kk",
    "Azerbaijani": "az", 
    "Uzbek": "uz", 
    "Khmer": "km", 
    "Sinhala": "si", 
    "Nepali": "ne",
    "Pashto": "ps", 
    "Punjabi": "pa", 
    "Amharic": "am", 
    "Yoruba": "yo", 
    "Igbo": "ig", 
    "Hausa": "ha",
    "Zulu": "zu", 
    "Somali": "so", 
    "Malagasy": "mg", 
    "Luxembourgish": "lb", 
    "Icelandic": "is", 
    "Māori": "mi",
    "Swedish":"sv",
}

# Sort the languages alphabetically by name
languages_sorted = sorted(languages_with_codes.items(), key=lambda x: x[0])

# Prepare a list for the dropdown that combines language name and code
languages_dropdown = [f"{name} ({code})" for name, code in languages_sorted]

# Sidebar dropdown for target language selection, defaulting to English
st.write("**Etapa 1** - Selecione seu idioma.")
user_language_selection = st.selectbox("Select your language:", languages_dropdown, key="translate_to", index=[name for name, code in languages_sorted].index("Portuguese"))

# Extract the ISO language code from the target language selection
selected_language_code = user_language_selection.split('(')[-1].strip(')')
language = selected_language_code

#original_selected_language_code = "pt"
#original_user_language_selection = "pt"
language_style = "Original"
source_language = "en"
target_languages = [selected_language_code]



def display_feedback():
    user_feedback=""
    
    with st.form(key='feedback_form'):
        st.subheader("Was this answer helpful?")
        rating = st.radio("Rate from 1 (Not helpful) to 5 (Very helpful):", range(1, 6), horizontal=True)
        open_feedback = st.text_area("Additional feedback (optional):", placeholder="Your comments here...")
        submit_button = st.form_submit_button('Submit Feedback')

        if submit_button:
            user_feedback = {'rating': rating, 'open_feedback': open_feedback}
            st.session_state['user_feedback'] = user_feedback  # Directly updating the session state
            
            st.session_state['chat_feedback'] = user_feedback
            st.write("Here is the feedback:" + str(user_feedback))
            st.success("Thank you for your feedback!")
            
        
        
    return user_feedback  # Returning feedback for immediate use (optional)



def log_to_azure_table(chat_question, chat_answer, user_feedback, chat_session_id, chat_interactions):
    """
    Log each interaction to an Azure Table without checking for duplicate SessionIDs.

    :param chat_question: The question asked.
    :param chat_answer: The answer provided.
    :param user_feedback: The feedback received, serialized if it's a dict.
    :param chat_session_id: The session ID for the interaction.
    :param chat_interactions: The number of interactions.
    :param azure_table_service: The Azure Table Service client instance.
    """
    # Ensure feedback is a string. Serialize if it's a dict.
    if isinstance(user_feedback, dict):
        user_feedback = json.dumps(user_feedback)
    
    # Get the current date and time as a unique RowKey
    now = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")

    # Create a dictionary to hold the log data
    new_log_data = {
        'PartitionKey': 'LogData',  # Consider changing 'LogData' to something more specific if needed
        'RowKey': str(now),
        'Question': chat_question,
        'Answer': chat_answer,
        'Feedback': user_feedback,
        'SessionID': chat_session_id,
        'Interactions': chat_interactions
    }

    # Get a reference to the table
    table_client = azure_table_service.get_table_client(table_name='vtargusbrassistantapptable')
    
    

    # Insert the new log data
    table_client.create_entity(entity=new_log_data)
    
    # Iterate through entities to find and delete ones with empty 'Question' or 'Answer'
    entities = table_client.list_entities()
    for entity in entities:
        if entity.get('Question') == "" or entity.get('Answer') == "":
            table_client.delete_entity(partition_key=entity['PartitionKey'], row_key=entity['RowKey'])

   




def feedback_cleaner(user_feedback):
    # Check if user_feedback is None
    if user_feedback is None:
        return "user_feedback is None"

    # If user_feedback is a string, convert it to a dictionary
    if isinstance(user_feedback, str):
        try:
            user_feedback = ast.literal_eval(user_feedback)
            if not isinstance(user_feedback, dict):
                return "Invalid user_feedback format"
        except ValueError:
            return "Invalid user_feedback format"
    elif not isinstance(user_feedback, dict):
        return "user_feedback is neither a dictionary nor a valid string representation of a dictionary"

    # Extract the values from the user_feedback dictionary
    feedback_type = user_feedback.get('type', '')
    score = user_feedback.get('score', '')
    text = user_feedback.get('text', '')
    
    # Mapping for the score based on the 'type'
    score_mapping = {
        'thumbs': {
            '👍': 'thumb up',
            '👎': 'thumb down'
        }
    }
    
    # Get the cleaned score based on the feedback type and score
    cleaned_score = score_mapping.get(feedback_type, {}).get(score, score)
    
    # Format the cleaned data into the desired string format
    cleaned_feedback = f"'score: {cleaned_score}', 'comment': '{text}'"
    
    return cleaned_feedback

def count_chatbot_answers():
    table_client = azure_table_service.get_table_client('vtargusbrassistantapptable')

    # Query to fetch all log entries
    entities = table_client.list_entities()

    # Count the number of entries
    answer_count = len(list(entities))

    return answer_count


def harmonize_text(text):
    # Capitalize the first letter of each sentence
    text = '. '.join(sentence.capitalize() for sentence in text.split('. '))

    # Fix punctuation spacing (e.g., "hello ,world" becomes "hello, world")
    text = re.sub(r'\s([?.!",](?:\s|$))', r'\1', text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text

def retrieve_questions():
    table_client = azure_table_service.get_table_client('vtargusbrassistantapptable')
    questions = [entity['Question'] for entity in table_client.list_entities() if 'Question' in entity]

    # Harmonize each question
    harmonized_questions = [harmonize_text(question) for question in questions]

    return harmonized_questions



def calculate_average_questions_per_session(table_service_client):
    """
    Calculate the average number of questions (interactions) per session from an Azure Table.

    :param table_service_client: Instance of TableServiceClient connected to your Azure Table storage.
    :param table_name: The name of the table from which to fetch the data.
    :return: The average number of questions per session.
    """
    table_client = azure_table_service.get_table_client('vtargusbrassistantapptable')
    
    # Initialize counters
    total_questions = 0  # This now represents the sum of all questions across sessions
    total_sessions = 0   # Total number of sessions
    total_interactions = 0
    count_entities = 0
    
    # Fetch all entities and sum up the 'Interactions' values
    entities = table_client.list_entities()
    for entity in entities:
        total_interactions += int(entity.get('Interactions', 0))
        count_entities += 1  # Increment for each entity processed
    
    # Debugging: Print the total sum of all 'Interactions'
    #st.write("Total sum of all interactions:", total_interactions)
    
    # Calculate the average interactions if there are entities
    if count_entities > 0:
        average_interactions = total_interactions / count_entities
        return format(average_interactions, '.1f')  # Format to one decimal place
    else:
        return "No data found"  # Handle case with no entities



average = calculate_average_questions_per_session(azure_table_service)






def cluster_questions(questions, num_clusters=10):
    model = SentenceTransformer('all-MiniLM-L6-v2')

    # Generating embeddings for each question
    embeddings = model.encode(questions, convert_to_tensor=True)

    # Clustering
    clustering_model = KMeans(n_clusters=num_clusters)
    clustering_model.fit(embeddings)
    cluster_assignment = clustering_model.labels_

    clustered_questions = {}
    for question, cluster_id in zip(questions, cluster_assignment):
        clustered_questions.setdefault(cluster_id, []).append(question)

    return clustered_questions

def top_questions_in_clusters(clustered_questions):
    top_questions = []
    for cluster_id, questions in clustered_questions.items():
        # Count frequency of each question in the cluster
        question_freq = {}
        for question in questions:
            question_freq[question] = question_freq.get(question, 0) + 1

        # Find the most frequent question in the cluster
        most_frequent_question = max(question_freq, key=question_freq.get)
        top_questions.append((most_frequent_question, question_freq[most_frequent_question]))

    # Sort the questions by frequency and select top 10
    top_questions.sort(key=lambda x: x[1], reverse=True)
    return top_questions[:10]




def plot_question_frequencies(top_10_questions, total_questions):
    questions, frequencies = zip(*top_10_questions)
    percentages = [freq / total_questions * 100 for freq in frequencies]

    plt.barh(questions, percentages)
    plt.xlabel('Percentage')
    plt.title(f'Top 10 Frequently Asked Questions - in percentage related to {number_of_answers} asked questions.')
    plt.gca().invert_yaxis()  # To display the highest percentage at the top
    st.pyplot(plt)



def translate_text(text, source_lang, target_langs):
    """
    Translates text from a source language to one or more target languages using Azure Cognitive Services Translator.

    :param text: The text to translate.
    :param source_lang: The source language code (e.g., 'en' for English).
    :param target_langs: A list of target language codes (e.g., ['fr', 'de'] for French and German).
    :param key: Your Azure Translator subscription key.
    :param location: Your Azure Translator resource location.
    :return: The translated text.
    """
    endpoint = "https://api.cognitive.microsofttranslator.com"
    path = '/translate'
    constructed_url = endpoint + path

    params = {
        'api-version': '3.0',
        'from': source_lang,
        'to': target_langs
    }

    headers = {
        'Ocp-Apim-Subscription-Key': key,
        'Ocp-Apim-Subscription-Region': location,
        'Content-type': 'application/json',
        'X-ClientTraceId': str(uuid.uuid4())
    }

    body = [{
        'text': text
    }]

    response = requests.post(constructed_url, params=params, headers=headers, json=body)
    return response.json()






# Header    
def app():

    


    

    gif_url = 'https://upload.wikimedia.org/wikipedia/commons/thumb/2/29/Volvo-Iron-Mark-Black.svg/2048px-Volvo-Iron-Mark-Black.svg.png'

    #st.markdown(f'# <img src="{gif_url}" width="64" height="64"> Volvo Trucks Generative AI Knowledge Assistant', unsafe_allow_html=True)
    st.markdown(f'# Assistente de IA QCSTS para Argus América Latina', unsafe_allow_html=True)
    

    


    
app()


def translate_anything(text, source_language, target_languages):
    # Assume translate_text is a function call to a translation API
    translation_result = translate_text(text, source_language, target_languages)
    # Directly extract the translated text without re-encoding it to JSON
    translated_text = translation_result[0]["translations"][0]["text"]
    
    # Return the translated text directly
    return translated_text

#translate_anything(text, source_language, target_languages)
st.subheader(translate_anything("Um assistente inovador com tecnologia de IA que fornece assistência de diagnóstico técnico rápido e preciso aos técnicos das concessionárias de caminhões.", source_language, target_languages))

# Display the number of questions answered by the chatbot
number_of_answers = count_chatbot_answers()

st.write(translate_anything("""Este aplicativo foi projetado especificamente para auxiliar a equipe de serviço técnico GTT - Qualidade e Satisfação do Cliente LAm, fornecendo assistência de diagnóstico técnico rápido e preciso aos técnicos de revendedores de caminhões, aproveitando um banco de dados de documentos de instruções de carroceria da Volvo e casos Argus resolvidos anteriormente. Até o momento, ela respondeu a """, source_language, target_languages) + str(number_of_answers) + translate_anything(""" perguntas, com uma média de {average} perguntas por sessão.""", source_language, target_languages))

st.write(f"""Este sistema de IA pode interpretar descrições de avarias e recomendar procedimentos e soluções de diagnóstico, agilizando o processo de suporte técnico.
Se não tiver certeza do que perguntar, sinta-se à vontade para clicar no botão **'Sugira-me uma pergunta'** na barra lateral.""")


user_message = st.text_area("Ask your question and press 'Enter'.")
user_input = user_message 

#st.header("Info")




# Function to randomly select a question from the list
def get_random_question():
    questions = [
        "Suggest me some links where I can learn more about Generative AI.",
        "What is Generative AI?",
        "How can Generative AI benefit Volvo Trucks?",
        "What are some applications of Generative AI in the automotive industry?",
        "How does Generative AI compare to other AI technologies?",
        "What are the risks associated with Generative AI?",
        "How can we ensure the ethical use of Generative AI?",
        "What is the learning process for Generative AI?",
        "How can Generative AI be integrated into our existing systems?",
        "What are some notable projects or case studies involving Generative AI?",
        "How can Generative AI contribute to innovation at Volvo Trucks?"
        "How can Generative AI support Volvo Trucks' innovation initiatives?",
        "What distinguishes Generative AI from other forms of artificial intelligence?",
        "How might Generative AI impact the future of automotive design?",
        "What are the ethical considerations when implementing Generative AI?",
        "How could Generative AI enhance Volvo Trucks' operational efficiency?",
        "What are some potential challenges when integrating Generative AI into existing workflows?",
        "How can Generative AI contribute to sustainability efforts within Volvo Trucks?",
        "What are some real-world examples of Generative AI applications in the automotive industry?",
        "How can we ensure data privacy and security when utilizing Generative AI?",
        "What is the learning curve for implementing Generative AI solutions within Volvo Trucks?",
        "How could Generative AI be employed in predictive maintenance for Volvo Trucks?",
        "What are the data requirements for training Generative AI models?",
        "How can Generative AI foster collaboration between different departments within Volvo Trucks?",
        "How might Generative AI influence the development of smart transportation solutions?",
        "What are some potential partnerships or collaborations to accelerate Generative AI adoption within Volvo Trucks?",
        "How can Generative AI be leveraged for customer engagement and personalized experiences?",
        "What are the key metrics to evaluate the success of Generative AI implementations?",
        "How can Generative AI enhance decision-making processes within Volvo Trucks?",
        "What are some best practices for managing and maintaining Generative AI systems?",
        "How can Generative AI support Volvo Trucks in achieving its long-term strategic goals?",
        "How can Generative AI help in the development of autonomous driving technologies at Volvo Trucks?",
        "In what ways can Generative AI be utilized for fuel efficiency and emission reduction in our vehicles?",
        "How can Generative AI support the design process in the development of new truck models?",
        "What are some emerging trends in Generative AI that could be relevant for Volvo Trucks?",
        "How could Generative AI be used to enhance Volvo Trucks' supply chain management?",
        "What are the steps involved in developing a Generative AI project from concept to deployment within Volvo Trucks?",
        "How can Generative AI facilitate more effective data analysis and insights generation?",
        "In what ways can Generative AI help in improving the safety features of Volvo Trucks?",
        "How can Generative AI contribute to Volvo Trucks' vision of a data-driven organization?",
        "What is the potential of Generative AI in enhancing the customer experience and satisfaction?",
        "How can Generative AI be utilized in predictive analytics for better performance monitoring of our trucks?",
        "What type of data is most beneficial for training Generative AI models in an automotive context?",
        "How might Generative AI be used to advance Volvo Trucks' sustainability and environmental goals?",
        "What are the potential cost implications of integrating Generative AI technologies within Volvo Trucks' operations?",
        "How can Generative AI facilitate the development of smart infrastructure for better fleet management?",
        "What collaborations or partnerships might accelerate the adoption and benefits of Generative AI at Volvo Trucks?",
        "How can we ensure the reliability and robustness of Generative AI models in real-world operational settings?",
        "What are the legal and regulatory considerations for the deployment of Generative AI technologies within Volvo Trucks?",
        "How can Generative AI support Volvo Trucks in staying ahead in the competitive landscape of the automotive industry?",
        "How can we measure the ROI of Generative AI projects and initiatives within Volvo Trucks?",
        
        # New questions
        "How can generative AI be used to create more fuel efficient truck designs?",
        "What challenges exist in applying generative AI to automotive supply chain optimization?",
        "How might advances in natural language processing impact the driver experience in Volvo trucks?",
        "What are the risks of over-reliance on generative AI systems in manufacturing?",
        "How can generative AI enhance predictive maintenance for commercial truck fleets?",
        "What ethical considerations should guide the use of generative AI in the automotive industry?",
        "How might generative design be used to rapidly prototype new truck components?",
        "What are the main differences between supervised, unsupervised, and reinforcement learning algorithms?",
        "How can Volvo leverage generative AI to personalize the truck configuration process for customers?",
        "What safety considerations are important when deploying autonomous trucking powered by AI?"

        
    ]
    return random.choice(questions)       
    
    




# ... beginning of the world ...

# If the user has input a message, process it
if st.button(translate_anything('Click here to start the diagnostic!', source_language, target_languages)):
    if user_input:
        prompt = """
You are an AI assistant tasked with supporting our Technical Service team. Your role involves offering rapid and accurate technical diagnostics assistance to truck dealer technicians. This is achieved by utilizing a comprehensive database, which includes Volvo Bodybuilding Instruction documents and records of previously resolved Argus cases, all managed through Azure AI CognitiveSearch.

Your specific task is to analyze malfunction descriptions provided by users and recommend diagnostic procedures and solutions. Focus on the information within the {Solucao} field to guide your recommendations. When a user inputs the following malfunction description: """ + user_input + """, your response should:

1. Interpret the malfunction description accurately.
2. Find the top 3 most relevant correlations in the datasource that can help technicians to propose a solution to the user.
3. Recommend up to three diagnostic procedures and solutions based on the {Solucao} field, prioritizing relevance.
4. Summarize the diagnostic insightfully, incorporating all pertinent details.
5. Present a table summarizing the relevant information, including but not limited to CHASSIS_ID, DTC, FUNCTION_GROUP, PRODUCT_CLASS, Reclamacao, Resumo, SR_NUMBER, and Solucao fields.
#6. Provide links to the source documents or data for further reference.

Your assistance is vital in streamlining the technical support process, making it more efficient for our team and the technicians we support.
"""

        
        
        
        # Add user's message to the chat history
        conversation_history.append(f'User: {user_input}')
        
        # Add user's message to the conversation history
        st.session_state.conversation_history.append({"role": "user", "content": user_input})
    
        # Prepare messages for the OpenAI API call
        messages_to_send = st.session_state.conversation_history + [{"role": "user", "content": user_input}]
        
        
        
        
        
        
    
        with st.spinner(translate_anything('Fetching answer...', source_language, target_languages)):
            try:
                # Call the OpenAI API to get a response
                
                
                completion = client.chat.completions.create(
                messages=[{"role": "user", "content": prompt}],
                model=deployment_id,
                extra_body={
                    "dataSources": [
                        {
                            "type": "AzureCognitiveSearch",
                            "parameters": {
                                "endpoint": search_endpoint,
                                "key": search_key,
                                "indexName": search_index_name,
                                "semanticConfiguration": "default",
                                #"queryType": "semantic",
                                "topNDocuments": 10,
                                "filepathField": "filepath",
                                "titleField": "title",
                                "urlField": "url",
                                "vectorFields": [],
                                "strictness": 4,
                                "fieldsMapping": {
                                  "titleField": "Title",
                                  "urlField": "url",
                                  "filepathField": "filepath",
                                  "contentFields": [
                                    "CHASSIS_ID",
                                    "DTC",
                                    "FUNCTION_GROUP",
                                    "PRODUCT_CLASS",
                                    "Reclamacao",
                                    "Resumo",
                                    "SR_NUMBER",
                                    "Solucao",
                                    
                                    
                                    
                                  ],
                                  "contentFieldsSeparator": "\n"
                                }
                            }
                        }
                    ]
                }
            )
    
    
                # Extract and clean up the answer from the API response
                #answer = str(completion['choices'][0]['message']['content'])
                answer = str(completion.choices[0].message.content)
                answer = translate_anything(answer, source_language, target_languages)
                st.session_state['learning_path'] = answer
                text_to_translate = translate_anything(answer,source_language, target_languages)
                
                # Add the AI's answer to the chat history
                conversation_history.append(f'AI: {answer}')
                
                # Immediately save the answer to session state
                st.session_state['chat_answer'] = answer.strip()
                
    
                # Display the chat history with appropriate icons for user and AI
                for message in conversation_history:
                    if message.startswith('User:'):
                        icon = "https://static-00.iconduck.com/assets.00/chat-bubbles-question-icon-2038x2048-2n7jyhsj.png"
                        text_to_translate=message.replace("User:", "").strip()
                        translation_result = translate_text(text_to_translate, source_language, target_languages)
                        
                        # Safely access the translation result
                        if translation_result and 'translations' in translation_result[0] and len(translation_result[0]['translations']) > 0:
                            translated_text = translation_result[0]['translations'][0]['text']
                            #st.write(translated_text)
                        else:
                            translated_text = "Translation failed."
                        st.markdown(f'<img src="{icon}" width="32" height="32"> {translated_text}', unsafe_allow_html=True)
                        st.session_state['chat_question'] = message
                        
                    else:
                        icon = "/icongpt.png"
                        text_to_translate=message.replace("User:", "").strip()
                        translation_result = translate_text(text_to_translate, source_language, target_languages)
                        # Safely access the translation result
                        if translation_result and 'translations' in translation_result[0] and len(translation_result[0]['translations']) > 0:
                            translated_text = translation_result[0]['translations'][0]['text']
                            st.write(translated_text)
                        else:
                            translated_text = "Translation failed."
                        #st.markdown(f'<img src="{icon}" width="32" height="32"> {translated_text}', unsafe_allow_html=True)
                        st.session_state['chat_answer'] = message
    
                # Increment the response counter
                if 'response_counter' in st.session_state:
                    st.session_state['response_counter'] += 1
                    #st.write("Session Iteractions: " + str(st.session_state['response_counter']))
                    # Make sure to set the chat_answer state after getting the answer
                    chat_answer = st.session_state.get('chat_answer')
                    chat_question = str(st.session_state.get('chat_question'))
    
                    
    
                    # ... the world finish here ...
                    
                    
                    
                    
                    
                    
                    
                    
                    
                    chat_answer = str(st.session_state.get('chat_answer'))
                    chat_question = str(st.session_state.get('chat_question'))
                else:
                    st.session_state['response_counter'] = 1
                    chat_answer = str(st.session_state.get('chat_answer'))
                    chat_question = str(st.session_state.get('chat_question'))
    
            except Exception as e:
                st.error(f"An error occurred: {e}")
    
    
        #st.write("Please remember to share your feedback, to help us to improve this application.")
        
        # Capture the feedback from the user
        #st.write("debug 1: "+ str(chat_answer))
        #feedback = capture_feedback()
        
        #user_feedback = st.text_input("Please provide feedback on your experience: ")
    
        # Display feedback mechanism after an AI response is shown
        #st.session_state['user_feedback']=""
        #display_feedback()
    
        feedback = ""
    



# Sidebar Instructions
#st.sidebar.subheader('Continue to innovate')
#st.sidebar.markdown("""
### Instructions
#Enter your question related to Generative AI in the text box and click 'Get Answer' to retrieve information. You can also copy and paste on the questions you can see here below:""")

# Sidebar with random question suggestion and button to get a new question
#st.sidebar.markdown("### Suggested Questions")
#random_question = get_random_question()
#st.sidebar.write(random_question)
#if st.sidebar.button("Suggest me a question!"):
    #random_question = get_random_question()
    #st.sidebar.write(random_question)
    

# ... inside your Streamlit app ...

# Retrieve questions from Azure Table Storage
questions = retrieve_questions()

# Perform semantic clustering
clustered_questions = cluster_questions(questions)

# Find the top 10 most asked questions
top_10_questions = top_questions_in_clusters(clustered_questions)


with st.expander(translate_anything("Frequently Asked Questions",source_language, target_languages)):
    st.write(translate_anything("Here you have an update list of the 10 most frequently asked questions by our users and how many times they were asked. As you can see, unfortunately, not all questions are related to generative AI. In parenthesis the relative percentage",source_language, target_languages))
    total_questions = number_of_answers  # Assuming this is the total count of all questions asked

    # Display the results with a progressive number and percentage
    for index, (question, freq) in enumerate(top_10_questions, start=1):
        percentage = (freq / total_questions) * 100
        st.write(f"{index}. {question} - {freq} ({percentage:.2f})%")
    plot_question_frequencies(top_10_questions, number_of_answers)


# Log the chat interactions to Azure table
#chat_session_id = str(st.session_state['chat_session_id'])
# Now, when you log to Azure Table, ensure the feedback is correctly passed
chat_feedback = str(st.session_state.get('chat_feedback', {}))  # Safely access the feedback
chat_interactions = st.session_state['response_counter']
#st.write("debug 2: "+ str(chat_answer))    
    

log_to_azure_table(chat_question, chat_answer, chat_feedback, chat_session_id, chat_interactions)    
#st.session_state['chat_session_id'] = generate_new_session_id()

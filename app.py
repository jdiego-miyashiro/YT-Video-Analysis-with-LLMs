import os

import streamlit as st
from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI
from langchain.chains import MapReduceDocumentsChain, ReduceDocumentsChain
from langchain_text_splitters import CharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.prompts import PromptTemplate
from io import StringIO

from dotenv import load_dotenv, find_dotenv
from prompts import initial_chunk_prompt_template as initial_prompt
from prompts import refine_chunk_template as refine_prompt

# Load environment variables from .env file

def format_youtube_transcript(transcript_documents, output_folder='transcripts'):
    # Create a StringIO object to hold the content in memory
    memory_file = StringIO()
    
    # Define the output file name
    chunks = transcript_documents
    title = chunks[0].metadata['title']
    output_file = os.path.join(output_folder,title)
    with open(output_file, 'w') as file:
        # Iterate over each chunk in the list with an index to access the next chunk
        for i in range(len(chunks)):
            # Extract metadata and content from the current chunk
            title = chunks[i].metadata.get('title', 'Unknown Title')
            start_timestamp = chunks[i].metadata.get('start_timestamp', 'Unknown Timestamp')
            page_content = chunks[i].page_content
            
            # Determine the end timestamp
            if i + 1 < len(chunks):
                end_timestamp = chunks[i + 1].metadata.get('start_timestamp', 'Unknown Timestamp')
            else:
                end_timestamp = "End of Video"  # or some appropriate end marker
            
            # Prepare the content to write
            content = (
                f"Video Extract : {title}\n"
                f"[{start_timestamp}---->{end_timestamp}]:\n{page_content}\n"
                f"\n"
            )
            
            # Write the content to both the file and the StringIO object
            file.write(content)
            memory_file.write(content)
    
    # Get the content from the StringIO object as a string
    memory_content = memory_file.getvalue()
    
    # Return the full script content in memory
    return memory_content

def get_youtube_transcript(yt_uri, chunk_size_seconds=60):
    # Load YouTube transcript using the provided URI
    loader = YoutubeLoader.from_youtube_url(
        yt_uri,
        chunk_size_seconds=chunk_size_seconds,
        transcript_format=TranscriptFormat.CHUNKS,
        add_video_info=True,
        
    )
    
    transcript_documents = loader.load()

    formatted_transcript = format_youtube_transcript(transcript_documents)

    return formatted_transcript

def get_transcript_summary(transcript,initial_prompt,refine_prompt):
    llm = ChatOpenAI(temperature=0.2,max_tokens=3096)
        
    text_splitter = RecursiveCharacterTextSplitter(
        # Set a really small chunk size, just to show.
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(transcript)
    documents = text_splitter.create_documents(chunks)
    intial_prompt = PromptTemplate.from_template(initial_prompt)
    refine_chunk_prompt = PromptTemplate.from_template(refine_prompt)
    
    chain = load_summarize_chain(
    llm=llm,
    chain_type="refine",
    question_prompt=intial_prompt,
    refine_prompt=refine_chunk_prompt,
    return_intermediate_steps=False,
    input_key="input_documents",
    output_key="output_text")

    result = chain.invoke({"input_documents": documents}, return_only_outputs=False)
    return result['output_text']


    
def main():
    st.set_page_config(page_title="YouTube Video Analyzer", page_icon=":movie_camera:")

    st.subheader("Analyze and Chat About Your YouTube Videos")


    summary = ''
    
    load_dotenv(override=True)

    #print("Loaded OpenAI API Key:", openai_api_key)
    #print(os.environ)

    # maybe instantiate your llm here 
    
    with st.sidebar:
        st.header("Video Processing")
        
        yt_uri = st.text_input("Paste YouTube URI and click 'Process'", key='yt_uri')
        
        if st.button("Process"):
            with st.spinner("Processing video..."):
                # Call function to get the YouTube summary
                if yt_uri:
                    transcript = get_youtube_transcript(yt_uri)
                    
                    summary = get_transcript_summary(transcript,initial_prompt,refine_prompt)
                    
        
    st.write(summary)
    # Initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    # Display chat messages from history on app rerun
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])  
   
    # Accept user input
    if prompt := st.chat_input("Ask a question about your video:"):
        # Display user message in chat message container 
        with st.chat_message("user"):
            st.markdown(prompt)
        # Add user message to chat history, note the role is user
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Generate response
        response = f"Echo: {prompt}"
        with st.chat_message("assistant"):
            st.markdown(response)
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": response})
                    
if __name__ == '__main__':
    main()
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

from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

def format_youtube_transcript(transcript_documents, output_file='youtube_script.txt'):
    # Create a StringIO object to hold the content in memory
    memory_file = StringIO()
    
    # Define the output file name
    chunks = transcript_documents
    
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

def get_youtube_summary(yt_uri, chunk_size_seconds=45):
    # Load YouTube transcript using the provided URI
    loader = YoutubeLoader.from_youtube_url(
        yt_uri,
        chunk_size_seconds=chunk_size_seconds,
        transcript_format=TranscriptFormat.CHUNKS
    )
    
    transcript_documents = loader.load()

    formatted_transcript = format_youtube_transcript(transcript_documents, output_file='youtube_script.txt')

    return formatted_transcript

def main():
    st.set_page_config(page_title="YouTube Video Analyzer", page_icon=":movie_camera:")

    st.subheader("Analyze and Chat About Your YouTube Videos")

    user_question = st.text_input("Ask a question about your video:")

    summary = ''
    with st.sidebar:
        st.header("Video Processing")
        
        yt_uri = st.text_input("Paste YouTube URI and click 'Process'", key='yt_uri')
        
        if st.button("Process"):
            with st.spinner("Processing video..."):
                # Call function to get the YouTube summary
                if yt_uri:
                    summary = get_youtube_summary(yt_uri)
    st.write(summary)

if __name__ == '__main__':
    main()
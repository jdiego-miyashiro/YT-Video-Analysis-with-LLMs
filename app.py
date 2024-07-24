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
from langchain import hub
from langchain.chains import create_retrieval_chain
from langchain.chains import create_history_aware_retriever

from langchain.chains.combine_documents import create_stuff_documents_chain

from langchain_core.messages import HumanMessage
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from io import StringIO

from prompts import initial_chunk_prompt_template as initial_prompt
from prompts import refine_chunk_template as refine_prompt

# Load environment variables from .env file

def format_youtube_transcript(transcript_documents, output_folder='transcripts'):
    # Create a StringIO object to hold the content in memory
    memory_file = StringIO()
    
    # Define the output file name
    chunks = transcript_documents
    title = chunks[0].metadata['title']
    output_file = os.path.join(output_folder,'a')
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
    return formatted_transcript, transcript_documents

def get_transcript_summary(transcript, transcript_documents, initial_prompt, refine_prompt):
    llm = ChatOpenAI(temperature=0.2, max_tokens=3096)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=500,
        length_function=len,
        is_separator_regex=False,
    )
    
    chunks = text_splitter.split_text(transcript)
    transcript_chunk_documents = text_splitter.create_documents(chunks)
    
    for i, document in enumerate(transcript_chunk_documents):
        document.metadata.update({
            "video_title": transcript_documents[0].metadata['title'],
            "video_url": transcript_documents[0].metadata['source'],
            "video_author": transcript_documents[0].metadata['author']
        })
        
    intial_prompt = PromptTemplate.from_template(initial_prompt)
    refine_chunk_prompt = PromptTemplate.from_template(refine_prompt)
    
    chain = load_summarize_chain(
        llm=llm,
        chain_type="refine",
        question_prompt=intial_prompt,
        refine_prompt=refine_chunk_prompt,
        return_intermediate_steps=False,
        input_key="input_documents",
        output_key="output_text"
    )

    result = chain.invoke({"input_documents": transcript_chunk_documents}, return_only_outputs=False)
    return result['output_text'], transcript_chunk_documents

def initialize_retrieval_chain(transcript_chunk_documents, summary_chunk_documents):
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatOpenAI(temperature=0.2, max_tokens=3096)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    embeddings = OpenAIEmbeddings()
    db = Chroma.from_documents(transcript_chunk_documents + summary_chunk_documents, embeddings)
    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

def main():
    st.set_page_config(page_title="YouTube Video Analyzer", page_icon=":movie_camera:")

    st.subheader("Analyze and Chat About Your YouTube Videos")

    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,
        chunk_overlap=1000,
        length_function=len,
        is_separator_regex=False,
    )

    summary = ''
    load_dotenv(override=True)
    if "retrieval_chain" not in st.session_state:
        retrieval_chain = None  # Initialize retrieval_chain to None
    chat_history = []

    with st.sidebar:
        st.header("Video Processing")
        
        yt_uri = st.text_input("Paste YouTube URI and click 'Process'", key='yt_uri')
        
        if st.button("Process"):
            with st.spinner("Processing video..."):
                if yt_uri:
                    transcript, transcript_documents = get_youtube_transcript(yt_uri)
                    summary, transcript_chunk_documents = get_transcript_summary(transcript, transcript_documents, initial_prompt, refine_prompt)
                    
                    metadata = {
                        'video_title': transcript_documents[0].metadata['title'],
                        'doc_type': 'summary',
                        'video_url': transcript_documents[0].metadata['source'],
                        'video_author': transcript_documents[0].metadata['author']
                    }

                    summary_chunk_documents = summary_splitter.create_documents(
                        summary_splitter.split_text(summary), [metadata] * len(summary_splitter.split_text(summary))
                    )
                    
                    st.session_state.retrieval_chain = initialize_retrieval_chain(transcript_chunk_documents, summary_chunk_documents)

    st.write(summary)
    
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    if prompt := st.chat_input("Ask a question about your video:"):
        with st.chat_message("user"):
            st.markdown(prompt)
        
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        if st.session_state.retrieval_chain:
            ai_msg = st.session_state.retrieval_chain.invoke({"input": prompt, "chat_history": chat_history})
            response = ai_msg["answer"]
            chat_history.extend([HumanMessage(content=prompt), response])
        else:
            response = "The retrieval chain is not initialized. Please process a video first."

        with st.chat_message("assistant"):
            st.markdown(response)
        
        st.session_state.messages.append({"role": "assistant", "content": response})

if __name__ == '__main__':
    main()

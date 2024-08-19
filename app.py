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
from langchain_community.embeddings.sentence_transformer import (
    SentenceTransformerEmbeddings,
)

from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_experimental.graph_transformers import LLMGraphTransformer


from langchain_core.messages import HumanMessage
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from io import StringIO

from prompts import initial_chunk_prompt_template as initial_prompt
from prompts import refine_chunk_template as refine_prompt

from streamlit_agraph import agraph, Node, Edge, Config
from streamlit_agraph.config import Config

from collections import defaultdict

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
    llm = ChatOpenAI(temperature=0.5, max_tokens=3096)
    
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

def create_knowledge_graph(documents,temperature=.2,max_tokens=4096):
    graph_creator_llm = ChatOpenAI(temperature=temperature,max_tokens=max_tokens)

    llm_transformer = LLMGraphTransformer(llm=graph_creator_llm)
    graph_documents = llm_transformer.convert_to_graph_documents(documents)
        
    return graph_documents
def create_streamlit_graph_elements(graph_documents, k=100):
    """
    Constructs lists of nodes and edges for a Streamlit graph from a list of graph documents.
    Ensures that nodes and edges are unique, and only includes the top `k` nodes with the most connections.

    Parameters:
    graph_documents (list): List of documents where each document contains nodes and relationships.
    k (int): Number of top nodes to include based on their degree (number of connections).

    Returns:
    tuple: A tuple containing two lists, one for nodes and one for edges.
    """
    unique_nodes = {}
    unique_edges = {}

    node_degree = defaultdict(int)

    # Calculate the degree of each node
    for doc in graph_documents:
        for relationship in doc.relationships:
            if relationship.source and relationship.target:
                source_id = str(relationship.source.id)
                target_id = str(relationship.target.id)
                node_degree[source_id] += 1
                node_degree[target_id] += 1

    # Determine the top k nodes by degree
    top_k_nodes = set(sorted(node_degree, key=node_degree.get, reverse=True)[:k])

    # Function to scale node size based on degree
    def scale_node_size(degree, min_size=5, max_size=50, min_degree=1, max_degree=max(node_degree.values())):
        return min_size + (degree - min_degree) / (max_degree - min_degree) * (max_size - min_size)

    # Add the top k nodes to the nodes list
    for doc in graph_documents:
        for node in doc.nodes:
            node_id = str(node.id)
            if node_id in top_k_nodes and node_id not in unique_nodes:
                degree = node_degree[node_id]
                unique_nodes[node_id] = Node(
                    id=node_id,
                    label=node.id,
                    size=scale_node_size(degree),  # Scale size based on degree
                    title=node.id,
                    font={"size": 15, "color": "#FFFFFF"}  # White text color for high contrast
                )

    # Add one edge per node pair
    for doc in graph_documents:
        for relationship in doc.relationships:
            source_id = str(relationship.source.id)
            target_id = str(relationship.target.id)
            if source_id in top_k_nodes and target_id in top_k_nodes:
                edge_key = (source_id, target_id)
                if edge_key not in unique_edges:
                    # Select one relationship to represent the edge
                    unique_edges[edge_key] = relationship.type

    # Convert nodes dict to a list
    nodes = list(unique_nodes.values())

    # Convert the unique edges to Edge objects with custom font sizes and colors
    edges = [Edge(
        source=source, 
        target=target, 
        label=label, 
        type="CURVE_SMOOTH", 
        color="#DDDDDD",  # Light gray color for edges to stand out
        font={"size": 5, "color": "#FFFFFF"}  # White text color for edges
    ) for (source, target), label in unique_edges.items()]

    return nodes, edges
def initialize_retrieval_chain(transcript_chunk_documents, summary_chunk_documents):
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    llm = ChatOpenAI(temperature=0.2, max_tokens=2048)
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(transcript_chunk_documents + summary_chunk_documents, embeddings,persist_directory="./chroma_db_st")

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

    load_dotenv(override=True)

    if "retrieval_chain" not in st.session_state:
        st.session_state.retrieval_chain = None  # Initialize retrieval_chain to None

    if "summary" not in st.session_state:
        st.session_state.summary = ''

    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = None


    if "messages" not in st.session_state:
        st.session_state.messages = []

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
                    knowledge_graph = create_knowledge_graph(summary_chunk_documents)
                    st.session_state.knowledge_graph = knowledge_graph
                    st.session_state.summary = summary  # Save summary to session state

    
        
    st.write(st.session_state.summary)  # Display the summary
    if st.session_state.knowledge_graph:
        # Render the graph with a container
        nodes, edges = create_streamlit_graph_elements(st.session_state.knowledge_graph)
        
        for edge in edges:
            edge.font = {"size": 5}  # Adjust the font size here
        # Define the configuration for the graph visualization
        config = Config(
        width=700,  # Adjust based on your Streamlit container
        height=500,  # Adjust based on your Streamlit container
        directed=False,
        physics=True,  # Enable physics to allow dynamic node spreading
        collapsible=False,
        linkLength=2500,  # Increase this to spread nodes apart
        nodeSpacing=1000,  # Increase to spread out nodes
        gravity=-1000,  # Decrease to make nodes less attracted to the center
        repulsion=2000,  # Increase to make nodes repel each other more
        springLength=250,  # Increase to make connected nodes further apart
        springStrength=0.0001,  # Decrease to make connections less strong
        highlightColor="#F7A7A6",
    )
        #config = Config(height=600, width=800, directed=True, nodeHighlightBehavior=True)
        # Render the graph using agraph with the specified configuration
        agraph(nodes=nodes, edges=edges, config=config)
        
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

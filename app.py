import os
import streamlit as st

from dotenv import load_dotenv
from langchain.chains.summarize import load_summarize_chain
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.chains import (
    MapReduceDocumentsChain, 
    ReduceDocumentsChain, 
    create_retrieval_chain, 
    create_history_aware_retriever
)
from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain_community.document_loaders import YoutubeLoader
from langchain_community.document_loaders.youtube import TranscriptFormat
from langchain_core.prompts import PromptTemplate
from langchain import hub
from langchain_community.vectorstores import Neo4jVector
from langchain_community.graphs import Neo4jGraph
from langchain_community.embeddings.sentence_transformer import SentenceTransformerEmbeddings
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_chroma import Chroma
from langchain_core.messages import HumanMessage

from io import StringIO

from prompts import initial_chunk_prompt_template as initial_prompt
from prompts import refine_chunk_template as refine_prompt

from streamlit_agraph import agraph, Node, Edge, Config

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

def scale_node_size(degree, min_size=1, max_size=50, min_degree=1, max_degree=1):
    if max_degree == min_degree:
        return min_size
    return min_size + (degree - min_degree) / (max_degree - min_degree) * (max_size - min_size)

def create_streamlit_graph_elements(graph_documents, k=100, min_size=1, max_size=50):
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

    # Add the top k nodes to the nodes list
    for doc in graph_documents:
        for node in doc.nodes:
            node_id = str(node.id)
            if node_id in top_k_nodes and node_id not in unique_nodes:
                degree = node_degree[node_id]
                unique_nodes[node_id] = Node(
                    id=node_id,
                    label=node.id,
                    size=scale_node_size(degree, min_size, max_size, min(node_degree.values()), max(node_degree.values())),  # Scale size based on degree
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

def initialize_base_rag_retrieval_chain(transcript_chunk_documents, summary_chunk_documents):
    combine_docs_chain = create_chain_to_combine_docs()
    embeddings = OpenAIEmbeddings()

    db = Chroma.from_documents(transcript_chunk_documents + summary_chunk_documents, embeddings,persist_directory="./chroma_db_st")

    retriever = db.as_retriever()
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

def initialize_graphrag_retrieval_chain(graph_documents):
    

    graph = Neo4jGraph()

    # Store to neo4j
    graph.add_graph_documents(
    graph_documents, 
    baseEntityLabel=True, 
    include_source=True
    )

    vector_index = Neo4jVector.from_existing_graph(
        OpenAIEmbeddings(),
        search_type="hybrid",
        node_label="Document",
        text_node_properties=["text"],
        embedding_node_property="embedding"
    )
    retriever = vector_index.as_retriever()
    
    #history_aware_retriever = create_history_aware_retriever(retriever)

    combine_docs_chain = create_chain_to_combine_docs()
    
    retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)
    return retrieval_chain

def create_chain_to_combine_docs(temperature=.2,max_tokens=2048):
    llm = ChatOpenAI(temperature=temperature, max_tokens=max_tokens)
    retrieval_qa_chat_prompt = hub.pull("langchain-ai/retrieval-qa-chat")
    combine_docs_chain = create_stuff_documents_chain(llm, retrieval_qa_chat_prompt)
    return combine_docs_chain

def main():# Set the page configuration (title and icon) for the Streamlit app
    st.set_page_config(page_title="YouTube Video Analyzer", page_icon=":movie_camera:")

    # Display the subheader for the main title of the app
    # st.subheader("Analyze and Chat About Your YouTube Videos")

    # Initialize a text splitter to break down video transcripts into smaller chunks for processing
    summary_splitter = RecursiveCharacterTextSplitter(
        chunk_size=2000,    # Size of each chunk in characters
        chunk_overlap=1000, # Overlap between chunks to ensure context
        length_function=len, # Function used to measure the length of the text
        is_separator_regex=False # Boolean flag for splitting by regex
    )

    # Load environment variables from a .env file (useful for storing API keys, etc.)
    load_dotenv(override=True)

    # Session state variables to persist data across interactions

    # Holds the retrieval chain for processing user queries about the video
    if "retrieval_chain" not in st.session_state:
        st.session_state.retrieval_chain = None  # Initialize to None

    # Stores the summary of the video transcript
    if "summary" not in st.session_state:
        st.session_state.summary = ''  # Initialize to an empty string

    # Stores the knowledge graph generated from the video transcript
    if "knowledge_graph" not in st.session_state:
        st.session_state.knowledge_graph = None  # Initialize to None

    # Stores the chat messages between the user and the AI assistant
    if "messages" not in st.session_state:
        st.session_state.messages = []  # Initialize to an empty list

    # Stores the chat messages between the user and the AI assistant
    if "video_title" not in st.session_state:
        st.session_state.video_title = ''  # Initialize to an empty list

    # Local variable to keep track of chat history during a session
    chat_history = []

    # Sidebar for user inputs and controls
    with st.sidebar:
        # Header for the video processing section
        st.header("Video Processing")

        # Text input for the user to paste the YouTube video URL
        yt_uri = st.text_input("Paste YouTube URI and click 'Process'", key='yt_uri')

        # Button to start processing the video once the URL is provided
        if st.button("Process"):
            with st.spinner("Processing video..."):
                if yt_uri:  # Check if the user has provided a URL
                    # Retrieve the transcript of the YouTube video
                    transcript, transcript_documents = get_youtube_transcript(yt_uri)

                    # Generate a summary of the transcript
                    summary, transcript_chunk_documents = get_transcript_summary(
                        transcript, 
                        transcript_documents, 
                        initial_prompt, 
                        refine_prompt
                    )

                    # Metadata dictionary to store additional information about the video
                    metadata = {
                        'video_title': transcript_documents[0].metadata['title'],
                        'doc_type': 'summary',
                        'video_url': transcript_documents[0].metadata['source'],
                        'video_author': transcript_documents[0].metadata['author']
                    }

                    st.session_state.video_title = transcript_documents[0].metadata['title']
                    

                    # Create summary documents from the split summary text
                    summary_chunk_documents = summary_splitter.create_documents(
                        summary_splitter.split_text(summary), 
                        [metadata] * len(summary_splitter.split_text(summary))
                    )

                    # Generate a knowledge graph based on the video transcript and summary
                    knowledge_graph = create_knowledge_graph(summary_chunk_documents + transcript_chunk_documents)
                    
                    # Initialize the retrieval chain using the created knowledge graph
                    st.session_state.retrieval_chain = initialize_graphrag_retrieval_chain(knowledge_graph)
                    
                    # Save the knowledge graph and summary to session state for persistence
                    st.session_state.knowledge_graph = knowledge_graph
                    st.session_state.summary = summary  # Save the summary to session state

        

        if st.session_state.knowledge_graph:
            # Calculate the total number of nodes in the knowledge graph
            total_nodes = len(set(node.id for doc in st.session_state.knowledge_graph for node in doc.nodes))

            # Slider to select the number of top nodes to display in the knowledge graph
            k = st.slider(
                f"Select the number of nodes to display", 
                min_value=1, 
                max_value=total_nodes, 
                value=min(100, total_nodes)
            )

            # Slider for selecting the minimum node size in the graph visualization
            min_size = st.slider(
                "Minimum node size", 
                min_value=1, 
                max_value=50, 
                value=1
            )

            # Slider for selecting the maximum node size in the graph visualization
            max_size = st.slider(
                "Maximum node size", 
                min_value=10, 
                max_value=100, 
                value=50
            )

    # Subheader for the knowledge graph section
    st.subheader(st.session_state.video_title)

    # Check if the knowledge graph exists in session state before displaying it
    if st.session_state.knowledge_graph:
        # Create the graph elements (nodes and edges) for visualization
        nodes, edges = create_streamlit_graph_elements(st.session_state.knowledge_graph, k=k)
        
        # Adjust the font size of the edges in the graph
        for edge in edges:
            edge.font = {"size": 5}
        
        # Define the configuration for the graph visualization using agraph
        config = Config(
            width=700,          # Set the width of the graph container
            height=350,         # Set the height of the graph container
            directed=False,     # Set to False for undirected graphs
            physics=True,       # Enable physics to allow dynamic node spreading
            collapsible=False,  # Disable collapsibility of nodes
            linkLength=2500,    # Increase link length to spread nodes apart
            nodeSpacing=1000,   # Increase spacing between nodes
            gravity=-1000,      # Decrease gravity to make nodes less attracted to the center
            repulsion=2000,     # Increase repulsion to spread nodes further apart
            springLength=250,   # Increase spring length to make connected nodes further apart
            springStrength=0.0001,  # Decrease spring strength to make connections less strong
            highlightColor="#F7A7A6",  # Set the color for highlighted nodes
        )

        # Render the knowledge graph using agraph with the specified configuration
        agraph(nodes=nodes, edges=edges, config=config)

    # Display the summary of the video transcript
    st.write(st.session_state.summary)

    # Display chat messages from the session state
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Input box for the user to ask a question about the video
    if prompt := st.chat_input("Ask a question about your video:"):
        with st.chat_message("user"):
            st.markdown(prompt)

        # Add the user's question to the session state messages
        st.session_state.messages.append({"role": "user", "content": prompt})

        # Process the user's question using the retrieval chain
        if st.session_state.retrieval_chain:
            ai_msg = st.session_state.retrieval_chain.invoke({"input": prompt, "chat_history": chat_history})
            response = ai_msg["answer"]
            chat_history.extend([HumanMessage(content=prompt), response])
        else:
            response = "The retrieval chain is not initialized. Please process a video first."

        # Display the AI assistant's response
        with st.chat_message("assistant"):
            st.markdown(response)

        # Add the assistant's response to the session state messages
        st.session_state.messages.append({"role": "assistant", "content": response})

   
# Run the main function when the script is executed
if __name__ == '__main__':
    main()

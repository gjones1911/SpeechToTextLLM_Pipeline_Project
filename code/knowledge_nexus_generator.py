from langchain.text_splitter import CharacterTextSplitter

# from langchain.embeddings import HuggingFaceEmbeddings
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer, models
from langchain_community.vectorstores import FAISS
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document

from neo4j import GraphDatabase

from sklearn.metrics.pairwise import cosine_similarity

import joblib
import os
import json
from tqdm import tqdm

import re
import pandas as pd
from datetime import datetime
import torch

# used for basic knowledge graph building
import networkx as nx

from pdf2image import convert_from_path

base_small_embedding_model = "sentence-transformers/all-MiniLM-L6-v2"
base_big_embedding_model = "sentence-transformers/multi-qa-mpnet-base-dot-v1"
best_current_embedding_model_small="thenlper/gte-base"
best_current_embedding_model_big="thenlper/gte-big"
test_neo4j_url = "new4j+s://75ef5069.databases.neo4j.io:7687"
test_neo4j_pwd = "H_iq9moIcpZAMzaPa2SDZM1D0cVyMXvqu4akFfC03PE"






class knowledge_nexus_generator:
    def __init__(self,
                neo4j_username: str= os.getenv("NEO4J_UNAME"),
                neo4j_password: str= os.getenv("NEO4J_PWD"),
                neo4j_url: str= os.getenv("NEO4J_URL"), 
                neo4j_db: str="geralddb",
                **kwargs, 
        ):

        ####### Neo4J knowledge graph stuff
        self.neo4j_username= neo4j_username
        self.neo4j_url= neo4j_url
        self.neo4j_db = neo4j_db
        self.driver = GraphDatabase.driver(self.neo4j_url, auth=(self.neo4j_username, neo4j_password))
        self.vector_store = None
        self.embedding_model_name = None
        self.embeddings_tool = None
        self.documents = None
        self.files = None
        self.base_embeddings = None
        self.nexus_path=None

    def escape_string(self, value):
        """
        Escapes both single quotes and backslashes in a string for Cypher queries.
        If the value is not a string, it is returned as is.
        """
        if isinstance(value, str):
            # First, escape backslashes, then escape single quotes
            return value.replace("\\", "/").replace("'", "\\'")
        return value
    
    ####################################################################################################
    ###  Vector-Store/Embedding Tools
    ####################################################################################################
    def process_pdf_into_document_chunks(self, pdf_path, chunk_size: int=None, chunk_overlap: int=None):
        """Loads and optionally chunks a pdf document into a documents object"""
        loader = PyPDFLoader(pdf_path)
        if chunk_size and chunk_overlap:
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = text_splitter.split_documents(documents)
        else:
            documents = loader.load()
        for chunk in documents:
            chunk.metadata["source"] = os.path.basename(pdf_path)  # Add file name
            if "page" in chunk.metadata:  # Preserve page number if available
                chunk.metadata["page"] = chunk.metadata["page"]
        return documents

    def process_pdfs_to_vector_store_and_embeddings(self, 
        pdf_paths, 
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
        chunk_size=50000, chunk_overlap=200, 
        ):
        self.documents = []
        self.files = []
        for pdf_path in pdf_paths:
            # store base file name
            self.files.append(os.path.basename(pdf_path))
            
            # Load and split the document
            documents = self.process_pdf_into_document_chunks(
                pdf_path, 
                chunk_size, 
                chunk_overlap,
            )
            self.documents.extend(documents)
        
        # Generate embeddings using Hugging Face
        self.embeddings_model_name = embedding_model_name 
        self.embeddings_tool = HuggingFaceEmbeddings(model_name=embedding_model_name)
        self.base_embeddings = torch.stack([torch.tensor(self.embeddings_tool.embed_query(page.page_content)) for page in self.documents])

        # Save to FAISS vector store
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings_tool)
        
        return self.documents, self.vector_store, self.embeddings_tool

    def load_string_doc_update(self, document: str, page: str, source: str, **kwargs):
        new_document = [Document(page_content=document, metadata={"page": page, "source": source})]
        self.documents.extend(new_document)
        self.update_vector_embeddings()
        
    def load_string_doc(self, document: str, page: str, source: str, **kwargs):
        new_document = [Document(page_content=document, metadata={"page": page, "source": source})]
        self.documents.extend(new_document)

    def add_document(self, document: Document):
        self.documents.extend(document)
        
    def add_document_update(self, document: Document):
        self.documents.extend([document])
        self.update_vector_embeddings()
    
    def add_documents(self, new_documents: list):
        # for document in documents:
        #     # try:
        #     #     # page = document.metadata["page"]
        #     #     # source = document.metatdata["source"]
        #     #     # self.load_string_doc(document, page, source)
        #     # except Exception as ex:
        self.documents.extend(new_documents)
        self.update_vector_embeddings()
    
    def load_update_docset(self, pdf_file, chunk_size: int=None, chunk_overlap: int=None):
        new_documents = self.process_pdf_into_document_chunks(pdf_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.add_documents(new_documents)

    def store_local_embeddings_model(self, embedding_model_name, local_model_name, destination_dir):
        """ Stores an embedding model from huggingface as a local sentence transformer model at the given local path by wrapiing 
            it in SentenceTransformer format if needed, and saves it under a specified folder inside a destination directory. 
        
        Args:         
         embedding_model_nameath (str): HF model name or local path. 
         destination_dir (str): Base directory to store the wrapped model. 
         local_model_name_name (str): Subfolder name for the wrapped model. 
         
        Returns: 
            None        
        """    
            
        # make sure the directory exists
        os.makedirs(destination_dir, exist_ok=True)

        # define the full path to where the local model will be stored
        full_model_path = os.path.join(destination_dir, local_model_name)

        word_embedding_model = models.Transformer(embedding_model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        sentence_model.save(full_model_path)
        print(f"💾 Model saved to: {full_model_path}")
        return

    def load_embeddings_model(self, embedding_model_name):
        if os.path.exists(model_name_or_path):
            # If already wrapped and saved, just load it
            if os.path.exists(os.path.join(full_model_path, "modules.json")):
                print(f"✅ Using previously saved wrapped model at: {embedding_model_name}")
                return HuggingFaceEmbeddings(model_name=embedding_model_name)
            else:
                print(f"📦 Wrapping local Hugging Face model from: {embedding_model_name}")
                word_embedding_model = models.Transformer(embedding_model_name)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
                sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                sentence_model.save(embedding_model_name)
                return HuggingFaceEmbeddings(model_name=embedding_model_name)
        else:
            return HuggingFaceEmbeddings(model_name=embedding_model_name)
 

        
    
    def save_knowledge_nexus(self, save_folder, exist_ok=True, **kwargs):
        """
        Saves the vector_store and embedding model name in a specified folder.
        """
        # Create the save folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=exist_ok)
        
        # use base folder name as pre-fix for all files
        base_name = os.path.basename(save_folder)  
        try:
            # create paths to the folder for each object
            vector_store_path = os.path.join(save_folder, f"{base_name}_vector_store")
            embeddings_path = os.path.join(save_folder, f"{base_name}_embedding_model.joblib")
            documents_path = os.path.join(save_folder, f"{base_name}_documents.joblib")
            file_names_path = os.path.join(save_folder, f"{base_name}_file_names.joblib")
            
            # store the objects
            self.vector_store.save_local(vector_store_path)
            joblib.dump(self.embeddings_model_name, embeddings_path)
            joblib.dump(self.documents, documents_path)
            joblib.dump(self.files, file_names_path,)
        except Exception as ex:
            print("There was an error saving the knowledge_nexus!, see below")
            print(f"EXCEPTION: {ex}")
        return
        
        
        
    def load_knowledge_nexus(self, knowledge_nexus_folder, **kwargs):
        """loads a vector store from disk"""
        save_folder = knowledge_nexus_folder
        self.nexus_path=save_folder
        # use base folder name as pre-fix for all files
        base_name = os.path.basename(save_folder)  
        
        # create paths to the folder for each object
        try:
            vector_store_path = os.path.join(save_folder, f"{base_name}_vector_store")
            embeddings_path = os.path.join(save_folder, f"{base_name}_embedding_model.joblib")
            documents_path = os.path.join(save_folder, f"{base_name}_documents.joblib")
            file_names_path = os.path.join(save_folder, f"{base_name}_file_names.joblib")
            
            self.embedding_model_name = joblib.load(embeddings_path)
#             self.embeddings_tool = HuggingFaceEmbeddings(model_name=self.embedding
            self.embeddings_tool = self.load_embeddings_model(self.embedding_model_name)            
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings_tool, allow_dangerous_deserialization=True)
            
            self.documents = joblib.load(documents_path)
            self.files = joblib.load(file_names_path)
        except Exception as ex:
            print("There was an error loading the knowledge_nexus!, see below")
            print(f"EXCEPTION: {ex}")
        return

    def update_vector_embeddings(self, ):
        """This method updates the vector_store and the embeddings. This assumes 
           you have made some change to the documents. So you should call this when you add a new document
           
           Arguments:
                - None
            Returns:
                - None
        """
        if self.documents:
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings_tool)
            self.base_embeddings = torch.stack([torch.tensor(self.embeddings_tool.embed_query(page.page_content)) for page in self.documents])
        else:
            print("Warning! No documents loaded! You must pass either or data_file_TXT or data_file_PDF"
                  " as lists or strings of the indicated document paths to load_pdf_file(), or load_document_list(), but at least one.") 

    def merge_vector_store(self, new_vector_store):
        """
        Merges a new vector store into the current vector store, updates the document set,
        and recalculates vector embeddings.
    
        Args:
            new_vector_store: The vector store object to be merged with the existing vector store.
    
        Raises:
            ValueError: If the new vector store is None or incompatible with the current vector store.
            Exception: If an error occurs during merging or updating embeddings.
        """
        try:
            # Validate the new vector store
            if not new_vector_store:
                raise ValueError("The new vector store cannot be None.")
            if not hasattr(new_vector_store, 'merge_from') or not hasattr(new_vector_store, 'texts'):
                raise ValueError("The new vector store is missing required methods ('merge_from', 'texts').")
    
            # Merge the new vector store into the existing one
            self.vector_store.merge_from(new_vector_store)
    
            # Extract new documents from the incoming vector store and update the document set
            new_docs = [doc for doc in new_vector_store.texts]
            self.documents.extend(new_docs)
    
            # Update vector embeddings for the combined document set
            self.update_vector_embeddings()
    
            print(f"Successfully merged vector store. Added {len(new_docs)} new documents.")
    
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"An error occurred while merging vector stores: {e}")
    
    def clear_docs(self, ):
        """This just empties the ucrrent document list and document names list"""
        self.files = []
        self.documents = []
    
    #######################################################################################################
    def query_similarity_search(self, 
                                query: str, 
                                k: int=3, 
                                min_score: float=20.70, 
                                reverse: bool=False, mode="min", 
                                verbose=False, **kwargs,
                               ):
        """This will query the current instance of the vector_store object for the top k documents
           That are similiar to it with the given similiary threshold as the lower bound of similarity.
           This is mainly for testing the stores for performance, relevance.

            Arguments:
                - query (string): a question or statement to search for similar documents with
                - k (int): optional, represents the upper bound on the number of documents returned
                - min_score (float): optional, the minimum similarity score for a document and the 
                                     query for it to be returned
            Returns:
                - list_documents[documents] , list_similarity_scores[float]
        """
        if verbose:
            print("Query:\n",  query, "\n>>>>>>>>>>>>>>>>\n\n")
        # Query the vector store to find documents with similarity scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        # if verbose:
        #     print("Query result:\n", results, "\n>>>>>>>>>>>>>>>>\n\n")
        
        # simplify result object
        results = [
            (doc, score) for doc, score in results
        ]

        
        def min_tool(score, min_score):
            return score <= min_score
        compare_tool = min_tool
            
        # sort results by similarity score in descending order
        sorted_results = sorted(results, key=lambda x: x[1], reverse=reverse)
        
        # Filter based on the min_score threshold
        filtered_results = [
            (doc, score) for doc, score in sorted_results if compare_tool(score, min_score)
        ]

        
        # Separate documents and scores for the return value
        list_documents = [doc for doc, score in filtered_results]
        list_similarity_scores = [score for doc, score in filtered_results]
        
        return list_documents, list_similarity_scores

    ####################################################################################################
    ###  NEO4J-Knowledge Graph Tools
    #################################################################################################### 
    
    #################################################################################################### 
    ##########################    *** TEXT WORK ***
    def create_text_node(self, tx, node_label, text, cid, source_file, page, next_chunk_id):
        
        tx.run(
            f"""
            CREATE (n:{node_label} {{id: $id, content: $content, source_file: $source_file, page: $page}})
            """,
            id=cid,
            content=text,
            source_file=source_file,
            page=page
        )

        return

    def delete_all_nodes_in_batches(self, driver=None, database=None, batch_size=100):
        """
        Delete all nodes and relationships in batches to prevent out-of-memory errors.
        
        Args:
        - driver: The Neo4j driver instance.
        - batch_size (int): The number of nodes to delete in each batch.
        """
        
        # if no driver provided use the one set when this instance is created
        driver = driver if driver else self.driver
        query = """
        MATCH (n)
        WITH n LIMIT $batch_size
        DETACH DELETE n
        RETURN count(n)
        """
        # if no database was given use the one set when this instance was created
        if not database:
            database = self.neo4j_db
        with self.driver.session(database=database) as session:
            deleted = batch_size
            while deleted > 0:
                try:
                    result = session.run(query, batch_size=batch_size)
                    deleted = result.single()[0]
                    print(f"Deleted {deleted} nodes")
                except Exception as e:
                    print(f"Error during deletion: {str(e)}")
                    break
    
    def quick_query(self, tx, cypher_query, **kwargs):
        results = tx.run(cypher_query)
    
    def quick_cypher_query_(self, cypher_query, **kwargs):
        with self.driver.session(database=self.neo4j_db) as session:
            return session.execute_write(self.quick_query, cypher_query)
    
    def create_sequential_relationships(self, tx, 
            node_label, page_property="page", source_property="source_file",
            relationship_name="FOLLOWED_BY"):
        """
        Creates sequential relationships (e.g., "FOLLOWED_BY") between nodes with consecutive `page_property`
        values and the same `source_property`.

        Args:
            tx: The Neo4j transaction object.
            node_label (str): Label of the nodes.
            page_property (str): Property name representing the page or sequence identifier.
            source_property (str): Property name representing the source (e.g., document name).
            relationship_name (str): Name of the relationship to create (default: "FOLLOWED_BY").
        """
        query = f"""
            MATCH (a:{node_label}), (b:{node_label})
            WHERE a.{source_property} = b.{source_property}
            AND a.{page_property} = b.{page_property} - 1
            MERGE (a)-[:{relationship_name} {{source: a.{source_property}}}]->(b)
        """
        tx.run(query)
    
    def create_sequential_chunk_page_relations(self, 
            node_label, page_property="page", source_property="source_file",
            relationship_name="FOLLOWED_BY",
            database='neo4j',
        ):

        with self.driver.session(database=database) as session:
            session.execute_write(
                    self.create_sequential_relationships, node_label=node_label, 
                    page_property=page_property, source_property=source_property,
                    relationship_name=relationship_name
            )

    def create_generic_relation(self, 
            tx, 
            node_label_a, node_label_b, 
            identifier_label_a, identifier_label_b, 
            identifier_value_a, identifier_value_b, 
            relationship_name='RELATED_TO'
        ):
        """
        Desc:
            Creates a relationship of a given name between two nodes.

        Args:
            tx: The Neo4j transaction object.
            node_label_a (str): Label of the first node.
            node_label_b (str): Label of the second node.
            identifier_label_a (str): Property name used to identify the first node.
            identifier_label_b (str): Property name used to identify the second node.
            identifier_value_a: Property value of the first node identifier.
            identifier_value_b: Property value of the second node identifier.
            relationship_name (str): optional, Name of the relationship to create. Defaults to 'RELATED_TO'

        Example:
            create_relationship(
                tx,
                "Person", "Car",
                "name", "license_plate",
                "Alice", "XYZ-123",
                "OWNS"
            )
        """
        
        query = f"""
            MATCH (a:{node_label_a} {{{identifier_label_a}: $identifier_value_a}}),
                (b:{node_label_b} {{{identifier_label_b}: $identifier_value_b}})
            MERGE (a)-[:{relationship_name}]->(b)
        """
        tx.run(
            query,
            identifier_value_a=identifier_value_a,
            identifier_value_b=identifier_value_b
        )

    def create_nodes_from_documents(self, documents, node_label, database, create_sequential_edges=True):
        with self.driver.session(database=database) as session:
            for cid, chunk in enumerate(documents):
                source_file = chunk.metadata.get("source", "Unknown")
                page_number = chunk.metadata.get("page", "Unknown")
                text = chunk.page_content
                
                # Determine the ID of the next chunk
                if create_sequential_edges:
                    next_chunk_id = cid + 1 if cid + 1 < len(documents) else None
                else:
                    next_chunk_id = None

                session.execute_write(self.create_text_node, node_label, text, cid, source_file, page_number, next_chunk_id)
        print("Neo4j database populated with text chunks.")
        return 
    
    def create_chunk_similarity_relation(self, tx, source_id, target_id, similarity_score):
        # Create SIMILAR_TO relationship
        tx.run(
            """
            MATCH (a:CHUNK {id: $source_id}), (b:CHUNK {id: $target_id})
            MERGE (a)-[:SIMILAR_TO {score: $score}]->(b)
            """,
            source_id=source_id,
            target_id=target_id,
            score=similarity_score
        )

    def create_chunk_similarity_relationships(self, documents, database, similarity_threshold=.75):
        # Step 1:  Generate embeddings for all text chunks
        texts = [chunk.page_content for chunk in documents]
        embeddings = self.embeddings_tool.embed_documents(texts)
        added = 0
        with self.driver.session(database=database) as session:
            # Step 2: Create SIMILAR_TO edges based on cosine similarity
            similarity_matrix = cosine_similarity(embeddings)
            for source_id in range(len(similarity_matrix)):
                for target_id in range(source_id+1, len(similarity_matrix)): # avoid self edges
                    similarity_score = similarity_matrix[source_id][target_id]
                    # print(f"Simularity score between nodes: {source_id} vs {target_id}= {similarity_score:.2f}\n\n")
                    if similarity_score >= similarity_threshold:
                        session.execute_write(self.create_chunk_similarity_relation, source_id, target_id, similarity_score)
                        added += 1

        print(f"Neo4j database CHUNK nodes SIMILAR_TO had {added} relationships created with lb: {similarity_threshold:.2f}.")
    
    #################################################################################################### 
    ##########################    *** Data-Frame Work ***
    def create_node_query_from_row(self, row, node_label, place_holder="'PLACEHOLDER'"):
        properties = ", ".join(
                # Handle datetime objects
                f"`{col.lower()}`: datetime('{row[col].isoformat()}')" if isinstance(row[col], datetime) and not pd.isnull(row[col])
                # Handle lists
                else f"`{col.lower()}`: [{', '.join(map(repr, row[col]))}]" if isinstance(row[col], list)
                # Handle missing values (NaT, NaN, None)
                else f"`{col.lower()}`: {place_holder}" if pd.isnull(row[col])
                # Handle strings
                else f"`{col.lower()}`: '{self.escape_string(row[col])}'" if isinstance(row[col], str)
                # Handle other types (int, float, etc.)
                else f"`{col.lower()}`: {row[col]}"
                for col in row.index
            )
        # Create the Cypher query for node creation
        query = f"CREATE (n:{node_label} {{{properties}}})"
        return query


    def create_query_batch(self, rows, node_label, place_holder="'PLACEHOLDER'"):
        """
        Create a batch of queries for node creation from a list of rows.
        """
        queries = []
        # create node creation queries for each row in the df
        for _, row in rows.iterrows():
            queries.append(self.create_node_query_from_row(row=row, node_label=node_label, place_holder=place_holder))
        return queries

    def create_node_batch_from_queries(self, session, queries):
        results = []
        with session.begin_transaction() as tx:
            for query in queries:
                result = tx.run(query)
                results.append(result)
        return results
    
    def create_nodes_from_df(self, df, node_label, database, batch_size, place_holder="'PLACEHOLDER'"):
        """
            Create nodes in Neo4j from a pandas DataFrame using batch queries for efficiency.
            
            Args:
            - df (pd.DataFrame): The DataFrame containing the data.
            - node_label (str): The label for the nodes in Neo4j.
            - driver: Neo4j GraphDatabase driver.
            - database (str): The name of the Neo4j database where nodes should be created.
            - batch_size (int): Number of queries to execute in one transaction for efficiency.
            - place_holder (string, float, None): optional, used to represent missing/na data in graph nodes 
            
            Returns:
            - None: Executes the Cypher queries to create nodes.
        """
        # Create a session to the DB
        
        with self.driver.session(database=database) as session:
            # create a node creation query for each a batch-size set of rows 
            # in the df where the columns are used as properties of the node
            # use the tqdm to show the progress of the batching
            for i in tqdm(range(0, len(df), batch_size), total=(len(df) // batch_size) + 1, desc="Creating nodes"):
                batch = df.iloc[i:i+batch_size]
                queries = self.create_query_batch(batch, node_label, place_holder=place_holder)
                results = self.create_node_batch_from_queries(session, queries)
        
        print("Node creation completed.")
        return
    
    def remove_nodes_by_label(self, tx, label):
        """
        Removes all nodes of a given type (label) along with their relationships.

        Args:
            tx: The Neo4j transaction object.
            label (str): The label of the nodes to remove.
        """
        query = f"""
            MATCH (n:{label})
            DETACH DELETE n
        """
        tx.run(query)


    def remove_the_labeled_nodes(self, label, database='neo4j'):
        with self.driver.session(database=database) as session:
            results = session.execute_write(self.remove_nodes_by_label, label)
        return results


    def create_relationships(self, df, node_label1, node_label2, attribute1, attribute2, relationship_type, database, batch_size=1000):
        """
        Create relationships between two nodes in batches using all other DataFrame columns as relationship properties.
        
        Args:
        - driver: The Neo4j driver instance.
        - df: A pandas DataFrame containing the data for relationships.
        - node_label1 (str): The label for the first node type (e.g., 'PATIENT').
        - node_label2 (str): The label for the second node type (e.g., 'DIAGNOSIS').
        - attribute1 (str): The column in the DataFrame to match the first node's property (e.g., 'patient_id').
        - attribute2 (str): The column in the DataFrame to match the second node's property (e.g., 'dx_cd_desc').
        - relationship_type (str): The type of relationship to create (e.g., 'HAS_DIAGNOSIS').
        - database (str): The Neo4j database where relationships should be created.
        - batch_size (int): The number of relationships to create per batch.
        """
        driver = self.driver
        # Get all column names except for the ones used to match nodes
        relationship_properties = [col for col in df.columns if col not in [attribute1, attribute2]]
        
        if not relationship_properties:
            print("Warning: No properties to add to relationships. Creating relationships without properties.")
        
        # Cypher query template to create the relationship
        query_create_relationship = f"""
        MATCH (n1:{node_label1} {{{attribute1.lower()}: $value1}}), (n2:{node_label2} {{{attribute2.lower()}: $value2}})
        MERGE (n1)-[r:{relationship_type} {{ {', '.join([f'{prop}: $' + prop for prop in relationship_properties])} }}]->(n2)
        """ if relationship_properties else f"""
        MATCH (n1:{node_label1} {{{attribute1.lower()}: $value1}}), (n2:{node_label2} {{{attribute2.lower()}: $value2}})
        MERGE (n1)-[r:{relationship_type}]->(n2)
        """
        
        total_batches = len(df) // batch_size + 1  # Total number of batches
        
        with driver.session(database=database) as session:
            # tqdm progress bar for batch processing
            for i in tqdm(range(0, len(df), batch_size), total=total_batches, desc="Creating Relationships"):
                # Batch processing
                batch = df.iloc[i:i + batch_size]
                
                for index, row in batch.iterrows():
                    value1 = row[attribute1]
                    value2 = row[attribute2]
                    
                    # Collect relationship properties from all columns except attribute1 and attribute2
                    props = {prop: row[prop] for prop in relationship_properties}
                    
                    try:
                        # Log the full Cypher query being executed
                        props_string = ', '.join([f"{key}: '{value}'" if isinstance(value, str) else f"{key}: {value}" for key, value in props.items()])
                        cypher_query = f"""
                        MATCH (n1:{node_label1} {{{attribute1}: '{value1}'}}), (n2:{node_label2} {{{attribute2}: '{value2}'}})
                        MERGE (n1)-[r:{relationship_type} {{{props_string}}}]->(n2)
                        """
                        # print(f"Executing Cypher query for row {index}:\n{cypher_query}")
                        
                        # Run the Cypher query to create the relationship
                        session.run(query_create_relationship, {
                            'value1': value1,
                            'value2': value2,
                            **props
                        })
    
                    except Exception as e:
                        print(f"Error processing row {index}: {e}")
                        continue
    
                print(f"Processed batch from {i} to {i + len(batch)}")


    def query_for_summary_chunks(self, database, user_text: str, query_text: str, score_threshold: float, num_results: int):
        query_summary = """CALL db.index.fulltext.queryNodes(
                          "contentIndex", $query_text
                         
                        ) 
                        YIELD node, score
                        WHERE score >= $score_threshold
                        MATCH (node)-[:SUMMARIZES]->(related:CHUNK)
                        WITH node.source AS source, node.page AS page, node.content AS summary_content, score, collect(related.content) AS summarized_chunks
                        ORDER BY score DESC
                        LIMIT $num_results
                        RETURN source, page, summary_content, summarized_chunks;
                    """
        
        # attempt to find a summary chunk that matches and if found the chunk it summarizes
        with self.driver.session(database=database) as session:
            query_results =  session.run(
                query_summary, 
                query_text=query_text, 
                score_threshold=score_threshold, 
                num_results=num_results
            )
    
            
            results_strings = []
            rstr = ""
            chunks_ret = {}
            statement = f"The above is a summary of the below information. If it is useful in answering or responding to:\n\n{user_text}\n\n,utilize the below text to answer the user question.\n"
            
            # Process and print results
            for record in query_results:
                # print(f"Record: {record}")
                page, source = record["page"], record["source"]
                rtxt = f"source: {source}, page: {page}\n\n" + record["summary_content"] + f"\n{statement}\n\t\tOriginal Text:\n\n"
                rstr += rtxt
                results_strings.append(record["summary_content"])
                chunks_ret[rtxt] = []
                # print("summary_text:\n", record["summary_content"], "-"*50 + "\n" + "-"*50 + "\n\n")
                # print(statement)
                # print("Summarized Text Chunks:", record["summarized_chunks"])
                for summarized_chunk in record["summarized_chunks"]:
                    rstr += summarized_chunk + "\n" + "-"*50 + "-"*50 + "\n\n"
                    results_strings.append(summarized_chunk)
                    chunks_ret[rtxt].append(summarized_chunk)
        return rstr, results_strings, chunks_ret


    def query_for_chunks(self, database, user_text: str, query_text: str, score_threshold: float, num_results: int):
        associated_chunk_query = """CALL db.index.fulltext.queryNodes(
                  "contentIndex", $query_text
                ) 
                YIELD node, score
                WHERE score >= $score_threshold
                
                // Match FOLLOWED_BY relationships in both directions
                OPTIONAL MATCH (node)-[:FOLLOWED_BY]->(next)
                OPTIONAL MATCH (node)<-[:FOLLOWED_BY]-(prev)
                
                // Aggregate results before returning
                WITH node, score, collect(next.content) AS followed_by_next, collect(prev.content) AS followed_by_prev
                RETURN node.content AS chunk_content, score, node.page AS page, node.source as source, followed_by_next, followed_by_prev
                ORDER BY score DESC
                LIMIT $num_results;
            """
        database = database if database else self.neo4j_db
        query_text = f'"{query_text}"'
        print(f"query_text:\n{query_text}\n")
        
        # attempt to find a regular chunk that matches and if found the chunk it summarizes
        with self.driver.session(database=database) as session:
            query_results =  session.run(
                associated_chunk_query, 
                query_text=query_text, 
                score_threshold=score_threshold, 
                num_results=num_results
            )

            
            results_strings = []
            rstr = ""
            statement = "The set of text consists of a set of sequential pages from indicated document.\n"
            
            # print(query_results)
            
            # Process and print results
            for record in query_results:
                # print(f"Record: {record}")
                
                page, source = record["page"], record["source"]
                chunk = record["chunk_content"]
                

                # Create previous chunk statment
                for previous_chunk in record["followed_by_prev"]:
                    rstr += f"\n{statement}\n\t\tDocument Text:\n\n" + "-"*50 + "\n\n" + f"source: {source}, page: {page-1}\n\n" + previous_chunk + "\n" + "-"*50 + "\n\n"
                    results_strings.append(previous_chunk)

                # Create current chunk statment
                rstr += f"Document Text: \t\t\t" + f"source: {source}, page: {page}\n\n" + f"{chunk}" + "\n" + "-"*50 + "\n\n"
                results_strings.append(chunk)

               
                for next_chunk in record["followed_by_next"]:
                    rstr += f"source: {source}, page: {page+1}\n\n" + next_chunk + "\n" + "-"*50 + "\n\n"
                    results_strings.append(next_chunk)
                    
            
        return rstr, results_strings

    def sanitize_query_text2(self, query_text):
        # Remove newlines and excess whitespace
        query_text = re.sub(r'\s+', ' ', query_text)
        # Escape Lucene special characters
        lucene_special_chars = r'[\+\-\!\(\)\{\}\[\]\^"~\*\?:\\]'
        query_text = re.sub(lucene_special_chars, r'\\\g<0>', query_text)
        return query_text.strip()

    def sanitize_query_text(self, query_text):
        # replacements = {
        #     "/":" ",
        #     # ".": "_",
        #     # " ": "_",
        #     # "-": "_",
        # }
        # for o,n in replacements.items():
        #     query_text = query_text.replace(o,n)
        return query_text.strip()
    
    def run_summary_text_query(self, database, user_text: str, query_text: str, score_threshold: float, num_results: int):
        """This will run a query on the neo4j DB to find any chunks summarized by the indicated summary_chunks content
        """
        database = database if database else self.neo4j_db
        query_text = f'"{query_text}"'
        # print("Query text: ", query_text)
        rstr = ""
        result_string_list = []
        rstr_summary, results_strings_summary, chunks_summary_dict = self.query_for_summary_chunks(database, 
                                                                         user_text, query_text, 
                                                                         score_threshold, num_results)
        for summary_statement in chunks_summary_dict:
            # print(summary_statement)
            rstr += summary_statement
            result_string_list += [summary_statement]
            for chunk in chunks_summary_dict[summary_statement]:
                chunk = self.sanitize_query_text(chunk)
                # print(f"\n\n\t\t\t***CHUNKS2222***\n\n{chunk}" + "-"*50 + "\n" + "-"*50 + "\n")
                rstring, results_strings_chunks = self.query_for_chunks(database, 
                                                                   user_text, chunk, 
                                                                   score_threshold, num_results)
                rstr += "\n\n\t\t\t***CHUNKS***\n\n" + rstring + "\n" + "-"*50 + "\n" + "-"*50 +"\n\n"
                results_strings = results_strings_summary + results_strings_chunks
                result_string_list += [rstring]
        
        return result_string_list, rstr



    def query_summary_texts_process_into_knowledge_statement(self, database, user_query: str, query_text: str, score_threshold: float, num_results: int):

        database = database if database else self.neo4j_db
        
        # run the query to find the knowledge node and it's related information
        results_strings, results_string = self.run_summary_text_query(database, user_query, query_text, score_threshold, num_results)
        return results_strings, results_string

# uses nx to build graphs instead of neo4j
class KnowledgeNexusGeneratorNX:
    def __init__(self, 
                 **kwargs):
       
        self.G = None
        self.vector_store = None
        self.embedding_model_name = None
        self.embeddings_tool = None
        self.documents = []
        self.files = None
        self.base_embeddings = None
        self.nexus_path = None
        self.docset = []
        # self.doc_names = joblib.load(file_paths["doc_names"])
        # self.loaded_file_str = self.doc_delim.join(self.doc_names)
        self.doc_names = None
        self.loaded_file_str = None
        self.summary_dict={}

    ###########################################################################
    ### Utility Methods
    ###########################################################################
    @staticmethod
    def initialize_folder(folder_path, verbose=True):
        """
        Checks if a folder exists at the specified path. If it doesn't exist, creates the folder.

        Args:
            folder_path (str): The path to the folder to be checked/created.
        """
        if not os.path.exists(folder_path):
            try:
                os.makedirs(folder_path)
                print(f"Folder created at: {folder_path}")
            except Exception as e:
                print(f"Error creating folder at {folder_path}: {e}")
        else:
            print(f"Folder already exists at: {folder_path}")

    ####################################################################################################
    ###  Vector-Store/Embedding Tools
    ########################################################################################
    ############
    def process_pdf_into_document_chunks(self, pdf_path, chunk_size: int=None, chunk_overlap: int=None):
        """Loads and optionally chunks a pdf document into a documents object"""
        loader = PyPDFLoader(pdf_path)
        if chunk_size and chunk_overlap:
            documents = loader.load()
            text_splitter = CharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
            documents = text_splitter.split_documents(documents)
        else:
            documents = loader.load()
        for chunk in documents:
            chunk.metadata["source"] = os.path.basename(pdf_path)  # Add file name
            if "page" in chunk.metadata:  # Preserve page number if available
                chunk.metadata["page"] = chunk.metadata["page"]
        return documents

    # def store_local_embeddings_model(self, embedding_model_name, local_model_name, destination_dir):
    #     """ Stores an embedding model from huggingface as a local sentence transformer model at the given local path by wrapiing 
    #         it in SentenceTransformer format if needed, and saves it under a specified folder inside a destination directory. 
        
    #     Args:         
    #      embedding_model_nameath (str): HF model name or local path. 
    #      destination_dir (str): Base directory to store the wrapped model. 
    #      local_model_name_name (str): Subfolder name for the wrapped model. 
         
    #     Returns: 
    #         None        
    #     """    
            
    #     # make sure the directory exists
    #     os.makedirs(destination_dir, exist_ok=True)

    #     # define the full path to where the local model will be stored
    #     full_model_path = os.path.join(destination_dir, local_model_name)

    #     word_embedding_model = models.Transformer(embedding_model_name)
    #     pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
    #     sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
    #     sentence_model.save(full_model_path)
    #     print(f"💾 Model saved to: {full_model_path}")
    #     return

    
    def load_embeddings_model(self, embedding_model_name):
        print(f"loading embedding model: {embedding_model_name}")
        if os.path.exists(embedding_model_name):
            # If already wrapped and saved, just load it
            if os.path.exists(os.path.join(embedding_model_name, "modules.json")):
                print(f"✅ Using previously saved wrapped model at: {embedding_model_name}")
                return HuggingFaceEmbeddings(model_name=embedding_model_name)
            else:
                print(f"📦 Wrapping local Hugging Face model from: {embedding_model_name}")
                word_embedding_model = models.Transformer(embedding_model_name)
                pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
                sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
                sentence_model.save(embedding_model_name)
                return HuggingFaceEmbeddings(model_name=embedding_model_name)
        else:
            return HuggingFaceEmbeddings(model_name=embedding_model_name)

    def process_docset_to_vector_store_and_embeddings(self, 
            docset, embedding_model=best_current_embedding_model_small,    
            files="", **kwargs,
        ):
        self.documents = docset
        self.files = [files]
        # Generate embeddings using Hugging Face
        self.embeddings_model_name = embedding_model
        # self.embeddings_tool = HuggingFaceEmbeddings(model_name=embedding_model)
        self.embeddings_tool = self.load_embeddings_model(self.embedding_model_name)
        self.base_embeddings = torch.stack([torch.tensor(self.embeddings_tool.embed_query(page.page_content)) for page in self.documents])

        # Save to FAISS vector store
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings_tool)
        
        return self.documents, self.vector_store, self.embeddings_tool
    
    def process_pdfs_to_vector_store_and_embeddings(self, 
        pdf_paths, 
        embedding_model_name="sentence-transformers/all-MiniLM-L6-v2", 
        chunk_size=50000, chunk_overlap=200, 
        ):
        self.documents = []
        self.files = []
        for pdf_path in pdf_paths:
            # store base file name
            self.files.append(os.path.basename(pdf_path))
            
            # Load and split the document
            documents = self.process_pdf_into_document_chunks(
                pdf_path, 
                chunk_size, 
                chunk_overlap,
            )
            self.documents.extend(documents)

        # Generate embeddings using Hugging Face
        self.embeddings_model_name = embedding_model_name 
        self.embeddings_tool = self.load_embeddings_model(self.embeddings_model_name)
        self.base_embeddings = torch.stack([torch.tensor(self.embeddings_tool.embed_query(page.page_content)) for page in self.documents])

        # Save to FAISS vector store
        self.vector_store = FAISS.from_documents(self.documents, self.embeddings_tool)
        
        return self.documents, self.vector_store, self.embeddings_tool

    def load_string_doc_update(self, document: str, page: str, source: str, **kwargs):
        new_document = [Document(page_content=document, metadata={"page": page, "source": source})]
        self.documents.extend(new_document)
        self.update_vector_embeddings()
        
    def load_string_doc(self, document: str, page: str, source: str, **kwargs):
        new_document = [Document(page_content=document, metadata={"page": page, "source": source})]
        self.documents.extend(new_document)

    def add_document(self, document: Document):
        self.documents.extend(document)
        
    def add_document_update(self, document: Document):
        self.documents.extend([document])
        self.update_vector_embeddings()
    
    def add_documents(self, new_documents: list):
        self.documents.extend(new_documents)
        self.update_vector_embeddings()
    
    def load_update_docset(self, pdf_file, chunk_size: int=None, chunk_overlap: int=None):
        new_documents = self.process_pdf_into_document_chunks(pdf_file, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        self.add_documents(new_documents)
    

    ###########################################################################
    ###       Storage and Retrieval of Knowledge states
    ##############################################################
    def store_local_embeddings_model(self, embedding_model_name, local_model_name, destination_dir):
        """ Stores an embedding model from huggingface as a local sentence transformer model at the given local path by wrapiing 
            it in SentenceTransformer format if needed, and saves it under a specified folder inside a destination directory. 
        
        Args:         
         embedding_model_nameath (str): HF model name or local path. 
         destination_dir (str): Base directory to store the wrapped model. 
         local_model_name_name (str): Subfolder name for the wrapped model. 
         
        Returns: 
            None        
        """    
            
        # make sure the directory exists
        os.makedirs(destination_dir, exist_ok=True)

        # define the full path to where the local model will be stored
        full_model_path = os.path.join(destination_dir, local_model_name)

        word_embedding_model = models.Transformer(embedding_model_name)
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(), pooling_mode='mean')
        sentence_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        
        sentence_model.save(full_model_path)
        print(f"💾 Model saved to: {full_model_path}")
        return
   

    def save_knowledge_nexus(self, save_folder, exist_ok=True, **kwargs):
        """
        Saves the vector_store and embedding model name in a specified folder.
        """
        # Create the save folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=exist_ok)
        
        # use base folder name as pre-fix for all files
        base_name = os.path.basename(save_folder)  
        try:
            # create paths to the folder for each object
            vector_store_path = os.path.join(save_folder, f"{base_name}_vector_store")
            embeddings_path = os.path.join(save_folder, f"{base_name}_embedding_model.joblib")
            documents_path = os.path.join(save_folder, f"{base_name}_documents.joblib")
            file_names_path = os.path.join(save_folder, f"{base_name}_file_names.joblib")
            graph_names_path = os.path.join(save_folder, f"{base_name}_knowledge_graph.joblib")
            
            # store the objects
            self.vector_store.save_local(vector_store_path)
            joblib.dump(self.embeddings_model_name, embeddings_path)
            joblib.dump(self.documents, documents_path)
            joblib.dump(self.files, file_names_path)
            joblib.dump(self.G, graph_names_path)
        except Exception as ex:
            print("There was an error saving the knowledge_nexus!, see below")
            print(f"EXCEPTION: {ex}")
        return
        
    def load_knowledge_nexus(self, knowledge_nexus_folder, **kwargs):
        """loads a vector store from disk"""
        save_folder = knowledge_nexus_folder
        self.nexus_path=save_folder
        # use base folder name as pre-fix for all files
        base_name = os.path.basename(save_folder)  
        
        # create paths to the folder for each object
        try:
            vector_store_path = os.path.join(save_folder, f"{base_name}_vector_store")
            embeddings_path = os.path.join(save_folder, f"{base_name}_embedding_model.joblib")
            documents_path = os.path.join(save_folder, f"{base_name}_documents.joblib")
            file_names_path = os.path.join(save_folder, f"{base_name}_file_names.joblib")
            graph_names_path = os.path.join(save_folder, f"{base_name}_knowledge_graph.joblib")
            
            self.embedding_model_name = joblib.load(embeddings_path)
            # self.embeddings_tool = HuggingFaceEmbeddings(model_name=self.embedding_model_name)
            self.embeddings_tool = self.load_embeddings_model(self.embedding_model_name)
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings_tool, allow_dangerous_deserialization=True)
            
            self.documents = joblib.load(documents_path)
            self.files = joblib.load(file_names_path)
            self.G = joblib.load(file_names_path)
        except Exception as ex:
            print("There was an error loading the knowledge_nexus!, see below")
            print(f"EXCEPTION: {ex}")
        return


    ############
    def save_knowledge_nexus(self, save_folder, exist_ok=True, **kwargs):
        """
        Saves the vector_store and embedding model name in a specified folder.
        """
        # Create the save folder if it doesn't exist
        os.makedirs(save_folder, exist_ok=exist_ok)
        
        # use base folder name as pre-fix for all files
        base_name = os.path.basename(save_folder)  
        try:
            # create paths to the folder for each object
            vector_store_path = os.path.join(save_folder, f"{base_name}_vector_store")
            embeddings_path = os.path.join(save_folder, f"{base_name}_embedding_model.joblib")
            documents_path = os.path.join(save_folder, f"{base_name}_documents.joblib")
            file_names_path = os.path.join(save_folder, f"{base_name}_file_names.joblib")
            graph_names_path = os.path.join(save_folder, f"{base_name}_knowledge_graph.joblib")
            
            # store the objects
            self.vector_store.save_local(vector_store_path)
            joblib.dump(self.embeddings_model_name, embeddings_path)
            joblib.dump(self.documents, documents_path)
            joblib.dump(self.files, file_names_path)
            joblib.dump(self.G, graph_names_path)
        except Exception as ex:
            print("There was an error saving the knowledge_nexus!, see below")
            print(f"EXCEPTION: {ex}")
        return
        
        
        
    def load_knowledge_nexus(self, knowledge_nexus_folder, **kwargs):
        """loads a vector store from disk"""
        save_folder = knowledge_nexus_folder
        self.nexus_path=save_folder
        # use base folder name as pre-fix for all files
        base_name = os.path.basename(save_folder)  
        
        # create paths to the folder for each object
        try:
            vector_store_path = os.path.join(save_folder, f"{base_name}_vector_store")
            embeddings_path = os.path.join(save_folder, f"{base_name}_embedding_model.joblib")
            documents_path = os.path.join(save_folder, f"{base_name}_documents.joblib")
            file_names_path = os.path.join(save_folder, f"{base_name}_file_names.joblib")
            graph_names_path = os.path.join(save_folder, f"{base_name}_knowledge_graph.joblib")
            
            self.embedding_model_name = joblib.load(embeddings_path)
#             self.embeddings_tool = HuggingFaceEmbeddings(model_name=self.embedding
            self.embeddings_tool = self.load_embeddings_model(embedding_model_name=self.embedding_model_name)            
            self.vector_store = FAISS.load_local(vector_store_path, self.embeddings_tool, allow_dangerous_deserialization=True)
            
            self.documents = joblib.load(documents_path)
            self.files = joblib.load(file_names_path)
            self.G = joblib.load(file_names_path)
        except Exception as ex:
            print("There was an error loading the knowledge_nexus!, see below")
            print(f"EXCEPTION: {ex}")
        return


    def update_vector_embeddings(self, ):
        """This method updates the vector_store and the embeddings. This assumes 
           you have made some change to the documents. So you should call this when you add a new document
           
           Arguments:
                - None
            Returns:
                - None
        """
        if self.documents:
            self.vector_store = FAISS.from_documents(self.documents, self.embeddings_tool)
            self.base_embeddings = torch.stack([torch.tensor(self.embeddings_tool.embed_query(page.page_content)) for page in self.documents])
        else:
            print("Warning! No documents loaded! You must pass either or data_file_TXT or data_file_PDF"
                  " as lists or strings of the indicated document paths to load_pdf_file(), or load_document_list(), but at least one.") 

    def merge_vector_store(self, new_vector_store):
        """
        Merges a new vector store into the current vector store, updates the document set,
        and recalculates vector embeddings.
    
        Args:
            new_vector_store: The vector store object to be merged with the existing vector store.
    
        Raises:
            ValueError: If the new vector store is None or incompatible with the current vector store.
            Exception: If an error occurs during merging or updating embeddings.
        """
        try:
            # Validate the new vector store
            if not new_vector_store:
                raise ValueError("The new vector store cannot be None.")
            if not hasattr(new_vector_store, 'merge_from') or not hasattr(new_vector_store, 'texts'):
                raise ValueError("The new vector store is missing required methods ('merge_from', 'texts').")
    
            # Merge the new vector store into the existing one
            self.vector_store.merge_from(new_vector_store)
    
            # Extract new documents from the incoming vector store and update the document set
            new_docs = [doc for doc in new_vector_store.texts]
            self.documents.extend(new_docs)
    
            # Update vector embeddings for the combined document set
            self.update_vector_embeddings()
    
            print(f"Successfully merged vector store. Added {len(new_docs)} new documents.")
    
        except ValueError as ve:
            print(f"ValueError: {ve}")
        except Exception as e:
            print(f"An error occurred while merging vector stores: {e}")
    
    def clear_docs(self, ):
        """This just empties the current document list and document names list"""
        self.files = []
        self.documents = []
    

    #######################################################################################################
    ####                                Retrieval Tools                                                ####
    #######################################################################################################
    def query_similarity_search(self, 
                                query: str, 
                                k: int=3, 
                                min_score: float=.70, 
                                reverse: bool=False, 
                                mode="min", 
                                verbose=True,
                               ):
        """This will query the current instance of the vector_store object for the top k documents
           That are similiar to it with the given similiary threshold as the lower bound of similarity.
           This is mainly for testing the stores for performance, relevance.

            Arguments:
                - query (string): a question or statement to search for similar documents with
                - k (int): optional, represents the upper bound on the number of documents returned
                - min_score (float): optional, the minimum similarity score for a document and the 
                                     query for it to be returned
            Returns:
                - list_documents[documents] , list_similarity_scores[float]
        """
            
        
        # Query the vector store to find documents with similarity scores
        results = self.vector_store.similarity_search_with_score(query, k=k)
        if verbose:
            print(f"Query in side knexus_gen:\n{query}\n")
            print('returned results:\n', results)
            result_string = '\n>>>>>>\n\n'.join([d.page_content for d,s in results])
            print(f"\nResults:\n{result_string}")
        
        
        # simplify result object into a list of tuples
        results = [(doc, score) for doc, score in results]

        def min_tool(score, min_score):
            if verbose:
                print(score <= min_score)
                print(score, min_score)
                print("-----------------\n\n")
            return score <= min_score
        compare_tool = min_tool
            
        # sort results by similarity score in descending order
        sorted_results = sorted(results, key=lambda x: x[1], reverse=reverse)
        
        # Filter based on the min_score threshold
        filtered_results = [
            (doc, score) for doc, score in sorted_results if compare_tool(score, min_score)
        ]


        # Separate documents and scores for the return value
        list_documents = [doc for doc, score in filtered_results]
        list_similarity_scores = [score for doc, score in filtered_results]
        return list_documents, list_similarity_scores


    #######################################################################################################
    ####                                Graph Construction                                             ####
    #######################################################################################################
    @staticmethod
    def add_doc_nodes(G, documents, type_string="chunk"):
        for idx, doc in enumerate(documents):
            node_id = f"page_{idx+1}"
            G.add_node(
                node_id,
                page=node_id,
                page_content=doc.page_content,
                metadata=doc.metadata,
                type=type_string,
            )
        return G

    @staticmethod
    def add_summary_nodes_and_edges(G, summary_dict):
        for summary, content in summary_dict.items():
            # Find nodes with matching page_content for the content value
            for node, data in G.nodes(data=True):
                if data.get("page_content") == content:
                    # Create a unique ID for the summary node
                    summary_node_id = f"{data['page']}_summary"
                    
                    # Add the summary node
                    G.add_node(
                        summary_node_id,
                        page=summary_node_id,
                        page_content=summary,
                        metadata=data.get("metadata"),  # Copy metadata from the document node
                        type="summary",
                    )
                    
                    # Add an edge between the summary node and the content node
                    G.add_edge(summary_node_id, node, name="SUMMARIZES")
        return G
    
    def generate_summary_keys_words(self, assistant, slist, directive_command="", verbose=True):
        """
        Generates summaries for a list of texts using a conversational pipeline LLM.
        Summaries are concise (up to 5 words) and stored in a dictionary mapping
        each summary to its corresponding input text.
    
        Args:
            slist (list[str]): A list of strings to summarize.
            directive_command (str, optional): Custom directive for the LLM. If not provided,
                                               a default summarization directive will be used.
    
        Returns:
            dict: A dictionary where keys are the generated summaries and values are the corresponding input texts.
        """
        # Default directive for summarization
        base_directive_summarize = (
            "You are to summarize some set of text. The user will give you a chunk of text, "
            "you will summarize the concept of the text in at most 5 words. Do not repeat a summary."
        )
        # Use the base directive if none is provided
        directive_command = directive_command or base_directive_summarize
    
        # Ensure the assistant bot is available
        bot = assistant
    
        # Dictionary to store summaries and their corresponding input texts
        summary_dict = {}
    
        # Iterate over the list of texts to summarize
        for entry in slist:
            try:
                # Create a conversation with the summarization directive
                conversation = [assistant.system(directive_command)]
    
                # Send the directive to the assistant (e.g., LLM)
                _ = assistant.generate_response(conversation)
    
                # Add the user's input text to the conversation
                prompt = assistant.user(entry)
                conversation.append(prompt)
    
                # Generate the summary
                summary = self.generate_response(conversation)
    
                # Store the summary and its corresponding text in the dictionary
                summary_dict[summary.strip()] = entry
                if verbose:
                    print(f"Generated summary: '{summary.strip()}' for entry: '{entry}'")
    
            except Exception as e:
                if verbose:
                    print(f"Error summarizing entry: '{entry}'. Exception: {e}")
        
        # Update the instance's summary dictionary and return it
        self.summary_dict = summary_dict
        if verbose:
            print(f"Successfully generated {len(summary_dict)} summaries.")
        return summary_dict

    @staticmethod
    def generate_ordered_node_edges(G, nodes, step=2):
        """
        Adds edges to a graph (G) in an ordered manner, iterating through the given nodes.
        Edges are added between pairs of nodes with a specified step size.
    
        Args:
            G (networkx.Graph): The graph to which edges will be added.
            nodes (list): A list of nodes to connect with edges.
            step (int, optional): The step size for pairing nodes. Default is 2.
    
        Returns:
            networkx.Graph: The updated graph with added edges.
    
        Notes:
            - The method assumes the nodes exist in the graph before adding edges.
            - An edge weight of 1 is assigned to all added edges.
        """
        try:
            # Validate inputs
            if not isinstance(G, nx.Graph):
                raise TypeError("G must be a networkx.Graph object.")
            if not isinstance(nodes, list):
                raise TypeError("nodes must be a list of graph nodes.")
            if not all(node in G for node in nodes):
                raise ValueError("All nodes in the input list must exist in the graph.")
    
            # Iterate through the nodes with the specified step size
            for i in range(0, len(nodes) - 1, step):
                # Check if an edge already exists before adding
                if not G.has_edge(nodes[i], nodes[i + 1]):
                    G.add_edge(nodes[i], nodes[i + 1], weight=1)
    
            print(f"After edges added, node count: {len(G.nodes)}, edge count: {len(G.edges)}")
            return G

        except Exception as e:
            print(f"An error occurred while generating edges: {e}")
            return G

    def create_sequential_chunk_page_relations(G, node_names):
        """
            This will build edges between pages from a given document
        """
        
        d_count = len(node_names)
        try:
            for idx_a in range(d_count-1):
                if not G.has_edge(node_names[idx_a], node_names[idx_a+1]):
                    G.add_edge(node_names[idx_a], node_names[idx_a+1], weight=1, name="FOLLOWED_BY")
            return G
        except Exception as ex:
            print(f"EX: {ex}\n")
            return G            

    def add_similarity_edges(self, G, cmprfunc, N=3, threshold=.7):
        # for each node find the top N most similar other nodes and 
        # as long as they are above the threshold based on cmprfunc, add an edge
        for node, data in G.nodes(data=True):
            doc_str = data['page_content']
            pid1 = data['page']
            # query the store using the node content
            sim_docs = self.vector_store.similarity_search_with_score(doc_str, k=N)
            for doc, score in sim_docs:
                # Get the content of the similar document
                doc_str2 = doc.page_content
                pid2 = doc.page
        
                # Add an edge if similarity meets the threshold and no edge exists
                if doc_str != doc_str2 and cmprfunc(score, threshold) and not G.has_edge(pid1, pid2):
                    G.add_edge(pid1, pid2, weight=score,
                               name="SIMILAR_TO")
        return G

    def build_similarity_graph_by_vector_store(self, documents, N=3, threshold=0.7, mode="max", verbose=True):
        """
        Builds a similarity graph based on a vector store, connecting documents with edges
        if their similarity exceeds the specified threshold.
    
        Args:
            documents (list): A list of document strings or objects to include in the graph.
            N (int, optional): The number of top similar documents to retrieve for each document. Default is 3.
            threshold (float, optional): The similarity threshold for adding edges. Default is 0.7.
            mode (str, optional): Comparison mode for similarity ("max" or "min"). Default is "max".
            verbose (bool, optional): Whether to print detailed progress and final graph statistics. Default is True.
    
        Returns:
            networkx.Graph: The constructed similarity graph.
        """
        try:
            # Initialize an empty graph
            G = nx.Graph()
    
            # # Add nodes to the graph from current documents
            G = self.add_doc_nodes(G, documents, type_string="chunk")
    
            # Generate ordered node edges
            # G = self.generate_ordered_node_edges(G, documents, step=2)

            # add edges between sequential pages
            G = self.create_sequential_chunk_page_relations(G, node_names)
            
            # Define the comparison function based on the mode
            if mode == 'max':
                cmprfunc = lambda a, b: a > b and a != 1  # Avoid self-loops with similarity == 1
            else:
                cmprfunc = lambda a, b: a < b

            # 
            G = self.add_similarity_edges(G,cmprfunc, N, threshold)
    
            # Print final statistics if verbose is enabled
            if verbose:
                print(f"Final node count: {len(G.nodes)}")
                print(f"Final edge count: {len(G.edges)}")
    
            # Save the graph to the instance
            self.G = G
            return G
    
        except Exception as e:
            print(f"An error occurred while building the similarity graph: {e}")
            return nx.Graph()  # Return an empty graph in case of failure

    def summarize_and_add_edges(self, G, documents, assistant, directive_command="", verbose=True):
        slist = [d.page_content for d in documents]
        summary_dict = self.generate_summary_keys_words(assistant, slist, 
                                         directive_command=directive_command, 
                                         verbose=verbose)
        G = self.add_summary_nodes_and_edges(G, summary_dict)
        return G


class KnowledgeNexusManagerNX(KnowledgeNexusGeneratorNX):
    def __init__(self, 
                 nexus_path:str=None,
                 **kwargs):
        super().__init__(**kwargs)
       
        self.load_nexus_manager(nexus_path)
        self.nexus_path=nexus_path

    def load_nexus_manager(self, nexus_path):
        if nexus_path:
            self.load_knowledge_nexus(nexus_path)
            return
        else:
            return  
    


class KnowledgeNexusManager(knowledge_nexus_generator):
    def __init__(self,
                 neo4j_username: str= os.getenv("NEO4J_UNAME"),
                 neo4j_password: str= os.getenv("NEO4J_PWD"),
                 neo4j_url: str= os.getenv("NEO4J_URL"), 
                 neo4j_db: str="geralddb",
                 nexus_path: str=None, # optional inital knowledge domain to load into manager
                 **kwargs, 
            ):

        super().__init__(
                    neo4j_username=neo4j_username,
                    neo4j_password=neo4j_password,
                    neo4j_url=neo4j_url, 
                    neo4j_db=neo4j_db,
                    **kwargs,
            )
        
        
            
        self.load_nexus_manager(nexus_path)


    def load_nexus_manager(self, nexus_path):
        if nexus_path:
            
            self.load_knowledge_nexus(nexus_path)
            return
        else:
            return


        

class ScopeOfWorkHelper:
    def __init__(self,**kwargs):
        pass
    @staticmethod
    def pull_sow_prompt(doc_content):
        return doc_content.split("\n\n**Scope")[0]
    
    @staticmethod
    def make_scope_of_work(row, description_col, sow_col, work_cntr_col, type_col):
        desc_text = row[description_col].replace("<(>&<)>", "&")
        wrk_cntr = row[work_cntr_col]
        m_type = row[type_col]
    
        scope = row[sow_col]
        ret_str = "**Task Description:**\n{desc}\n\n"\
                  "**Scope of Work**\nwork center: {wrkcntr}\nmaintenance type:{mtype}\nDescription of Task:{sow}\n\n"
        return ret_str.format(desc=desc_text, wrkcntr=wrk_cntr, mtype=m_type, sow=scope)
    
    
    def make_scope_statments(self, df, data_path, desc_col="Description", sow_col="Long-Text", 
                             wrk_cntr_col="Main Work Center", mtype_col="Order Type",
                             verbose=False,
                            ):
        docset = []
        for idx, row in df.iterrows():
            ret_str = self.make_scope_of_work(row, desc_col, sow_col, wrk_cntr_col, mtype_col)
            if verbose:
                print(f"IDX:{idx}")
                print(ret_str)
                print("---------------------\n\n")
            docset.append(Document(
                page_content=ret_str,
                metadata={
                    "row_idx": idx,
                    "source": data_path
                }
            ))
        return docset
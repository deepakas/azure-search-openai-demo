import argparse
import asyncio
import glob
import logging
import os
from typing import Optional

from azure.core.credentials import AzureKeyCredential
from azure.core.credentials_async import AsyncTokenCredential
from azure.identity.aio import DefaultAzureCredential
from azure.search.documents.aio import SearchClient
from openai import AsyncOpenAI
from dotenv import load_dotenv
from rich.logging import RichHandler

from prepdocslib.fileprocessor import FileProcessor
from prepdocslib.servicesetup import (
    OpenAIHost,
    build_file_processors,
    clean_key_if_exists,
    setup_embeddings_service,
    setup_openai_client,
)
from azure.search.documents.indexes import SearchIndexClient
from azure.search.documents.indexes.models import (
    SearchIndex,
    SearchField,
    SearchFieldDataType,
    SimpleField,
    SearchableField,
    VectorSearch,
    HnswAlgorithmConfiguration,
    VectorSearchProfile,
)

logger = logging.getLogger("prepdocs_simple")


def sanitize_document_id(filename: str) -> str:
    """Sanitize filename to create valid Azure Search document ID.
    
    Document keys can only contain letters, digits, underscore (_), dash (-), or equal sign (=).
    """
    import re
    
    # Remove file extension
    name_without_ext = os.path.splitext(filename)[0]
    
    # Replace spaces and special characters with underscores
    # Keep only alphanumeric, underscores, and dashes
    sanitized = re.sub(r'[^a-zA-Z0-9_-]', '_', name_without_ext)
    
    # Remove consecutive underscores
    sanitized = re.sub(r'_+', '_', sanitized)
    
    # Remove leading/trailing underscores
    sanitized = sanitized.strip('_')
    
    return sanitized


def extract_metadata_from_filepath(file_path: str) -> dict:
    """Extract client name, assessment year, engagement name, and extracted_year from file path.
    
    Expected path structure: data/{client}/year={year}/engagement={engagement}
    Example: data/msquared/year=2024/engagement=msquared nyc affordable v1
    Also attempts to extract year from filename using patterns like YYYY or year=YYYY
    """
    import re
    metadata = {}
    
    try:
        # Normalize path to use forward slashes
        normalized_path = file_path.replace("\\", "/")
        path_parts = normalized_path.split("/")
        
        # Extract client name (typically first folder after 'data')
        if "data" in path_parts:
            data_idx = path_parts.index("data")
            if data_idx + 1 < len(path_parts):
                client = path_parts[data_idx + 1]
                if client and client != "data":
                    metadata["client_name"] = client
        
        # Extract assessment year from year=XXXX pattern in path
        for part in path_parts:
            if part.startswith("year="):
                year = part.split("=", 1)[1]
                if year:
                    metadata["assessment_year"] = year
                    break
        
        # Extract engagement name from engagement=XXX pattern
        for part in path_parts:
            if part.startswith("engagement="):
                engagement = part.split("=", 1)[1]
                if engagement:
                    metadata["engagement_name"] = engagement
                    break
        
        # Extract year from filename if present
        filename = os.path.basename(file_path)
        # Look for 4-digit year patterns (YYYY) in filename
        year_matches = re.findall(r'\b(20\d{2}|19\d{2})\b', filename)
        if year_matches:
            metadata["extracted_year"] = year_matches[0]
    except Exception as e:
        logger.debug(f"Could not extract metadata from filepath {file_path}: {e}")
    
    return metadata


async def extract_metadata_from_content(
    content: str,
    filename: str,
    openai_client: AsyncOpenAI,
    model_name: str = "gpt-4o",
) -> dict:
    """Extract metadata from document content using GPT.
    
    Note: client_name, engagement_name, and assessment_year are extracted from folder structure,
    so we only extract document classification and audit area tags from content.
    """
    try:
        # Create a prompt to extract metadata from the content
        prompt = f"""Analyze the following document content and extract the following metadata in JSON format:
- document_category: The category of document (must be one of: "Financial Statements", "Risk Assessment", "Trial Balance and General Ledger", "General Planning", "Fieldwork", "Wrap-Up", "Other")
- extracted_year: The year mentioned in the document content (4-digit year like 2024, or null if not found)

Document filename: {filename}

Document content (first 2000 characters):
{content[:2000]}

Return only valid JSON with these fields. If a field cannot be determined, use null."""

        response = await openai_client.chat.completions.create(
            model=model_name,
            messages=[
                {
                    "role": "system",
                    "content": "You are an expert auditor AI that extracts structured metadata from financial and audit documents. Return valid JSON only.",
                }
                ,
                {"role": "user", "content": prompt},
            ],
            temperature=0.3,
            max_completion_tokens=500,
        )

        # Parse the response
        import json

        response_text = response.choices[0].message.content
        # Find JSON in response
        json_start = response_text.find("{")
        json_end = response_text.rfind("}") + 1
        if json_start != -1 and json_end > json_start:
            metadata = json.loads(response_text[json_start:json_end])
            return metadata
        else:
            logger.warning(f"Could not extract JSON from GPT response for {filename}")
            return {}
    except Exception as e:
        logger.warning(f"Error extracting metadata for {filename}: {e}")
        return {}


async def create_search_index(
    search_client,
    search_endpoint: str,
    search_api_key: Optional[str],
    azure_credential: AsyncTokenCredential,
    index_name: str,
    embeddings_service: Optional[object],
) -> None:
    """Create search index if it doesn't exist."""
    try:
        logger.info(f"Creating search index '{index_name}'...")
        
        # Create search index client
        if search_api_key:
            credential = AzureKeyCredential(search_api_key)
        else:
            credential = azure_credential
        
        index_client = SearchIndexClient(endpoint=search_endpoint, credential=credential)
        
        # Get embedding dimension from embeddings service
        embedding_dimensions = embeddings_service.open_ai_dimensions if embeddings_service else 1536
        
        # Define fields for the index
        fields = [
            SimpleField(name="id", type=SearchFieldDataType.String, key=True, sortable=True),
            SearchableField(name="content", type=SearchFieldDataType.String),
            SimpleField(name="sourcepage", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="sourcefile", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            # Metadata fields extracted from document content
            SimpleField(name="document_category", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="extracted_year", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="client_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="engagement_name", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="assessment_year", type=SearchFieldDataType.String, filterable=True, facetable=True),
            SimpleField(name="file_path", type=SearchFieldDataType.String, filterable=True),
        ]
        
        # Add embedding field if embeddings are available
        if embeddings_service:
            fields.append(
                SearchField(
                    name="embedding",
                    type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
                    searchable=True,
                    vector_search_dimensions=embedding_dimensions,
                    vector_search_profile_name="myHnswProfile",
                )
            )
        
        # Create vector search configuration if embeddings are available
        vector_search = None
        if embeddings_service:
            vector_search = VectorSearch(
                algorithms=[
                    HnswAlgorithmConfiguration(name="myHnsw"),
                ],
                profiles=[
                    VectorSearchProfile(
                        name="myHnswProfile",
                        algorithm_configuration_name="myHnsw",
                    ),
                ],
            )
        
        # Create the index
        index = SearchIndex(
            name=index_name,
            fields=fields,
            vector_search=vector_search,
        )
        
        result = index_client.create_index(index)
        logger.info(f"Successfully created index '{index_name}'")
    except Exception as e:
        logger.error(f"Error creating search index: {e}", exc_info=True)
        raise


async def process_and_index_files(
    file_processors: dict,
    search_client: SearchClient,
    embeddings_service: Optional[object],
    all_files: list[str],
    category: Optional[str] = None,
    search_endpoint: Optional[str] = None,
    search_api_key: Optional[str] = None,
    index_name: Optional[str] = None,
    azure_credential: Optional[AsyncTokenCredential] = None,
    openai_client: Optional[AsyncOpenAI] = None,
) -> None:
    """Process files and index them directly into Azure AI Search.
    
    Processes each file end-to-end: extract -> chunk -> embed -> index
    before moving to the next file.
    """
    
    logger.info(f"Found {len(all_files)} files to process")
    
    if not all_files:
        logger.warning("No files found to process")
        return
    
    # Create index once before processing files
    try:
        logger.info(f"Creating search index '{index_name}' if it doesn't exist...")
        await create_search_index(search_client, search_endpoint, search_api_key, azure_credential, index_name, embeddings_service)
    except Exception as e:
        logger.warning(f"Index creation: {e}")
    
    # Cache for file metadata
    file_metadata_cache: dict[str, dict] = {}
    
    # Process files one at a time
    for file_idx, file_path in enumerate(all_files, 1):
        try:
            logger.info(f"[{file_idx}/{len(all_files)}] Processing file: {file_path}")
            
            # Get appropriate processor for file type based on extension
            file_ext = os.path.splitext(file_path)[1].lower()
            processor = file_processors.get(file_ext)
            
            if not processor:
                logger.warning(f"No processor found for file: {file_path} (ext: {file_ext})")
                continue
            
            # Read the file and parse it to get pages
            try:
                with open(file_path, "rb") as file:
                    pages = [page async for page in processor.parser.parse(content=file)]
            except Exception as e:
                logger.error(f"Failed to parse file {file_path}: {e}")
                continue
            
            # Extract metadata from first page if not cached
            filename = os.path.basename(file_path)
            if filename not in file_metadata_cache and pages:
                logger.info(f"Extracting metadata from {filename}...")
                # First, extract from filepath structure
                filepath_metadata = extract_metadata_from_filepath(file_path)
                # Then, extract from content using GPT
                model_deployment = os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-4o")
                content_metadata = await extract_metadata_from_content(pages[0].text, filename, openai_client, model_deployment)
                # Merge with filepath metadata taking precedence
                metadata = {**content_metadata, **filepath_metadata}
                file_metadata_cache[filename] = metadata
            else:
                metadata = file_metadata_cache.get(filename, {})
            
            # Convert pages to search documents using the splitter
            chunks = list(processor.splitter.split_pages(pages))
            documents_to_index: list[dict] = []
            
            for chunk_num, chunk in enumerate(chunks):
                # Create document ID without invalid characters
                sanitized_filename = sanitize_document_id(filename)
                doc = {
                    "id": f"{sanitized_filename}-{chunk.page_num}-{chunk_num}",
                    "content": chunk.text,
                    "sourcepage": f"{filename}#{chunk.page_num}",
                    "sourcefile": filename,
                }
                
                # Add category if provided
                if category:
                    doc["category"] = category
                
                # Add extracted metadata
                if metadata:
                    doc["document_category"] = metadata.get("document_category")
                    # Ensure extracted_year is a string
                    extracted_year = metadata.get("extracted_year")
                    if extracted_year is not None:
                        doc["extracted_year"] = str(extracted_year)
                    doc["client_name"] = metadata.get("client_name")
                    doc["engagement_name"] = metadata.get("engagement_name")
                    doc["assessment_year"] = metadata.get("assessment_year")
                
                # Add file path metadata (relative to data root)
                # Extract the relative path from the glob pattern matching
                try:
                    # Normalize paths for comparison
                    normalized_file = os.path.normpath(file_path)
                    # Try to find the common data path
                    if "data" in normalized_file:
                        # Get path relative to the "data" directory
                        data_idx = normalized_file.lower().rfind(os.sep + "data" + os.sep)
                        if data_idx != -1:
                            rel_path = normalized_file[data_idx + len(os.sep):]
                        else:
                            # Check if starts with data
                            if normalized_file.lower().startswith("data"):
                                rel_path = normalized_file
                            else:
                                rel_path = normalized_file
                    else:
                        rel_path = normalized_file
                    # Normalize to forward slashes for consistency
                    doc["file_path"] = rel_path.replace(os.sep, "/")
                except Exception as e:
                    logger.warning(f"Failed to extract relative file path: {e}")
                    doc["file_path"] = file_path
                
                # Add embedding if embeddings service is available
                if embeddings_service:
                    try:
                        embeddings = await embeddings_service.create_embeddings([chunk.text])
                        doc["embedding"] = embeddings[0]
                    except Exception as e:
                        logger.warning(f"Failed to generate embedding for {doc['id']}: {e}")
                
                documents_to_index.append(doc)
            
            # Upload documents for this file immediately
            if documents_to_index:
                logger.info(f"Indexing {len(documents_to_index)} documents from {filename}")
                try:
                    results = await search_client.upload_documents(documents_to_index)
                    successful = sum(1 for r in results if r.succeeded)
                    logger.info(f"Successfully indexed {successful}/{len(results)} documents from {filename}")
                except Exception as e:
                    logger.error(f"Error uploading documents for {filename}: {e}")
            else:
                logger.warning(f"No documents generated from {filename}")
        
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}", exc_info=True)
            continue
    
    logger.info(f"Completed processing all {len(all_files)} files")


async def main(
    data_path: str,
    category: Optional[str],
    document_intelligence_service: Optional[str],
    document_intelligence_key: Optional[str],
    use_multimodal: bool,
    azure_credential: AsyncTokenCredential,
    openai_client: AsyncOpenAI,
    openai_embeddings_service: Optional[object],
) -> None:
    """Main function to orchestrate file processing and indexing."""
    
    # Discover files using glob
    all_files = []
    try:
        # Ensure data_path is a glob pattern that supports recursive search
        # If user provides "./data/*", convert to "./data/**" for recursive matching
        if data_path.endswith("/*"):
            data_path = data_path[:-1] + "**"
        
        logger.debug(f"Using glob pattern: {data_path}")
        all_files = glob.glob(data_path, recursive=True)
        all_files = [f for f in all_files if os.path.isfile(f)]
    except Exception as e:
        logger.error(f"Error listing files: {e}")
        return
    
    # Setup file processors - returns a dict of parser_name -> parser objects
    file_processors = build_file_processors(
        azure_credential=azure_credential,
        document_intelligence_service=document_intelligence_service,
        document_intelligence_key=document_intelligence_key,
        use_local_pdf_parser=os.getenv("USE_LOCAL_PDF_PARSER") == "true",
        use_local_html_parser=os.getenv("USE_LOCAL_HTML_PARSER") == "true",
        process_figures=use_multimodal,
    )
    
    logger.info(f"Available processors: {list(file_processors.keys())}")
    for proc_name, proc in file_processors.items():
        logger.debug(f"  - {proc_name}: {type(proc).__name__}")
    
    # Setup search client
    search_endpoint = os.environ.get("AZURE_SEARCH_ENDPOINT")
    index_name = os.environ.get("AZURE_SEARCH_INDEX")
    search_api_key = os.environ.get("AZURE_SEARCH_API_KEY")
    
    if not search_endpoint or not index_name:
        raise ValueError("AZURE_SEARCH_ENDPOINT and AZURE_SEARCH_INDEX must be set in .env")
    
    # Use API key if provided, otherwise use credential
    if search_api_key:
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=AzureKeyCredential(search_api_key),
        )
    else:
        search_client = SearchClient(
            endpoint=search_endpoint,
            index_name=index_name,
            credential=azure_credential,
        )
    
    # Process and index files
    await process_and_index_files(
        file_processors=file_processors,
        search_client=search_client,
        embeddings_service=openai_embeddings_service,
        all_files=all_files,
        category=category,
        search_endpoint=search_endpoint,
        search_api_key=search_api_key,
        index_name=index_name,
        azure_credential=azure_credential,
        openai_client=openai_client,
    )
    
    await search_client.close()


if __name__ == "__main__":  # pragma: no cover
    parser = argparse.ArgumentParser(
        description="Simple document preparation: read files, process with Document Intelligence, and index into Azure AI Search"
    )
    parser.add_argument(
        "--datapath",
        default="./data/*",
        help="Path pattern for files to process (default: ./data/*)"
    )
    parser.add_argument(
        "--category",
        help="Category value for all indexed documents"
    )
    parser.add_argument(
        "--searchkey",
        required=False,
        help="Optional. Azure AI Search account key"
    )
    parser.add_argument(
        "--documentintelligencekey",
        required=False,
        help="Optional. Azure Document Intelligence account key"
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    # Load environment variables from .env file
    load_dotenv()
    
    if args.verbose:
        logging.basicConfig(
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        logger.setLevel(logging.DEBUG)
    else:
        logging.basicConfig(
            format="%(message)s",
            datefmt="[%X]",
            handlers=[RichHandler(rich_tracebacks=True)]
        )
        logger.setLevel(logging.INFO)
    
    # Validate required environment variables
    required_vars = [
        "AZURE_SEARCH_ENDPOINT",
        "AZURE_SEARCH_INDEX",
        "AZURE_OPENAI_ENDPOINT",
        "AZURE_OPENAI_CHATGPT_DEPLOYMENT",
        "AZURE_OPENAI_CHATGPT_MODEL",
        "AZURE_OPENAI_EMB_MODEL_NAME",
        "AZURE_OPENAI_EMB_DEPLOYMENT",
    ]
    
    missing_vars = [var for var in required_vars if not os.getenv(var)]
    if missing_vars:
        logger.error(f"Missing required environment variables: {', '.join(missing_vars)}")
        logger.error("Please configure these in your .env file")
        exit(1)
    
    azure_credential: AsyncTokenCredential = DefaultAzureCredential()
    
    OPENAI_HOST = OpenAIHost(os.environ.get("OPENAI_HOST", "azure"))
    
    # Setup OpenAI client
    openai_client, azure_openai_endpoint = setup_openai_client(
        openai_host=OPENAI_HOST,
        azure_credential=azure_credential,
        azure_openai_service=os.getenv("AZURE_OPENAI_SERVICE"),
        azure_openai_custom_url=os.getenv("AZURE_OPENAI_CUSTOM_URL"),
        azure_openai_api_key=os.getenv("AZURE_OPENAI_KEY"),
        openai_api_key=clean_key_if_exists(os.getenv("OPENAI_API_KEY")),
        openai_organization=os.getenv("OPENAI_ORGANIZATION"),
    )
    
    emb_model_dimensions = int(os.getenv("AZURE_OPENAI_EMB_DIMENSIONS", "1536"))
    openai_embeddings_service = setup_embeddings_service(
        OPENAI_HOST,
        openai_client,
        emb_model_name=os.environ.get("AZURE_OPENAI_EMB_MODEL_NAME", "text-embedding-3-small"),
        emb_model_dimensions=emb_model_dimensions,
        azure_openai_deployment=os.getenv("AZURE_OPENAI_EMB_DEPLOYMENT"),
        azure_openai_endpoint=azure_openai_endpoint,
        disable_batch=False,
    )
    
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        # Get document intelligence key from args or environment variable
        doc_intel_key = clean_key_if_exists(args.documentintelligencekey) or os.getenv("AZURE_DOCUMENTINTELLIGENCE_KEY")
        
        loop.run_until_complete(
            main(
                data_path=args.datapath,
                category=args.category,
                document_intelligence_service=os.getenv("AZURE_DOCUMENTINTELLIGENCE_SERVICE"),
                document_intelligence_key=doc_intel_key,
                use_multimodal=os.getenv("USE_MULTIMODAL", "").lower() == "true",
                azure_credential=azure_credential,
                openai_client=openai_client,
                openai_embeddings_service=openai_embeddings_service,
            )
        )
    finally:
        try:
            loop.run_until_complete(openai_client.close())
            loop.run_until_complete(azure_credential.close())
        except Exception as e:
            logger.debug(f"Error closing clients: {e}")
        loop.close()

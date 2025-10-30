"""
Production-grade FastAPI application with S3, DynamoDB, and Bedrock integration
Optimized for uvicorn deployment on RedHat Linux EC2
"""
import os
import uuid
import logging
from typing import Optional, Dict, Any
from datetime import datetime, timezone

import boto3
from fastapi import FastAPI, File, UploadFile, HTTPException, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from botocore.exceptions import ClientError, NoCredentialsError

from langchain_aws import ChatBedrock
from langchain_core.messages import HumanMessage

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# ==================== Configuration ====================
AWS_REGION = os.getenv("AWS_REGION", "us-east-1")
S3_BUCKET = os.getenv("S3_BUCKET", "r2dr-docs")
DYNAMODB_TABLE = os.getenv("DYNAMODB_TABLE", "r2d2-session-log")
BEDROCK_MODEL_ID = os.getenv("BEDROCK_MODEL_ID", "anthropic.claude-3-5-sonnet-20240620-v1:0")

# ==================== Pydantic Models ====================
class ChatRequest(BaseModel):
    """Request model for AI chat"""
    message: str = Field(..., min_length=1, max_length=5000, description="User message")
    session_id: Optional[str] = Field(None, description="Session ID for conversation tracking")

class ChatResponse(BaseModel):
    """Response model for AI chat"""
    response: str
    session_id: str
    timestamp: str

class UploadResponse(BaseModel):
    """Response model for PDF upload"""
    job_id: str
    filename: str
    s3_key: str
    status: str
    timestamp: str

class DynamoDataResponse(BaseModel):
    """Response model for DynamoDB data retrieval"""
    job_id: str
    data: Dict[str, Any]
    timestamp: str

class HealthCheckResponse(BaseModel):
    """Response model for health check"""
    status: str
    timestamp: str
    version: str

# ==================== AWS Clients ====================
def get_s3_client():
    """Initialize S3 client"""
    try:
        return boto3.client('s3', region_name=AWS_REGION)
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AWS credentials not configured"
        )

def get_dynamodb_client():
    """Initialize DynamoDB client"""
    try:
        return boto3.client('dynamodb', region_name=AWS_REGION)
    except NoCredentialsError:
        logger.error("AWS credentials not found")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AWS credentials not configured"
        )

def get_bedrock_chat():
    """Initialize Bedrock chat client"""
    try:
        return ChatBedrock(
            model_id=BEDROCK_MODEL_ID,
            region_name=AWS_REGION,
            model_kwargs={
                "max_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
            }
        )
    except Exception as e:
        logger.error(f"Failed to initialize Bedrock client: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Bedrock service unavailable"
        )

# ==================== FastAPI App ====================
app = FastAPI(
    title="Production FastAPI Service",
    description="FastAPI with S3, DynamoDB, and Bedrock Claude 3.5 Sonnet",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS Configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure based on your security requirements
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ==================== API Routes ====================

@app.get("/", tags=["Root"])
async def root():
    """Root endpoint with API information"""
    return {
        "service": "Production FastAPI Service",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "upload_pdf": "POST /upload-pdf",
            "get_data": "GET /data/{job_id}",
            "chat": "POST /chat",
            "docs": "/docs"
        }
    }

@app.get("/health", response_model=HealthCheckResponse, tags=["Health"])
async def health_check():
    """Health check endpoint for load balancer and monitoring"""
    return HealthCheckResponse(
        status="healthy",
        timestamp=datetime.now(timezone.utc).isoformat(),
        version="1.0.0"
    )

@app.post("/upload-pdf", response_model=UploadResponse, tags=["S3 Operations"])
async def upload_pdf_to_s3(file: UploadFile = File(...)):
    """
    Upload PDF file to S3 bucket and log metadata to DynamoDB

    - **file**: PDF file to upload
    """
    # Validate file type
    if not file.filename.lower().endswith('.pdf'):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only PDF files are allowed"
        )

    # Generate unique job ID
    job_id = str(uuid.uuid4())
    timestamp = datetime.now(timezone.utc)
    timestamp_str = timestamp.strftime("%Y%m%d_%H%M%S")

    # Create S3 key with organized structure
    s3_key = f"uploads/{timestamp_str}/{job_id}_{file.filename}"

    try:
        # Read file content
        file_content = await file.read()
        file_size = len(file_content)

        # Upload to S3
        s3_client = get_s3_client()
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=file_content,
            ContentType='application/pdf',
            Metadata={
                'job_id': job_id,
                'original_filename': file.filename,
                'upload_timestamp': timestamp.isoformat(),
                'file_size': str(file_size)
            }
        )

        logger.info(f"File uploaded to S3: {s3_key} (job_id: {job_id})")

        # Store metadata in DynamoDB
        dynamodb_client = get_dynamodb_client()
        dynamodb_client.put_item(
            TableName=DYNAMODB_TABLE,
            Item={
                'job_id': {'S': job_id},
                'filename': {'S': file.filename},
                's3_key': {'S': s3_key},
                's3_bucket': {'S': S3_BUCKET},
                'file_size': {'N': str(file_size)},
                'upload_timestamp': {'S': timestamp.isoformat()},
                'status': {'S': 'uploaded'},
                'type': {'S': 'pdf_upload'}
            }
        )

        logger.info(f"Metadata stored in DynamoDB for job_id: {job_id}")

        return UploadResponse(
            job_id=job_id,
            filename=file.filename,
            s3_key=s3_key,
            status="success",
            timestamp=timestamp.isoformat()
        )

    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"AWS service error: {error_code} - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"AWS service error: {error_code}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during upload: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="File upload failed"
        )

@app.get("/data/{job_id}", response_model=DynamoDataResponse, tags=["DynamoDB Operations"])
async def get_data_from_dynamodb(job_id: str):
    """
    Retrieve data from DynamoDB by job_id

    - **job_id**: Unique job identifier (partition key)
    """
    try:
        dynamodb_client = get_dynamodb_client()

        # Get item from DynamoDB
        response = dynamodb_client.get_item(
            TableName=DYNAMODB_TABLE,
            Key={'job_id': {'S': job_id}}
        )

        # Check if item exists
        if 'Item' not in response:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail=f"Job ID '{job_id}' not found in database"
            )

        # Parse DynamoDB item to Python dict
        item = response['Item']
        parsed_data = {}

        for key, value in item.items():
            if 'S' in value:
                parsed_data[key] = value['S']
            elif 'N' in value:
                parsed_data[key] = value['N']
            elif 'BOOL' in value:
                parsed_data[key] = value['BOOL']
            elif 'NULL' in value:
                parsed_data[key] = None
            else:
                parsed_data[key] = str(value)

        logger.info(f"Data retrieved for job_id: {job_id}")

        return DynamoDataResponse(
            job_id=job_id,
            data=parsed_data,
            timestamp=datetime.now(timezone.utc).isoformat()
        )

    except HTTPException:
        raise
    except ClientError as e:
        error_code = e.response.get('Error', {}).get('Code', 'Unknown')
        logger.error(f"DynamoDB error: {error_code} - {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Database error: {error_code}"
        )
    except Exception as e:
        logger.error(f"Unexpected error during data retrieval: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Data retrieval failed"
        )

@app.post("/chat", response_model=ChatResponse, tags=["AI Chat"])
async def chat_with_bedrock(request: ChatRequest):
    """
    Chat with Claude 3.5 Sonnet via AWS Bedrock

    - **message**: User message to send to AI
    - **session_id**: Optional session ID for conversation tracking
    """
    try:
        # Initialize Bedrock chat
        chat = get_bedrock_chat()

        # Generate or use provided session ID
        session_id = request.session_id or str(uuid.uuid4())

        # Create message
        messages = [HumanMessage(content=request.message)]

        # Get AI response
        logger.info(f"Sending message to Bedrock (session: {session_id})")
        response = chat.invoke(messages)
        ai_response = response.content

        # Log chat to DynamoDB
        timestamp = datetime.now(timezone.utc)
        chat_id = str(uuid.uuid4())

        try:
            dynamodb_client = get_dynamodb_client()
            dynamodb_client.put_item(
                TableName=DYNAMODB_TABLE,
                Item={
                    'job_id': {'S': chat_id},
                    'session_id': {'S': session_id},
                    'user_message': {'S': request.message},
                    'ai_response': {'S': ai_response},
                    'model': {'S': BEDROCK_MODEL_ID},
                    'timestamp': {'S': timestamp.isoformat()},
                    'type': {'S': 'chat'}
                }
            )
            logger.info(f"Chat logged to DynamoDB (chat_id: {chat_id})")
        except Exception as log_error:
            logger.warning(f"Failed to log chat to DynamoDB: {str(log_error)}")

        return ChatResponse(
            response=ai_response,
            session_id=session_id,
            timestamp=timestamp.isoformat()
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Chat request failed: {str(e)}")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="AI chat service unavailable"
        )

# ==================== Exception Handlers ====================

@app.exception_handler(HTTPException)
async def http_exception_handler(request, exc):
    """Handle HTTP exceptions"""
    logger.warning(f"HTTP {exc.status_code}: {exc.detail}")
    return JSONResponse(
        status_code=exc.status_code,
        content={
            "error": exc.detail,
            "status_code": exc.status_code,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

@app.exception_handler(Exception)
async def general_exception_handler(request, exc):
    """Handle general exceptions"""
    logger.error(f"Unhandled exception: {str(exc)}", exc_info=True)
    return JSONResponse(
        status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
        content={
            "error": "Internal server error",
            "status_code": 500,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
    )

# ==================== Main Entry Point ====================

if __name__ == "__main__":
    import uvicorn

    # Uvicorn configuration for development
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
        workers=1,
        log_level="info"
    )

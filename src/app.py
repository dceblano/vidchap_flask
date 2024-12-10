import os
import torch
import boto3
import sys
from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertForSequenceClassification
from dotenv import load_dotenv
from botocore.exceptions import ClientError

# Add the parent directory to the system path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.pipelines.model1_prediction_pipeline import Model1Pipeline
from src.pipelines.model2_prediction_pipeline import Model2Pipeline

# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# AWS S3 configuration from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
BUCKET_NAME = os.getenv("S3_BUCKET_NAME")

# Initialize the S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

LOCAL_MODEL_DIR = "model"  # Base directory for models

def download_model_from_s3(bucket_name, model_path, local_dir):
    """
    Downloads the model files from S3 to a local directory, maintaining folder structure.
    """
    # Ensure the model folder exists (models/best_bert_model)
    model_dir = os.path.join(local_dir, model_path)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
        print(f"Created local model directory: {model_dir}")

    # Debug: Print bucket and model path
    print(f"Bucket Name: {bucket_name}")
    print(f"Model Path: {model_path}")

    # Check if model_path is None or empty
    if not model_path:
        print("Error: Model path is None or empty.")
        return
    
    try:
        # List all objects in the S3 folder (model files)
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=model_path)
        
        # Print the response to check if files are present
        print(f"S3 List Response: {response}")

        if 'Contents' not in response:
            print(f"No objects found for prefix: {model_path}")
            return
        
        # Iterate over each object in the S3 directory
        for obj in response['Contents']:
            print(f"Found object: {obj['Key']}")
            # Generate local path for the downloaded file
            # Extract sub-directory path from the model_path to maintain the folder structure
            relative_path = os.path.relpath(obj['Key'], model_path)
            local_file_path = os.path.join(model_dir, relative_path)
            
            # Create subdirectories if they don't exist
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            
            # Download the file from S3 to the local path
            s3.download_file(bucket_name, obj['Key'], local_file_path)
            print(f"Downloaded {obj['Key']} to {local_file_path}")
    
    except ClientError as e:
        # Handle AWS-specific exceptions
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            print(f"Access Denied: {e}")
        elif error_code == 'NoSuchKey':
            print(f"Model not found: {e}")
        else:
            print(f"S3 ClientError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

# Load BERT model and tokenizer (Model 1)
def load_bert_model():
    model1_dir = os.path.join(LOCAL_MODEL_DIR, 'model/best_bert_model')
    model1_tokenizer = BertTokenizer.from_pretrained(model1_dir)
    model1 = BertForSequenceClassification.from_pretrained(model1_dir, num_labels=1)
    model1.eval()
    model1.to(device)
    return model1, model1_tokenizer

# Load BART model and tokenizer (Model 2)
def load_bart_model():
    model2_dir = os.path.join(LOCAL_MODEL_DIR, 'model/best_bart_model')
    model2_tokenizer = BartTokenizer.from_pretrained(model2_dir)
    model2 = BartForConditionalGeneration.from_pretrained(model2_dir)
    model2.eval()
    model2.to(device)
    return model2, model2_tokenizer

# Initialize the pipelines
model1_pipeline = None
model2_pipeline = None

@app.route('/vidchap/chaptering', methods=['POST'])
def chaptering():
    try:
        data = request.json
        if 'video_id' not in data:
            return jsonify({"error": "Missing video_id in request"}), 400
        video_id = data['video_id']

        # Process the video_id through Model 1 pipeline
        records_df = model1_pipeline.get_records_from_model1(video_id)
        if records_df.empty:
            return jsonify({"error": "No records returned from Model 1"}), 400
        print('Model 1 prediction completed!')
        print(records_df.shape)
        print(records_df.head())

        # Process the data through Model 2 pipeline
        response_data = model2_pipeline.process(records_df)

        return jsonify({"results": response_data}), 200

    except ValueError as ve:
        return jsonify({"error": str(ve)}), 400
    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    # Download models from S3
    download_model_from_s3(BUCKET_NAME, 'model/best_bert_model', LOCAL_MODEL_DIR)
    download_model_from_s3(BUCKET_NAME, 'model/best_bart_model', LOCAL_MODEL_DIR)

    # Load models
    model1, model1_tokenizer = load_bert_model()
    model2, model2_tokenizer = load_bart_model()

    # Initialize the pipelines
    model1_pipeline = Model1Pipeline(model1, model1_tokenizer, device)
    model2_pipeline = Model2Pipeline(model2, model2_tokenizer, device)
    
    port = int(os.environ.get("PORT", 10000))
    app.run(host='0.0.0.0', port=port)

import os
import torch
import boto3

from flask import Flask, request, jsonify
from transformers import BartForConditionalGeneration, BartTokenizer, BertTokenizer, BertForSequenceClassification
from pipelines.model1_prediction_pipeline import Model1Pipeline
from pipelines.model2_prediction_pipeline import Model2Pipeline
from dotenv import load_dotenv
from botocore.exceptions import ClientError


# Load environment variables from .env file
load_dotenv()

app = Flask(__name__)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# AWS S3 configuration from environment variables
AWS_ACCESS_KEY_ID = os.getenv("AWS_ACCESS_KEY_ID")
AWS_SECRET_ACCESS_KEY = os.getenv("AWS_SECRET_ACCESS_KEY")
AWS_DEFAULT_REGION = os.getenv("AWS_DEFAULT_REGION")
BUCKET_NAME = os.getenv("BUCKET_NAME")

# Initialize the S3 client
s3 = boto3.client(
    's3',
    aws_access_key_id=AWS_ACCESS_KEY_ID,
    aws_secret_access_key=AWS_SECRET_ACCESS_KEY,
    region_name=AWS_DEFAULT_REGION
)

# Load BERT model and tokenizer (Model 1)
model1_dir = "model/best_bert_model"
model1_tokenizer = BertTokenizer.from_pretrained(model1_dir)
model1 = BertForSequenceClassification.from_pretrained(model1_dir, num_labels=1)
model1.eval()
model1.to(device)

# Load BART model and tokenizer (Model 2)
model2_dir = "model/best_bart_model"
model2_tokenizer = BartTokenizer.from_pretrained(model2_dir)
model2 = BartForConditionalGeneration.from_pretrained(model2_dir)
model2.eval()
model2.to(device)

# Initialize the pipelines
model1_pipeline = Model1Pipeline(model1, model1_tokenizer, device)
model2_pipeline = Model2Pipeline(model2, model2_tokenizer, device)

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

def load_model_from_s3(bucket_name, model_path):
    try:
        # Check if the model exists in S3
        response = s3.list_objects_v2(Bucket=bucket_name, Prefix=model_path)
        if 'Contents' in response:
            for obj in response['Contents']:
                print(f"Object found: {obj['Key']}")
                # Download the model file
                s3.download_file(bucket_name, obj['Key'], f"local_path/{obj['Key']}")
                print(f"Downloaded {obj['Key']} successfully.")
        else:
            print("No objects found for the given prefix.")
    except ClientError as e:
        # Handle specific AWS S3 errors here
        error_code = e.response['Error']['Code']
        if error_code == 'AccessDenied':
            print(f"Access Denied: {e}")
        elif error_code == 'NoSuchKey':
            print(f"Model not found: {e}")
        else:
            print(f"S3 ClientError: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")

if __name__ == '__main__':
    # Optionally load models from S3 (you can call this when needed)
    load_model_from_s3(BUCKET_NAME, 'model/best_bert_model')
    load_model_from_s3(BUCKET_NAME, 'model/best_bart_model')
    
    app.run(debug=True)

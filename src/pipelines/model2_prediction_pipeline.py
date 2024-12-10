# src/pipelines/model2_pred_pipeline.py
import torch
import pandas as pd
from transformers import BartTokenizer

class Model2Pipeline:
    def __init__(self, model, tokenizer, device):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

    def prepare_input_data(self, row):
        previous_transcript = str(row['previous_transcript'])
        current_transcript = str(row['current_transcript'])
        next_transcripts = str(row['next_transcripts'])

        inputs = self.tokenizer(
            previous_transcript + " " + current_transcript + " " + next_transcripts,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=10000
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        # Debug: Print the input data
        #print(f"Prepared Inputs: {inputs}")

        return inputs

    def generate_predictions(self, inputs):
        # Model's maximum position embeddings
        max_position_embeddings = self.model.config.max_position_embeddings

        # Truncate the input tensors to the maximum position embeddings length
        truncated_input_ids = inputs['input_ids'][:, :max_position_embeddings]
        truncated_attention_mask = inputs['attention_mask'][:, :max_position_embeddings]

        try:
            with torch.no_grad():
                outputs = self.model.generate(
                    input_ids=truncated_input_ids, 
                    attention_mask=truncated_attention_mask, 
                    max_length=max_position_embeddings  # Ensure max_length does not exceed max_position_embeddings
                )
        except Exception as e:
            print(f"Error during model generation: {e}")
            return None
        
        predictions = self.tokenizer.batch_decode(outputs, skip_special_tokens=True)

        return predictions


    def process(self, records_df):
        records_df['previous_transcript'] = records_df['previous_transcript'].fillna('')
        records_df['current_transcript'] = records_df['current_transcript'].fillna('')
        records_df['next_transcripts'] = records_df['next_transcripts'].fillna('')

        predictions = []
        for idx, row in records_df.iterrows():
            inputs = self.prepare_input_data(row)
            pred = self.generate_predictions(inputs)
            predictions.append(pred[0])

        records_df['prediction'] = predictions
        response_data = records_df[['video_id', 'start_time', 'end_time', 'prediction']].to_dict(orient='records')
        return response_data


# src/pipelines/model1_pred_pipeline.py
import pandas as pd
import numpy as np
import re
import contractions
import torch
import torch.nn as nn
import torch.nn.functional as F
import requests

from tqdm import tqdm
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api.formatters import SRTFormatter
from contractions import fix
from torch.utils.data import DataLoader

class Model1Pipeline:
    def __init__(self, model, tokenizer, device, proxy_username, proxy_password):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.proxy_auth = (proxy_username, proxy_password)
        self.proxy_url = f'http://{proxy_username}:{proxy_password}@gate.smartproxy.com:10001'
        self.session = requests.Session()
        self.session.proxies = {
            'http': self.proxy_url,
            'https': self.proxy_url
        }

    def get_records_from_model1(self, video_id):
        
        df_transcript = self.get_transcriptions(video_id)
        print('YT transcript dataframe')
        print(df_transcript.shape)
        print(df_transcript.head())
        
        df_transcript_cleaned = self.cleanup(df_transcript)
        assert isinstance(df_transcript_cleaned, pd.DataFrame), "df_transcript_cleaned is not a DataFrame after cleanup"
        print('Cleaned dataframe')
        print(df_transcript_cleaned.shape)
        print(df_transcript_cleaned.head())
        
        df_transcript_segments = self.create_transcript_segments(df_transcript_cleaned)
        print('Created transcript segments')
        print(df_transcript_segments.shape)
        print(df_transcript_segments.head())
        
        df_with_prediction = self.add_predictions_to_dataframe(df_transcript_segments, self.model, self.tokenizer, self.device)
        print('Added predictions to dataframe')
        print(df_with_prediction.shape)
        print(df_with_prediction.head(20))
        
        df_extract_for_M2 = self.extract_data_for_M2(df_with_prediction, df_transcript_cleaned)
        print('Extracted data for M2')
        print(df_extract_for_M2.shape)
        print(df_extract_for_M2)
        
        df = self.prepare_input_for_M2(df_transcript_cleaned, df_extract_for_M2, transcript_column='transcript_cleaned', preprocess_func=self.perform_data_preprocessing)
        print('Prepared input for M2')
        print(df.shape)
        print(df.head())
        
        # = pd.read_csv("dataset/df_transformed_features.csv")
        return df[df["video_id"] == video_id]
    
    def get_transcriptions(self, video_id):
        try:
            # Log the proxy being used
            print(f"Using proxy: {self.proxy_url}")

            # Verify the IP before making a request
            ip_info_before = self.session.get("https://ipinfo.io").json()
            print(f"IP before fetching transcript: {ip_info_before['ip']}")

            # Call the YouTubeTranscriptApi to get video transcripts
            transcript_list = YouTubeTranscriptApi.get_transcript(video_id)

            # Verify the IP after the request
            ip_info_after = self.session.get("https://ipinfo.io").json()
            print(f"IP after fetching transcript: {ip_info_after['ip']}")

            # Convert the transcript list to a DataFrame
            data = []
            for entry in transcript_list:
                start_time = entry['start']
                duration = entry['duration']
                text = entry['text']
                end_time = start_time + duration
                data.append([video_id, start_time, end_time, text])

            df = pd.DataFrame(data, columns=['video_id', 'start_time', 'end_time', 'transcript'])
            return df

        except Exception as e:
            print(f"An error occurred: {e}")
            return pd.DataFrame()
        
    def cleanup(self, df):
        assert isinstance(df, pd.DataFrame), "Initial input df is not a DataFrame"

        df['transcript_cleaned'] = df['transcript'].apply(self.perform_data_preprocessing)
        
        # Ensure df is still a DataFrame after modification
        assert isinstance(df, pd.DataFrame), "cleanup did not return a DataFrame"
        print("cleanup completed successfully, returning a DataFrame.")
        return df

        
    def create_transcript_segments(self, df, context=10, placeholder='[PAD]'):
        chunk_dict = []
        df_segment = df.copy()
        
        # Group by video_id to ensure chunks are created for each video separately
        for video_id, group in df_segment.groupby('video_id'):
            group = group.reset_index(drop=True)  # Reset index within each video group

            # Iterate over the group to create context windows
            for i in range(len(group)):

                # Pad previous segments if we're near the start
                if i < context:
                    prev_segments = pd.Series([placeholder] * (context - i), name='transcript_cleaned')                
                    prev_segments = pd.concat([prev_segments, group.iloc[:i]['transcript_cleaned']])  # Use pd.concat instead of append
                else:
                    prev_segments = group.iloc[i - context:i]['transcript_cleaned']

                # Select current segments
                current_segments = group.iloc[i:i + context]

                # Select next segments
                next_segments = group.iloc[i + context:i + context + context]

                # Concatenate the segments for each boundary into single strings
                prev_text = ' '.join(prev_segments.tolist())  # Combine previous segments into a single string
                current_text = ' '.join(current_segments['transcript_cleaned'].tolist())  # Combine current segments into a single string
                next_text = ' '.join(next_segments['transcript_cleaned'].tolist())  # Combine next segments into a single string

                # Combine into a single chunk (video_id, start_time, transcript, label)
                chunk_transcript = f"[CLS] [PREVIOUS] {prev_text} [CURRENT][PREDICT] {current_text} [NEXT] {next_text} [SEP]"
                chunk_video_id = current_segments.iloc[0]['video_id']  # Assuming all have the same video_id
                chunk_start_time = current_segments.iloc[0]['start_time']  # Start time from the first current segment
                #chunk_label = current_segments.iloc[0]['labels']  # Label from the first current segment

                # Store the chunk in a dictionary
                chunk_dict.append({
                    'video_id': chunk_video_id,
                    'start_time': chunk_start_time,
                    'prev_segments': prev_text,
                    'current_segments': current_text,
                    'next_segments': next_text,
                    'transcript_segments': chunk_transcript,  # Now this contains the concatenated string
                    #'label': chunk_label
                })

        
        # Create the final DataFrame with chunks
        df_segment = pd.DataFrame(chunk_dict)

        #======================================================================================================================

        # Sort the DataFrame to ensure proper order
        df_segment = df_segment.sort_values(by=['video_id', 'start_time']).reset_index(drop=True)

        '''
        # Assign chapter numbers based on where label == 1
        df['chapter_number'] = (
            df.groupby('video_id')['label']
            .apply(lambda x: (x == 1).cumsum())
            .reset_index(level=0, drop=True)
        )
        # print(df.index)
        # print(df.groupby('video_id')['label'].apply(lambda x: (x == 1).cumsum()).index)

        # Forward-fill chapter numbers within each video group
        df['chapter_number'] = df.groupby('video_id')['chapter_number'].ffill()

        # Replace NaN with 0 for rows before the first chapter start
        df['chapter_number'] = df['chapter_number'].fillna(0).astype(int)
        '''

        return df_segment
    
    def tokenize_and_encode_data(self, transcripts, tokenizer, max_length=512):
        # Tokenize the transcripts using the tokenizer
        encoded_inputs = tokenizer(transcripts, 
                                padding='max_length', 
                                truncation=True, 
                                max_length=max_length, 
                                return_tensors="pt")
        
        # Decode the tokenized inputs back to readable text
        decoded_text = tokenizer.decode(encoded_inputs["input_ids"][0], skip_special_tokens=True)

        return encoded_inputs  # Return the encoded inputs for further processing if needed
    
    def add_predictions_to_dataframe(self, df, model, tokenizer, device, threshold=0.50, batch_size=16, max_length=512):
        """
        Add model predictions and raw probabilities to the original DataFrame.

        Parameters:
            df (pd.DataFrame): Original DataFrame containing the data.
            model (torch.nn.Module): Trained BERT model.
            tokenizer (BertTokenizer): Tokenizer used with the model.
            threshold (float): Threshold for converting probabilities into binary predictions.
            batch_size (int): Batch size for evaluation.
            max_length (int): Maximum sequence length for tokenization.

        Returns:
            pd.DataFrame: Updated DataFrame with new columns for raw probabilities and predictions.
        """
        model.eval()  # Ensure model is in evaluation mode

        # Tokenize the data
        encoded_inputs = self.tokenize_and_encode_data(df["transcript_segments"].tolist(), tokenizer, max_length=max_length)
        
        # Create a DataLoader
        dataset = BinaryClassificationDataset(encoded_inputs, labels=pd.Series([0] * len(df)))  # Dummy labels
        data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        # Collect predictions and probabilities
        all_probs = []
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Predicting", unit="batch"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)

                # Get model outputs
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = outputs.logits
                probabilities = torch.sigmoid(logits).squeeze()

                # Append probabilities
                if probabilities.ndim == 0:
                    all_probs.append(probabilities.cpu().item())  # Scalar case
                else:
                    all_probs.extend(probabilities.cpu().numpy())  # Batch case

        print(all_probs)  # Debug print

        # Add raw probabilities as a new column in the DataFrame
        df["probabilities"] = all_probs
        
        # Calculate dynamic threshold (90th percentile of probabilities)
        threshold = np.percentile(all_probs, 90)

        # Add predictions as a new column based on the given threshold
        df["predictions"] = (df["probabilities"] > threshold).astype(int)
        
        # Add a new column `post_prediction` that copies the `predictions` column
        df["post_prediction"] = df["predictions"]

        # Update the first row for each `video_id` to `1` in the `post_prediction` column
        df.loc[df.groupby("video_id").head(1).index, "post_prediction"] = 1

        return df
    
    def extract_data_for_M2(self, df, df_ref):
        # Prepare the final DataFrame
        df_for_model2 = df.copy()  # Work with a copy of df to avoid modifying the original

        # Initialize the 'end_time' column to None
        df_for_model2["end_time"] = None

        # Create a dictionary to map video_id to max end_time from df_ref
        max_end_times = df_ref.groupby('video_id')['end_time'].max().to_dict()

        # Group by video_id and process chapters
        for video_id, group in df_for_model2.groupby("video_id"):
            # Find the indices where post_prediction is 1
            indices = group.index[group["post_prediction"] == 1].tolist()

            for i, start_idx in enumerate(indices):
                if i < len(indices) - 1:
                    # Set end_time to the start_time of the next chapter
                    end_time = df_for_model2.loc[indices[i + 1], "start_time"]
                else:
                    # For the last chapter, get the max end_time from df_ref
                    end_time = max_end_times.get(video_id, None)
                df_for_model2.loc[start_idx, "end_time"] = end_time

        # Filter rows where both start_time and end_time are not null
        df_final_M2 = df_for_model2.dropna(subset=["start_time", "end_time"])

        # Retain only specific columns and reset the index
        df_final_M2 = df_final_M2[["video_id", "start_time", "end_time"]].reset_index(drop=True)    

        # Return the processed DataFrame for Model 2
        return df_final_M2

    
    def assign_transcript_to_chapters(self, df_transcript_cleaned, df_json, transcript_column='transcript'):
        #print(f"Type of df_transcript at start: {type(df_transcript_cleaned)}")  # Debug print
        #print(f"Type of df_json at start: {type(df_json)}")  # Debug print
        assert isinstance(df_transcript_cleaned, pd.DataFrame), "df_transcript is not a DataFrame"
        assert isinstance(df_json, pd.DataFrame), "df_json is not a DataFrame"
        
        df_json['transcript'] = None  # Placeholder for transcript text for each chapter

        # Iterate over each row in df_json to get video_id, start_time, and end_time for each chapter
        for idx, row in df_json.iterrows():
            video_id = row['video_id']  # Get the video ID for the chapter
            start_time = row['start_time']
            end_time = row['end_time']

            #print(f"Processing video_id: {video_id}, start_time: {start_time}, end_time: {end_time}")

            # Debugging Statements for video_id and Indexing
            #print(f"df_transcript['video_id'] sample:\n{df_transcript_cleaned['video_id'].head()}")
            #print(f"Type of video_id: {type(video_id)}")
            if video_id not in df_transcript_cleaned['video_id'].values:
                #print(f"video_id {video_id} not found in df_transcript")
                continue

            # Filter df_transcripts to get the transcript for the current video_id
            video_transcript = df_transcript_cleaned[df_transcript_cleaned['video_id'] == video_id]
            #print(f"Video transcript shape: {video_transcript.shape}")
            #print(f"Video transcript head:\n{video_transcript.head()}")

            # Filter transcript rows that fall within the chapter's time range
            chapter_transcript = video_transcript[
                (video_transcript['start_time'] >= start_time) &
                (video_transcript['start_time'] < end_time)
            ]
            #print(f"Chapter transcript shape: {chapter_transcript.shape}")
            #print(f"Chapter transcript head:\n{chapter_transcript.head()}")

            # Join all text entries in the 'transcript' column of chapter_transcript DataFrame into a single string
            df_json.at[idx, 'transcript'] = " ".join(chapter_transcript[transcript_column].dropna().astype(str).tolist())

        return df_json

    
    def create_feature_format(self, df):
        # Prepare the final DataFrame
        final_data = []

        # Loop through each unique video ID in the DataFrame
        for video_id in df['video_id'].unique():
            # Filter the DataFrame for the current video ID
            video_data = df[df['video_id'] == video_id]
            # Get the number of chapters for the current video
            chapters = video_data.shape[0]

            # Loop through each chapter in the video
            for i in range(chapters):
                # Get the cleaned transcript for the current chapter
                current_transcript = video_data.iloc[i]['transcript_cleaned']

                # Get all previous transcripts as a single string, or an empty string if it's the first chapter
                previous_transcript = ' '.join(video_data.iloc[:i]['transcript_cleaned'].tolist()) if i > 0 else ''

                # Get all next transcripts as a single string, or an empty string if it's the last chapter
                next_transcripts = ' '.join(video_data.iloc[i + 1:]['transcript_cleaned'].tolist()) if i < chapters - 1 else ''

                # Format the feature string with transcripts, stripping any leading/trailing whitespace
                feature_format = f"[{previous_transcript.strip()}], [{current_transcript.strip()}], [{next_transcripts.strip()}]"

                # Retrieve start and end times
                start_time = video_data.iloc[i]['start_time']
                end_time = video_data.iloc[i]['end_time']

                # Base dictionary for the current chapter
                chapter_data = {
                    'video_id': video_id,
                    'chapter_number': i + 1,
                    'start_time': start_time,
                    'end_time': end_time,
                    'feature_format': feature_format,
                    'previous_transcript': previous_transcript.strip(),
                    'current_transcript': current_transcript.strip(),
                    'next_transcripts': next_transcripts.strip(),
                }

                # Add 'label' only if 'sentences' column exists
                if 'sentences' in video_data.columns:
                    chapter_data['label'] = video_data.iloc[i]['sentences'].strip()

                # Append the chapter data to the final data list
                final_data.append(chapter_data)

        # Create a DataFrame from the final data list
        final_df = pd.DataFrame(final_data)

        return final_df
    
    def prepare_input_for_M2(
        self,
        df_transcript_cleaned,
        df_extracted_M2,
        transcript_column='transcript_cleaned',
        preprocess_func=None
    ):
        
        # Step 1: Assign transcripts to chapters
        print('Step 1')
        assert isinstance(df_transcript_cleaned, pd.DataFrame), "df_transcript_cleaned is not a DataFrame"
        df_chapter_transcript_M2 = self.assign_transcript_to_chapters(
            df_transcript_cleaned, df_extracted_M2
        )

        print('Step 2')
        # Step 2: Preprocess the transcript column if a preprocessing function is provided
        if preprocess_func:
            df_chapter_transcript_M2[transcript_column] = df_chapter_transcript_M2['transcript'].apply(preprocess_func)

        print('Step 3')
        # Step 3: Create transformed feature format for prediction
        df_transformed_features_predict_M2 = self.create_feature_format(df_chapter_transcript_M2)

        return df_transformed_features_predict_M2
    
    slang_dict = {
        "u": "you",
        "ur": "your",
        "lol": "laughing out loud",
        "idk": "i don't know",
        "btw": "by the way",
        "gonna": "going to",
        "wanna": "want to",
        "tho": "though",
        "im": "i am",
        # Add more slang mappings as needed
    }
    
    def replace_slang(self, text, slang_dict):
        """
        Replace slang words in the input text based on a given slang dictionary.
        
        Args:
            text (str): The input text containing slang.
            slang_dict (dict): Dictionary mapping slang words to their expanded forms.
        
        Returns:
            str: Text with slang replaced by standard forms.
        """
        words = text.split()
        return " ".join([slang_dict.get(word, word) for word in words])
    
    def perform_data_preprocessing(self, text):
        """
        Preprocesses input text by cleaning, tokenizing, removing stop words, and lemmatizing.
        
        Args:
            text (str): The input text to preprocess.
        
        Returns:
            str: The cleaned and preprocessed text, or "no content" if the input is invalid or empty.
        """
        
        if not isinstance(text, str):
            return "[no content]"  # Return "no content" for non-string input

        # Check if the string is only whitespace
        if text.strip() == '':
            return "[no content]"  # Return "no content" if the input is empty or only whitespace

        # Remove newline characters (to keep text on a single line)
        text = text.replace('\r\n', ' ').replace('\n', ' ')

        # Handle contractions (e.g., "it's" -> "it is")
        text = contractions.fix(text)

        # Lowercase the text for consistency
        text = text.lower()

        # Replace slang words
        text = self.replace_slang(text, self.slang_dict)

        # Remove unnecessary spaces before punctuation
        text = re.sub(r"\s+([,.?!])", r"\1", text)
        
        # Add space after punctuation if missing
        text = re.sub(r"([,.?!])(?=\w)", r"\1 ", text)

        # Remove any text inside square brackets, including the brackets themselves
        text = re.sub(r'\[.*?\]', '', text)

        # Remove numbers and parentheses
        text = re.sub(r'\d+\)\s*', '', text)  # Removes the number and the parenthesis
        
        # Remove numbers
        text = re.sub(r'\d+', '', text)

        # Check for non-ASCII characters
        cleaned_text = re.sub(r'[^\x00-\x7F]+', '', text).strip()
        
        # Check if the cleaned text is empty after preprocessing
        if cleaned_text == '':
            return "[no content]"  # Return "no content" if the text is empty after processing

        return cleaned_text

class BinaryClassificationDataset(torch.utils.data.Dataset):
    """
    Dataset for binary classification.
    """
    def __init__(self, encoded_inputs, labels):
        self.encoded_inputs = {key: val.clone().detach() for key, val in encoded_inputs.items()}
        self.labels = torch.tensor(labels.values)  # This ensures labels are a numpy array, which works with torch.tensor

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach().long() for key, val in self.encoded_inputs.items()}
        item['labels'] = self.labels[idx].clone().detach().float()

        return item
    
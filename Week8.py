import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os

from google.cloud import storage

# Initialize the client
client = storage.Client()

# Define the source bucket and file name
source_bucket_name = "source_bucket_name"
source_blob_name = "path/to/source/file"

# Define the destination bucket and file name
destination_bucket_name = "destination_bucket_name"
destination_blob_name = "path/to/destination/file"

# Get the source bucket and file
source_bucket = client.bucket(source_bucket_name)
source_blob = source_bucket.blob(source_blob_name)

# Download the file to a local file
local_file_path = "/tmp/local_file"
source_blob.download_to_filename(local_file_path)

# Upload the file to the destination bucket
destination_bucket = client.bucket(destination_bucket_name)
destination_blob = destination_bucket.blob(destination_blob_name)
destination_blob.upload_from_filename(local_file_path)

# Delete the local file
os.remove(local_file_path)

################################################################

import praw
import firebase_admin
from firebase_admin import credentials
from firebase_admin import firestore

# Initialize Firebase Admin SDK
cred = credentials.Certificate('path/to/credential.json')
firebase_admin.initialize_app(cred)

# Initialize the Reddit API client
reddit = praw.Reddit(client_id='your_client_id', client_secret='your_client_secret', user_agent='your_user_agent')

# Define the Cloud Function
def download_subreddit_data_to_firestore(request):
    # Get the subreddit name from the request payload
    request_json = request.get_json()
    subreddit_name = request_json['subreddit_name']

    # Get a reference to the Firestore database
    db = firestore.client()

    # Get the subreddit object
    subreddit = reddit.subreddit(subreddit_name)

    # Loop through the hot posts in the subreddit and save them to Firestore
    for submission in subreddit.hot():
        # Define the document ID
        document_id = f"{subreddit_name}_{submission.id}"

        # Define the document data
        document_data = {
            'title': submission.title,
            'author': submission.author.name,
            'score': submission.score,
            'url': submission.url,
            'created_utc': submission.created_utc
        }

        # Save the document to Firestore
        db.collection('subreddits').document(document_id).set(document_data)

    # Return a success message
    return f"Data from /r/{subreddit_name} saved to Firestore!"

import requests
import pandas as pd
import json

# reddit credentials file should be placed outside of 
# class exercises directory and should NOT be pushed to github
REDDIT_CREDENTIALS_FILE = (r"C:\Users\jackw\OneDrive\Documents\02a_MachineLearningInPython\reddit_credentials.json")

# This function gets the access token for Reddit API calls 
# It uses the Reddit Credentials file to authenticate
def get_access_token():
	rc = json.load(open(REDDIT_CREDENTIALS_FILE))
	auth = requests.auth.HTTPBasicAuth(rc['client_id'], rc['secret_token'])
	data = {
		'grant_type': 'password',
		'username': rc['username'],
		'password': rc['password']
	}
	headers = {'User-Agent': 'scraper/0.0.1'}
	resp = requests.post(
		'https://www.reddit.com/api/v1/access_token',
		auth=auth,
		data=data, 
		headers=headers)
	TOKEN = resp.json()['access_token']
	return TOKEN

# This function connects to reddit api using access token
# and downloads new posts from a subreddit
def get_new_posts(subreddit, TOKEN, limit=5):
	headers = {'User-Agent': 'scraper/0.0.1'}
	headers = {**headers, **{'Authorization': f"bearer {TOKEN}"}}
	params = {'limit': limit}
	url = "https://oauth.reddit.com/" + subreddit + "/new"
	resp = requests.get(url, headers=headers, params=params)
	posts = [p['data'] for p in resp.json()['data']['children']] 
	return posts

# testing the fuctions
token = get_access_token()
posts = get_new_posts('r/movies', token)

# Writing the downloaded data to json file
with open('reddit_posts.json', 'w', encoding='utf-8') as f:
	json.dump(posts, f, ensure_ascii=False, indent=4)
	
# writing the downloaded data to csv
df = pd.DataFrame()
for post in posts:
	df = df.append(post, ignore_index=True)
df.to_csv('reddit_posts.csv', index = False)
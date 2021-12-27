# Twitter Wrapped Bot
This project brings the popular Spotify Wrapped concept to Twitter.

Two main files used in this project:
- `stream_mentions.py`
This file is responsible for authenticating with the Twitter API and streaming real-time bot mentions.
- `generate_image.py`
This file is used to generate visuals that include the user's statistics.

# Images / Statistics generated
- Most liked tweet.
- Most retweeted tweet.
- Most quoted tweet.
- Word Cloud of frequently used words in tweets.
- Num. of tweets with more than 100 likes, 500 likes, 1,000 likes and 10,000 likes.
- Sentiment analysis of tweets from -100 to 100.

# Example
![Example](https://i.imgur.com/IRqGRui.png = 250x250)

# How To Run
1. Clone this repository
2. Apply for a Twitter developer account
3. Enter Twitter API details in twittercredentials.py
5. Run `stream_mentions.py`

# Optional
You may delete the following files, as it wont have any effect on the program
- `/python_api_calls`
- `alternative_to_streaming.py`
- `file_queue_test.py`
- `queue_test.txt`
- `two_image_format.py`
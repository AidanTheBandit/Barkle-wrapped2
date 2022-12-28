from PIL import Image, ImageFont, ImageDraw
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob  # sentiment analysis
import tweepy, twitter_credentials
import textwrap # not used currently, implement in future
import re, os # regex / saving and loading tweets
import numpy as np  # numerical python library
import pandas as pd  # store content into dataframes
import sys  # running python file with args, remove later


# Twitter API Client
def getClient():
    client = tweepy.Client(bearer_token=twitter_credentials.bearer_token,
                           consumer_key=twitter_credentials.consumer_key,
                           consumer_secret=twitter_credentials.consumer_secret,
                           access_token=None, access_token_secret=None)
    return client


# Return user information
def getUserInfo(user):
    client = getClient()
    user = client.get_user(username=user)
    return user.data


# Return recent tweets of user
def getUserRecentTweets(id):
    client = getClient()
    user_tweets = client.get_users_tweets(id=id,
                                          tweet_fields=['public_metrics,created_at'],
                                          exclude=['retweets', 'replies'],
                                          max_results=100,
                                          #start_time = '2021-09-02T00:00:00.000Z'
                                          )

    return user_tweets


# Store recent user tweets in file
def storeUserTweets(username, user_tweets):

    file_path = 'user_tweets/' + username + '.txt'

    # user has tweets
    if user_tweets.data is not None and len(user_tweets.data) > 0:

        # user tweets not stored in file
        if not os.path.exists(file_path):

            # create new file
            file = open(file_path, 'w', encoding='utf-8')

            # write each tweet into new file
            for x in user_tweets.data:
                file.write(cleanTweet(str(x)) + '\n')

            file.close()
            return True

        else:
            # return as error in future
            # print("User tweets file already exists")
            return False

    else:
        # user has no tweets
        # print("No tweets")
        return False


# Remove special characters and hyperlinks
# Modified code from freeCodeCamp.org
def cleanTweet(tweet):

    # replace ’ with '
    tweet = tweet.replace('’', '\'')

    return ' '.join(re.sub("(@[A-Za-z0-9]+)|([^0-9A-Za-z' \t])|(\w+:\/\/\S+)", " ", tweet).split())


# Add public metrics to dataframe
def tweetsToDataFrame(tweets):

    # Create new dataframe with tweet text
    df = pd.DataFrame(
        data=[tweet.text for tweet in tweets], columns=['tweets'])

    # Create columns for metrics
    df['retweet_count'] = np.array(
        [tweet.public_metrics.get('retweet_count') for tweet in tweets])
    df['reply_count'] = np.array(
        [tweet.public_metrics.get('reply_count') for tweet in tweets])
    df['like_count'] = np.array(
        [tweet.public_metrics.get('like_count') for tweet in tweets])
    df['quote_count'] = np.array(
        [tweet.public_metrics.get('quote_count') for tweet in tweets])
    df['created_at'] = np.array([tweet.created_at for tweet in tweets])

    return df


# Sentiment analysis, returns
# -1 for negative
# 0 for neutral
# 1 for positive
def analyse_sentiment(tweet):
    analysis = TextBlob(cleanTweet(tweet))

    if analysis.sentiment.polarity > 0:
        return 1
    elif analysis.sentiment.polarity == 0:
        return 0
    else:
        return -1

# Watermark text
tweet_wrapped_watermark = ["@TweetWrapped"]

# Fonts used in images
global_font = {
    "title": ImageFont.truetype("fonts/theboldfont.ttf", 70),
    "text": ImageFont.truetype("fonts/coolvetica-rg.otf", 60),
    "number": ImageFont.truetype("fonts/theboldfont.ttf", 100),
    "watermark": ImageFont.truetype("fonts/theboldfont.ttf", 40)
}

# Font colours
global_font_colour = {
    "title": (228, 179, 143),
    "text": (179, 145, 143),
    "number": (192, 222, 106),
    "watermark": (228, 179, 143)
}

# Text position and spacing
# global_text_pos = {
#     "x": 50,
#     "y": 50,
#     "spacer": 50
# }

# Larger text position and spacing
global_text_pos = {
    "x": 100,
    "y": 100,
    "spacer": 100
}


# Generate image 1
def generate_highest_metrics_image(username, most_likes, most_retweets, most_quotes):

    # Open black image
    img = Image.open("img/templates/purple_1000x1000.png")
    draw = ImageDraw.Draw(img)

    # Template size
    image_width, image_height = img.size

    font = global_font
    font_colour = global_font_colour

    x_pos = global_text_pos["x"]
    y_pos = global_text_pos["y"]
    spacer = global_text_pos["spacer"]

    # Content
    if most_likes > 1000:
        popularity_txt = "You're so Popular!"
    elif most_likes > 15:
        popularity_txt = "You're Growing!"
    elif most_likes > 5:
        popularity_txt  = "You're Doing OK!"
    elif most_likes > 1:
        popularity_txt = "You're Doing Meh."
    else:
        popularity_txt = "You're not popular :("
    
    title_text = [username + ",", popularity_txt]

    metrics_text = ["Most Stars", "Most Renotes", "Most Quotes"]
    metrics_values = [str(most_likes), str(most_retweets), str(most_quotes)]

    # Draw title
    draw.text((x_pos, y_pos), title_text[0],
              font_colour["title"], font=font["title"])
    draw.text((x_pos, y_pos + spacer*1.1),
              title_text[1], font_colour["title"], font=font["title"])

    # Draw metric text
    draw.text((x_pos, y_pos + spacer*3),
              metrics_text[0], font_colour["text"], font=font["text"])
    draw.text((x_pos, y_pos + spacer*4.5),
              metrics_text[1], font_colour["text"], font=font["text"])
    draw.text((x_pos, y_pos + spacer*6),
              metrics_text[2], font_colour["text"], font=font["text"])

    # Width to right align
    num_width_0 = font["number"].getsize(metrics_values[0])[0]
    num_width_1 = font["number"].getsize(metrics_values[1])[0]
    num_width_2 = font["number"].getsize(metrics_values[2])[0]

    # Draw metric values
    temp_x_pos = image_width - x_pos - num_width_0
    draw.text((temp_x_pos, y_pos + spacer*3),
              metrics_values[0], font_colour["number"], font=font["number"])

    temp_x_pos = image_width - x_pos - num_width_1
    draw.text((temp_x_pos, y_pos + spacer*4.5),
              metrics_values[1], font_colour["number"], font=font["number"])

    temp_x_pos = image_width - x_pos - num_width_2
    draw.text((temp_x_pos, y_pos + spacer*6),
              metrics_values[2], font_colour["number"], font=font["number"])

    # Draw watermark
    draw.text((image_width - 350, image_height - 60),
              tweet_wrapped_watermark[0], font_colour["title"], font=font["watermark"])

    # Save image
    img.save("img/outputs/highest_metrics/" +
             username + ".png")
    #print("Created highest metrics image.")


# Generate image 2
def generate_word_cloud_image(username):

    # Get user data
    text = open('user_tweets\\' + username + '.txt',
                'r', encoding='utf-8').read()

    # Stop words, add 'gt' to it
    stopwords = STOPWORDS.add('gt')

    # Mask
    custom_mask = np.array(Image.open(
        'img\\masks\\twitter_logo_1000x1000.png'))
    font = 'fonts\\SFProDisplay-Light.ttf'

    # WordCloud attributes
    wordCloud = WordCloud(
        width=1000, height=1000,
        font_path=font,
        mask=custom_mask,
        stopwords=stopwords,
        background_color=(80, 54, 89),
        color_func=lambda *args, **kwargs: (199, 219, 115),  # text colour
        include_numbers=False
        # margin = 10, background_color = None, mode = 'RGBA',
    )

    # Generate
    wordCloud.generate(text)

    # Use colour of mask image
    ## image_colours = ImageColorGenerator(custom_mask)
    ## wordCloud.recolor(color_func = image_colours)

    # Store to file
    wordCloud.to_file(
        'img\\outputs\\word_clouds\\' + username + '.png')

    # Open pre-gen word cloud
    img = Image.open(
        "img/outputs/word_clouds/" + username + ".png")
    draw = ImageDraw.Draw(img)

    # Template size
    image_width, image_height = img.size

    font = global_font
    font_colour = global_font_colour

    x_pos = global_text_pos["x"]
    y_pos = global_text_pos["y"]
    spacer = global_text_pos["spacer"]

    # Content
    title_text = ["What you're Barking."]
    #title_text = [username + ",", "Tweets Visualized."]

    # Draw title
    # Since word cloud image is large, move text away from it
    draw.text((x_pos-25, y_pos-25), title_text[0],
              font_colour["title"], font=font["title"])
    #draw.text((x_pos, y_pos + spacer), title_text[1], font_colour["title"], font = font["title"])

    # Draw watermark
    draw.text((image_width - 350, image_height - 60),
              tweet_wrapped_watermark[0], font_colour["title"], font=font["watermark"])

    # Save
    img.save("img/outputs/word_clouds/" + username + ".png")
    #print("Created word cloud image.")


# Generate image 3
def generate_likes_performance_image(username, likes_performance):

    # Open black image
    img = Image.open("img/templates/purple_1000x1000.png")
    draw = ImageDraw.Draw(img)

    # Template size
    image_width, image_height = img.size

    font = global_font
    font_colour = global_font_colour

    x_pos = global_text_pos["x"]
    y_pos = global_text_pos["y"]
    spacer = global_text_pos["spacer"]

    lp_title_text = ["Get Any Big Barks?"]  # LP = 'likes performance'

    lp_text = ["> 1 likes.", "> 5 likes.",
               "> 15 likes.", "> 10,000 likes."]
    lp_values = [str(likes_performance[100]), str(likes_performance[500]), str(
        likes_performance[1000]), str(likes_performance[10000])]
    lp_values_additional_text = ["barks"]

    # Likes Performance section
    # Right align
    #title_width = font["title"].getsize(lp_title_text[0])[0]
    #temp_x_pos = image_width - x_pos - title_width

    # Width to right align
    txt_width_0 = font["text"].getsize(lp_text[0])[0]
    txt_width_1 = font["text"].getsize(lp_text[1])[0]
    txt_width_2 = font["text"].getsize(lp_text[2])[0]
    txt_width_3 = font["text"].getsize(lp_text[3])[0]

    # Move base-level y-pos down
    # not relevant with 4 images, so ignore for now
    #y_pos = image_height/1.8

    # Draw title
    # Right-align not used currently
    # use temp_x_pos for right-align
    draw.text((x_pos, y_pos),
              lp_title_text[0], font_colour["title"], font=font["title"])

    # Draw lp text
    temp_x_pos = image_width - x_pos - txt_width_0
    draw.text((temp_x_pos, y_pos + spacer*1.8),
              lp_text[0], font_colour["text"], font=font["text"])

    temp_x_pos = image_width - x_pos - txt_width_1
    draw.text((temp_x_pos, y_pos + spacer*3.3),
              lp_text[1], font_colour["text"], font=font["text"])

    temp_x_pos = image_width - x_pos - txt_width_2
    draw.text((temp_x_pos, y_pos + spacer*4.8),
              lp_text[2], font_colour["text"], font=font["text"])

    temp_x_pos = image_width - x_pos - txt_width_3
    draw.text((temp_x_pos, y_pos + spacer*6.3),
              lp_text[3], font_colour["text"], font=font["text"])

    # Draw lp values
    draw.text((x_pos, y_pos + spacer*1.75),
              lp_values[0], font_colour["number"], font=font["number"])
    draw.text((x_pos, y_pos + spacer*3.25),
              lp_values[1], font_colour["number"], font=font["number"])
    draw.text((x_pos, y_pos + spacer*4.75),
              lp_values[2], font_colour["number"], font=font["number"])
    draw.text((x_pos, y_pos + spacer*6.25),
              lp_values[3], font_colour["number"], font=font["number"])

    # Draw additional text 'twitter' next to lp values
    value_width_1 = font["title"].getsize(lp_values[0])[0]
    value_width_2 = font["title"].getsize(lp_values[1])[0]
    value_width_3 = font["title"].getsize(lp_values[2])[0]
    value_width_4 = font["title"].getsize(lp_values[3])[0]
    draw.text((value_width_1 + x_pos + 50, y_pos + spacer*1.8),
              lp_values_additional_text[0], font_colour["text"], font=font["text"])
    draw.text((value_width_2 + x_pos + 50, y_pos + spacer*3.3),
              lp_values_additional_text[0], font_colour["text"], font=font["text"])
    draw.text((value_width_3 + x_pos + 50, y_pos + spacer*4.8),
              lp_values_additional_text[0], font_colour["text"], font=font["text"])
    draw.text((value_width_4 + x_pos + 50, y_pos + spacer*6.3),
              lp_values_additional_text[0], font_colour["text"], font=font["text"])

    # Draw watermark
    draw.text((image_width - 350, image_height - 60),
              tweet_wrapped_watermark[0], font_colour["title"], font=font["watermark"])

    # Save image
    img.save("img/outputs/likes_performance/" +
             username + ".png")
    #print("Created likes performance image.")


# Generate image 4
def generate_sentiment_analysis_image(username, sentiment):

    # Open black image
    img = Image.open("img/templates/purple_1000x1000.png")
    draw = ImageDraw.Draw(img)

    # Template size
    image_width, image_height = img.size

    font = global_font
    font_colour = global_font_colour

    x_pos = global_text_pos["x"]
    y_pos = global_text_pos["y"]
    spacer = global_text_pos["spacer"]

    # Classify based on numerical sentiment value (-100 to 100)
    if sentiment > 10:
        sentiment_class = "SUPER HAPPY!"  # EMOJI
        sentiment_emoji = Image.open(
            "img/emojis/grinning-face-with-sweat_1f605.png")
    elif sentiment > 5:
        sentiment_class = "HAPPY!"
        sentiment_emoji = Image.open(
            "img/emojis/beaming-face-with-smiling-eyes_1f601.png")
    elif sentiment > 0:
        sentiment_class = "KINDA HAPPY..."  # EMOJI
        sentiment_emoji = Image.open(
            "img/emojis/emoji-upside-down-face_1f643.png")
    elif sentiment > -5:
        sentiment_class = "SAD!"
        sentiment_emoji = Image.open(
            "img/emojis/face-with-head-bandage_1f915.png")
    else:
        sentiment_class = "DOWN BAD!"
        sentiment_emoji = Image.open("img\emojis\sleepy-face_1f62a.png")

    # Resize emoji
    (emoji_width, emoji_height) = (
        sentiment_emoji.width/2, sentiment_emoji.height/2)
    sentiment_emoji = sentiment_emoji.resize(
        (int(emoji_width), int(emoji_height)))

    # Sentiment title
    sentiment_title = ["How were you feeling?", "Happy or Sad?"]

    # Sentiment text
    sentiment_text = ["Emotionally  your  barks", "scored", str(
        sentiment), "meaning  you", "were...", sentiment_class]

    # Move base-level y-pos down
    # not relevant for 4 images, so ignore
    # y_pos = image_height/1.5

    # Draw sentiment title
    # Not using right-align
    # use temp_x_pos for right-align
    #title_width = font["title"].getsize(sentiment_title[0])[0]
    #temp_x_pos = image_width - x_pos - title_width
    draw.text((x_pos, y_pos),
              sentiment_title[0], font_colour["title"], font=font["title"])
    draw.text((x_pos, y_pos + spacer*1.1),
              sentiment_title[1], font_colour["title"], font=font["title"])

    # Item 1
    # Draw text 1
    draw.text((x_pos, y_pos + spacer * 3),
              sentiment_text[0], font_colour["text"], font=font["text"])

    # Item 2
    # Draw text 2
    draw.text((x_pos, y_pos + spacer * 4.25),
              sentiment_text[1], font_colour["text"], font=font["text"])

    # Item 3
    # Draw sentiment value
    # Get width of text to prevent overlap
    txtwrap_x_pos = font["text"].getsize(sentiment_text[1])[0] + x_pos + 25
    draw.text((txtwrap_x_pos, y_pos + spacer * 4.2),
              sentiment_text[2], font_colour["number"], font=font["number"])

    # Item 4
    # Draw text 4
    # Move text to be positioned after value number
    txtwrap_x_pos = font["number"].getsize(sentiment_text[2])[
        0] + txtwrap_x_pos + 25
    draw.text((txtwrap_x_pos, y_pos + spacer * 4.25),
              sentiment_text[3], font_colour["text"], font=font["text"])

    # Item 5
    # Draw text 5
    draw.text((x_pos, y_pos + spacer * 5.5),
              sentiment_text[4], font_colour["text"], font=font["text"])

    # Item 6
    # Draw sentiment class
    temp_x_pos = font["text"].getsize(sentiment_text[4])[0] + x_pos + 25
    draw.text((temp_x_pos, y_pos + spacer * 5.5),
              sentiment_text[5], font_colour["number"], font=font["number"])

    # If sentiment class is too long, draw emoji on new line
    # e.g 'SUPER HAPPY!' is too long
    if sentiment_class == "SUPER HAPPY!" or sentiment_class == "KINDA HAPPY...":
        # Draw sentiment emoji on new line
        img.paste(sentiment_emoji, (x_pos, int(y_pos + spacer * 6.8)))
    else:
        # Draw sentiment emoji after sentiment class
        txtwrap_x_pos = font["number"].getsize(sentiment_text[5])[
            0] + temp_x_pos + 25
        img.paste(sentiment_emoji, (txtwrap_x_pos, int(y_pos + spacer * 5.5)))

    # Draw watermark
    draw.text((image_width - 350, image_height - 60),
              tweet_wrapped_watermark[0], font_colour["title"], font=font["watermark"])

    # Save
    img.save("img/outputs/sentiment_analysis/" + username + ".png")
    #print("Created sentiment analysis image.")


# Test image gen without calling api
#if __name__ == "__main__":
#      # main(sys.argv[1])
#    username = "FinessTV"
#    most_likes = 5
#    most_retweets = 10
#    most_quotes = 2
#      likes_performance = {
#          100: 19,
#          500: 3,
#          1000: 0,
#          10000: 0
#      }
#      sentiment = 11
#      generate_highest_metrics_image(username, most_likes, most_retweets, most_quotes)
#      generate_word_cloud_image(username)
#      generate_likes_performance_image(username, likes_performance)
#      generate_sentiment_analysis_image(username, sentiment)


# Main method called by stream_mentions.py
def main(username):

    # Get user info, such as id
    user = getUserInfo(username)
    # Get tweets of user by id
    try:
        user_tweets = getUserRecentTweets(user.id)
    except Exception as e:
        print(e)
        return False

    # If user has already been processed, i.e. already used bot...
    # storeUserTweets will return false, and program will stop...
    # else, continue
    if storeUserTweets(username, user_tweets):

        # Get user stats
        df = tweetsToDataFrame(user_tweets.data)

        # Carry out sentiment analysis
        df['sentiment'] = np.array([analyse_sentiment(tweet)
                                    for tweet in df['tweets']])

        # Get average sentiment of all user tweets
        sentiment = np.average(df['sentiment']) * 100
        # Remove repeating demial e.g. 12.11111...
        sentiment = float("{0:.2f}".format(sentiment))

        # pd.set_option('display.max_rows', 100) # Change how many rows df prints
        # print(df.head(100))  # Print dataframe

        # print(dir(user_tweets.data)) # What attributes exist
        # print(user_tweets.data[0].public_metrics)

        # Get highest metrics
        most_likes = np.max(df['like_count'])
        most_retweets = np.max(df['retweet_count'])
        most_quotes = np.max(df['quote_count'])

        # How many tweets with more than X likes
        likes_performance = {
            100: len(df[df['like_count'] > 1]),
            # 500 likes metric only used in 4 image format
            500: len(df[df['like_count'] > 5]),
            1000: len(df[df['like_count'] > 15]),
            10000: len(df[df['like_count'] > 10000])
        }

        # Old 2 image format #
        # generate_highest_metrics_and_likes_performance_image(username,
        #                                                      most_likes,
        #                                                      most_retweets,
        #                                                      most_quotes,
        #                                                      likes_performance)

        # generate_word_clouds_and_sentiment_analysis_image(username, sentiment)

        # New 4 image format #

        # Generate image 1 - Highest metrics
        generate_highest_metrics_image(
            username, most_likes, most_retweets, most_quotes)

        # Generate image 2 - Word cloud
        generate_word_cloud_image(username)

        # Generate image 3 - Likes performance
        generate_likes_performance_image(username, likes_performance)

        # Generate image 4 - Sentiment analysis
        generate_sentiment_analysis_image(username, sentiment)

        return True

    else:
        return False
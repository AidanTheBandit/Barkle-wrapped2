import requests
import json
import time
from threading import Thread
from PIL import Image, ImageFont, ImageDraw
from wordcloud import WordCloud, STOPWORDS
from textblob import TextBlob
import numpy as np
import pandas as pd
import os
import io
from datetime import datetime, timedelta
import re
from collections import Counter
import websocket
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Barkle API endpoint
BARKLE_API_URL = "https://barkle.chat/api"
BARKLE_WS_URL = "wss://barkle.chat/streaming"

# Your Barkle API token
BARKLE_TOKEN = "your_barkle_token_here"

# Headers for API requests
headers = {
    "Authorization": f"Bearer {BARKLE_TOKEN}"
}

# Bot username (for mention filtering)
BOT_USERNAME = "your_bot_username_here"

# Global variables for styling
BACKGROUND_COLOR = (80, 54, 89)  # Purple background
TEXT_COLOR = (228, 179, 143)  # Light peach for main text
HIGHLIGHT_COLOR = (192, 222, 106)  # Light green for highlighted numbers

global_font = {
    "title": ImageFont.truetype("fonts/theboldfont.ttf", 70),
    "text": ImageFont.truetype("fonts/coolvetica-rg.otf", 60),
    "number": ImageFont.truetype("fonts/theboldfont.ttf", 100),
    "watermark": ImageFont.truetype("fonts/theboldfont.ttf", 40)
}

global_font_colour = {
    "title": TEXT_COLOR,
    "text": (179, 145, 143),
    "number": HIGHLIGHT_COLOR,
    "watermark": TEXT_COLOR
}

global_text_pos = {
    "x": 100,
    "y": 100,
    "spacer": 100
}

# Watermark text
barkle_wrapped_watermark = ["@BarkleWrapped"]

# Get user information
def getUserInfo(username):
    endpoint = f"{BARKLE_API_URL}/users/show"
    data = {
        "username": username
    }
    response = requests.post(endpoint, headers=headers, json=data)
    return response.json()

# Get all user's barks for the current year
def getUserYearlyBarks(user_id):
    endpoint = f"{BARKLE_API_URL}/users/notes"
    all_barks = []
    
    current_year = datetime.now().year
    start_date = datetime(current_year, 1, 1).isoformat() + "Z"
    
    until_id = None
    while True:
        data = {
            "userId": user_id,
            "limit": 100,
            "sinceDate": start_date
        }
        if until_id:
            data["untilId"] = until_id
        
        response = requests.post(endpoint, headers=headers, json=data)
        barks = response.json()
        
        if not barks:
            break
        
        all_barks.extend(barks)
        until_id = barks[-1]["id"]
        
        if datetime.fromisoformat(barks[-1]["createdAt"].replace("Z", "")).year < current_year:
            break
        
        time.sleep(1)
    
    return all_barks

# Store user barks in file
def storeUserBarks(username, user_barks):
    file_path = f'user_barks/{username}_yearly.txt'

    if user_barks and len(user_barks) > 0:
        with open(file_path, 'w', encoding='utf-8') as file:
            for bark in user_barks:
                file.write(cleanBark(bark['text']) + '\n')
        return True
    else:
        return False

# Clean bark text
def cleanBark(bark):
    bark = re.sub(r'http\S+', '', bark)
    bark = re.sub(r'@\w+', '', bark)
    bark = re.sub(r'#\w+', '', bark)
    bark = re.sub(r'[^a-zA-Z\s]', '', bark)
    return bark.lower().strip()

# Convert barks to DataFrame
def barksToDataFrame(barks):
    df = pd.DataFrame(barks)
    df['reactionCount'] = df['reactions'].apply(lambda x: sum(x.values()) if x else 0)
    df['created_at'] = pd.to_datetime(df['createdAt'])
    return df

# Sentiment analysis
def analyse_sentiment(text):
    analysis = TextBlob(cleanBark(text))
    return analysis.sentiment.polarity

# Generate image 1: Highest metrics
def generate_highest_metrics_image(username, most_reactions, most_rebarks, most_replies):
    img = Image.new('RGB', (1000, 1000), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    font = global_font
    font_colour = global_font_colour
    x_pos, y_pos, spacer = global_text_pos.values()

    if most_reactions > 1000:
        popularity_txt = "You're Popular!"
    elif most_reactions > 500:
        popularity_txt = "You're Growing!"
    elif most_reactions > 100:
        popularity_txt = "You're Doing OK!"
    elif most_reactions > 10:
        popularity_txt = "You're Doing Meh."
    else:
        popularity_txt = "You're not popular :("
    
    title_text = [f"{username},", popularity_txt]
    metrics_text = ["Most Reactions", "Most Rebarks", "Most Replies"]
    metrics_values = [str(most_reactions), str(most_rebarks), str(most_replies)]

    draw.text((x_pos, y_pos), title_text[0], font_colour["title"], font=font["title"])
    draw.text((x_pos, y_pos + spacer*1.1), title_text[1], font_colour["title"], font=font["title"])

    for i, (text, value) in enumerate(zip(metrics_text, metrics_values)):
        draw.text((x_pos, y_pos + spacer*(3 + i*1.5)), text, font_colour["text"], font=font["text"])
        num_width = font["number"].getsize(value)[0]
        draw.text((1000 - x_pos - num_width, y_pos + spacer*(3 + i*1.5)), value, font_colour["number"], font=font["number"])

    draw.text((650, 940), barkle_wrapped_watermark[0], font_colour["watermark"], font=font["watermark"])

    return img

# Generate image 2: Word cloud
def generate_word_cloud_image(username, text):
    mask = np.array(Image.open('img/masks/barkle_logo_1000x1000.png'))
    stopwords = set(STOPWORDS)
    stopwords.add('gt')

    wordcloud = WordCloud(
        width=900, height=900,
        background_color=None,
        mode="RGBA",
        mask=mask,
        stopwords=stopwords,
        font_path='fonts/SFProDisplay-Light.ttf',
        color_func=lambda *args, **kwargs: (199, 219, 115)
    ).generate(text)

    img = Image.new('RGB', (1000, 1000), color=BACKGROUND_COLOR)
    cloud_image = wordcloud.to_image()
    img.paste(cloud_image, (50, 50), cloud_image)

    draw = ImageDraw.Draw(img)
    draw.text((75, 75), "WHAT YOU'RE BARKING.", global_font_colour["title"], font=global_font["title"])
    draw.text((650, 940), barkle_wrapped_watermark[0], global_font_colour["watermark"], font=global_font["watermark"])

    return img

# Generate image 3: Reaction performance
def generate_reaction_performance_image(username, reaction_performance):
    img = Image.new('RGB', (1000, 1000), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    font = global_font
    font_colour = global_font_colour
    x_pos, y_pos, spacer = global_text_pos.values()

    draw.text((x_pos, y_pos), "GET ANY BIG BARKS?", font_colour["title"], font=font["title"])

    thresholds = [100, 500, 1000, 10000]
    for i, threshold in enumerate(thresholds):
        value = str(reaction_performance[threshold])
        text = f"> {threshold} reactions."
        
        draw.text((x_pos, y_pos + spacer*(2 + i*1.5)), value, font_colour["number"], font=font["number"])
        txt_width = font["text"].getsize(text)[0]
        draw.text((1000 - x_pos - txt_width, y_pos + spacer*(2 + i*1.5)), text, font_colour["text"], font=font["text"])
        
        value_width = font["number"].getsize(value)[0]
        draw.text((x_pos + value_width + 50, y_pos + spacer*(2 + i*1.5) + 15), "barks", font_colour["text"], font=font["text"])

    draw.text((650, 940), barkle_wrapped_watermark[0], font_colour["watermark"], font=font["watermark"])

    return img

# Generate image 4: Sentiment analysis
def generate_sentiment_analysis_image(username, sentiment):
    img = Image.new('RGB', (1000, 1000), color=BACKGROUND_COLOR)
    draw = ImageDraw.Draw(img)

    font = global_font
    font_colour = global_font_colour
    x_pos, y_pos, spacer = global_text_pos.values()

    draw.text((x_pos, y_pos), "HOW WERE YOU FEELING?", font_colour["title"], font=font["title"])
    draw.text((x_pos, y_pos + spacer*1.1), "HAPPY OR SAD?", font_colour["title"], font=font["title"])

    draw.text((x_pos, y_pos + spacer*3), "Emotionally your barks", font_colour["text"], font=font["text"])
    draw.text((x_pos, y_pos + spacer*4.25), "scored", font_colour["text"], font=font["text"])
    
    score_width = font["text"].getsize("scored")[0]
    draw.text((x_pos + score_width + 25, y_pos + spacer*4.2), f"{sentiment}", font_colour["number"], font=font["number"])
    
    sentiment_value = float(sentiment)
    if sentiment_value > 10:
        mood = "SUPER HAPPY!"
        emoji = Image.open("img/emojis/grinning-face-with-sweat_1f605.png")
    elif sentiment_value > 5:
        mood = "HAPPY!"
        emoji = Image.open("img/emojis/beaming-face-with-smiling-eyes_1f601.png")
    elif sentiment_value > 0:
        mood = "KINDA HAPPY..."
        emoji = Image.open("img/emojis/emoji-upside-down-face_1f643.png")
    elif sentiment_value > -5:
        mood = "SAD!"
        emoji = Image.open("img/emojis/face-with-head-bandage_1f915.png")
    else:
        mood = "DOWN BAD!"
        emoji = Image.open("img/emojis/sleepy-face_1f62a.png")

    draw.text((x_pos, y_pos + spacer*5.5), "meaning you", font_colour["text"], font=font["text"])
    draw.text((x_pos, y_pos + spacer*6.5), "were...", font_colour["text"], font=font["text"])
    draw.text((x_pos + 200, y_pos + spacer*6.5), mood, font_colour["number"], font=font["number"])

    emoji = emoji.resize((int(emoji.width/2), int(emoji.height/2)))
    img.paste(emoji, (x_pos, int(y_pos + spacer*7.5)), emoji)

    draw.text((650, 940), barkle_wrapped_watermark[0], font_colour["watermark"], font=font["watermark"])

    return img

# Upload image to Barkle drive
def upload_image_to_drive(image, filename):
    endpoint = f"{BARKLE_API_URL}/drive/files/create"
    
    img_byte_arr = io.BytesIO()
    image.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()

    files = {
        'file': (filename, img_byte_arr, 'image/png')
    }
    data = {
        'force': 'true'
    }
    
    response = requests.post(endpoint, headers=headers, files=files, data=data)
    return response.json()['id']

# Main function to generate Barkle Wrapped
def generate_barkle_wrapped(username):
    user = getUserInfo(username)
    user_barks = getUserYearlyBarks(user['id'])

    if storeUserBarks(username, user_barks):
        df = barksToDataFrame(user_barks)
        df['sentiment'] = np.array([analyse_sentiment(bark) for bark in df['text']])

        sentiment = np.mean(df['sentiment']) * 100
        sentiment = float("{0:.2f}".format(sentiment))

        most_reactions = np.max(df['reactionCount'])
        most_rebarks = np.max(df['renoteCount'])
        most_replies = np.max(df['repliesCount'])

        reaction_performance = {
            100: len(df[df['reactionCount'] > 100]),
            500: len(df[df['reactionCount'] > 500]),
            1000: len(df[df['reactionCount'] > 1000]),
            10000: len(df[df['reactionCount'] > 10000])
        }

        all_text = ' '.join(df['text'])

        images = [
            generate_highest_metrics_image(username, most_reactions, most_rebarks, most_replies),
            generate_word_cloud_image(username, all_text),
            generate_reaction_performance_image(username, reaction_performance),
            generate_sentiment_analysis_image(username, sentiment)
        ]

        drive_ids = []
        for i, image in enumerate(images):
            filename = f"{username}_barkle_wrapped_{i+1}.png"
            drive_id = upload_image_to_drive(image, filename)
            drive_ids.append(drive_id)

        return drive_ids
    else:
        return None

# Function to reply to mention with generated images
def reply_to_mention(bark_id, username, drive_ids):
    endpoint = f"{BARKLE_API_URL}/notes/create"
    data = {
        "replyId": bark_id,
        "text": f"Hey @{username}, here's your Barkle Wrapped for the year! Check out these four images summarizing your year on Barkle.",
        "fileIds": drive_ids
    }
    response = requests.post(endpoint, headers=headers, json=data)
    if response.status_code == 200:
        logger.info(f"Successfully replied to {username} with Barkle Wrapped")
    else:
        logger.error(f"Failed to reply to {username}. Status code: {response.status_code}")

# WebSocket connection handler
def on_message(ws, message):
    data = json.loads(message)
    if data['type'] == 'mention':
        note = data['body']
        if note['user']['username'] != BOT_USERNAME:  # Avoid self-mentions
            text = note['text'].lower()
            if 'wrapped' in text and f'@{BOT_USERNAME.lower()}' in text:
                logger.info(f"Received wrapped request from @{note['user']['username']}")
                username = note['user']['username']
                drive_ids = generate_barkle_wrapped(username)
                if drive_ids:
                    reply_to_mention(note['id'], username, drive_ids)

def on_error(ws, error):
    logger.error(f"WebSocket error: {error}")

def on_close(ws, close_status_code, close_msg):
    logger.info("WebSocket connection closed")

def on_open(ws):
    logger.info("WebSocket connection opened")
    auth_message = json.dumps({
        "type": "connect",
        "body": {
            "channel": "main",
            "id": "1"
        }
    })
    ws.send(auth_message)

# Main function to run the bot
def run_bot():
    websocket.enableTrace(True)
    ws = websocket.WebSocketApp(BARKLE_WS_URL,
                                on_open=on_open,
                                on_message=on_message,
                                on_error=on_error,
                                on_close=on_close)

    while True:
        try:
            ws.run_forever()
        except Exception as e:
            logger.error(f"WebSocket connection failed: {e}")
            logger.info("Attempting to reconnect in 10 seconds...")
            time.sleep(10)

if __name__ == "__main__":
    run_bot()
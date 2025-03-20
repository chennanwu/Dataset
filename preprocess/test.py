emotion_to_label = {
    'Happy': 0,
    'Sad': 1,
    'Disgust': 2,
    'Fear': 3,
    'Surprise': 4,
    'Anger': 5,
    'Neutral': 6
}

text = "Happy Neutral Disgust Sad Anger Anger Sad Disgust Neutral Happy Happy Neutral Disgust Sad Anger Anger Sad Disgust Neutral Happy Anger Sad Fear Neutral Surprise Surprise Neutral Fear Sad Anger Anger Sad Fear Neutral Surprise Surprise Neutral Fear Sad Anger Happy Surprise Disgust Fear Anger Anger Fear Disgust Surprise Happy Happy Surprise Disgust Fear Anger Anger Fear Disgust Surprise Happy Disgust Sad Fear Surprise Happy Happy Surprise Fear Sad Disgust Disgust Sad Fear Surprise Happy Happy Surprise Fear Sad Disgust"

labels = [emotion_to_label[emotion] for emotion in text.split()]
print(', '.join(map(str, labels)))
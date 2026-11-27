from emotion_detector import emotion_scores, classify

def label(**b):
    return classify(emotion_scores(b))

assert label() == "Neutral"
assert label(mouthSmileLeft=0.8, mouthSmileRight=0.8) == "Happy"
assert label(mouthFrownLeft=0.3, mouthFrownRight=0.3, browInnerUp=0.5) == "Sad"
assert label(mouthShrugLower=0.45, mouthFrownLeft=0.08, mouthFrownRight=0.08) == "Sad"  # pout, brows relaxed
assert label(mouthShrugLower=0.1, mouthFrownLeft=0.01, mouthFrownRight=0.01, browInnerUp=0.08) == "Neutral"  # resting face
assert label(browDownLeft=0.6, browDownRight=0.6) == "Angry"
assert label(jawOpen=0.7, browInnerUp=0.6, browOuterUpLeft=0.6, browOuterUpRight=0.6) == "Surprised"
print("all emotion rules OK")

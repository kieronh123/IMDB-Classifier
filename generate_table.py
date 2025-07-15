sample_predictions = [
    ("Absolutely loved this movie!", "POSITIVE", 0.98),
    ("Terrible acting and a boring plot.", "NEGATIVE", 0.96),
    ("It was okay, not the best I've seen.", "POSITIVE", 0.67),
]

print("| Review Text                             | Prediction | Confidence |")
print("|-----------------------------------------|------------|------------|")
for review, label, confidence in sample_predictions:
    review_wrapped = "<br>".join(review[i:i+30] for i in range(0, len(review), 30))
    print(f"| {review_wrapped:<37} | {label:<10} | {confidence:.2f}       |")

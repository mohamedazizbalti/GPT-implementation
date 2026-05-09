from datasets import load_dataset


def create_classification_dataset(corpus: str, num_samples: int = 500):
    labels = {
        "animal":    ["dog", "cat", "bird", "rabbit", "bear", "duck", "fish", "horse"],
        "magic":     ["magic", "wand", "spell", "witch", "wizard", "fairy", "dragon"],
        "emotion":   ["sad", "happy", "cry", "angry", "scared", "afraid", "lonely"],
        "adventure": ["forest", "mountain", "treasure", "journey", "explore", "cave"],
        "friendship": ["friend", "together", "share", "help", "kind", "care", "play"],
    }
    stories = [s.strip() for s in corpus.split("\n\n") if len(s.strip()) > 50]
    dataset = []
    label_to_idx = {label: idx for idx, label in enumerate(labels.keys())}
    for story in stories[:num_samples]:
        story_lower = story.lower()
        scores = {label: 0 for label in labels}
        for label, keywords in labels.items():
            for keyword in keywords:
                if keyword in story_lower:
                    scores[label] += 1
        best_label = max(scores, key=lambda l: scores[l])
        if scores[best_label] > 0:
            dataset.append({
                "text": story,
                "label": label_to_idx[best_label],
                "label_name": best_label
            })
    return dataset, label_to_idx


def load_ag_news(num_samples: int = 1000):
    dataset = load_dataset("ag_news")

    label_names = dataset["train"].features["label"].names
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    train_data = [
        {
            "text": item["text"],
            "label": item["label"],
            "label_name": label_names[item["label"]]
        }
        for item in dataset["train"]
    ]

    return train_data[:num_samples], label_to_idx


def load_emotion_dataset():
    dataset = load_dataset("emotion")

    label_names = dataset["train"].features["label"].names
    label_to_idx = {name: i for i, name in enumerate(label_names)}

    train_data = [
        {
            "text": item["text"],
            "label": item["label"],
            "label_name": label_names[item["label"]]
        }
        for item in dataset["train"]
    ]

    return train_data, label_to_idx


if __name__ == "__main__":
    data, mapping = load_ag_news()
    print("Len : ", len(data))

import numpy as np
import torchvision
import torchvision.transforms as transforms
from collections import Counter
import streamlit as st

# Başlık
st.title("CIFAR-10 k-NN Sınıflandırıcı")

# Kullanıcıdan seçimler
metric = st.selectbox("Mesafe metriğini seçiniz:", ["L1", "L2"])
k = st.slider("k değerini seçiniz:", 1, 15, 3)

# CIFAR-10 datasetini yükle
transform = transforms.Compose([transforms.ToTensor()])
trainset = torchvision.datasets.CIFAR10(root='./data', train=True,
                                        download=True, transform=transform)
testset = torchvision.datasets.CIFAR10(root='./data', train=False,
                                       download=True, transform=transform)

train_data = trainset.data[:1000]
train_labels = np.array(trainset.targets[:1000])
test_data = testset.data[:10]
test_labels = np.array(testset.targets[:10])

import pandas as pd

if st.button("Çalıştır"):
    results = []
    correct = 0
    for idx, test_img in enumerate(test_data):
        test_vec = test_img.flatten()
        train_vecs = train_data.reshape(len(train_data), -1)

        if metric == "L1":
            distances = np.sum(np.abs(train_vecs - test_vec), axis=1)
        else:  # L2
            distances = np.sqrt(np.sum((train_vecs - test_vec) ** 2, axis=1))

        nearest_indices = np.argsort(distances)[:k]
        nearest_labels = train_labels[nearest_indices]
        prediction = Counter(nearest_labels).most_common(1)[0][0]

        is_correct = (prediction == test_labels[idx])
        if is_correct:
            correct += 1

        results.append({
            "Test örneği": idx,
            "Tahmin": prediction,
            "Gerçek": test_labels[idx],
            "Durum": "✅ Doğru" if is_correct else "❌ Yanlış"
        })

    df = pd.DataFrame(results)

    st.write("Sonuçlar:")
    st.dataframe(df, use_container_width=True)

    st.success(f"Doğruluk: {correct}/{len(test_data)} = {correct/len(test_data)*100:.2f}%")


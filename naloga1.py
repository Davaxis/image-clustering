import numpy as np
import torch
from PIL import Image
from torchvision import transforms
import sys
import os
import random
from matplotlib import pyplot as plt
import math

images = {}

def read_data(dir):
    model = torch.hub.load('pytorch/vision:v0.10.0', 'squeezenet1_1', pretrained=True)
    model.eval()

    for img in os.listdir(dir):
        images[img] = Image.open(dir + '/' + img).convert('RGB')

    vectors = {}
    for key, img in images.items():
        preprocess = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
        input_tensor = preprocess(img)
        input_batch = input_tensor.unsqueeze(0)

        if torch.cuda.is_available():
            input_batch = input_batch.to('cuda')
            model.to('cuda')

        with torch.no_grad():
            output = model(input_batch)

        vectors[key] = output[0]

    return vectors

def cosine_dist(d1, d2):
    dist = np.dot(d1, d2.T)
    dist /= np.linalg.norm(d1) * np.linalg.norm(d2)
    return 1 - dist


def k_medoids(data, medoids):
    prev_cost = None
    clusters, cost = calc_costs(data, medoids)

    while prev_cost is None or cost < prev_cost:
        medoids = get_new_centers(data, clusters, medoids)
        prev_cost = cost
        clusters, cost = calc_costs(data, medoids)
    
    return clusters


def silhouette(el, clusters, data):
    a = notranja_razdalja(el, clusters, data)
    b = zunanja_razdalja(el, clusters, data)
    return (b-a)/max(a,b)


def silhouette_average(data, clusters):
    dist = []
    for cluster in clusters:
        d = silhouette(cluster[0], clusters, data)
        dist.append(d)

    return np.mean(dist)


def notranja_razdalja(el, clusters, data):
    cluster = [x for x in clusters if el in x][0]

    if len(cluster) == 0:
        return 0
    return np.mean([cosine_dist(data[el], data[othr]) for othr in cluster if othr != el])
    
def zunanja_razdalja(el, clusters, data):
    foreign = [x for x in clusters if el not in x]

    return min([sum(cosine_dist(data[el], data[othr]) for othr in cluster)/len(cluster) for cluster in foreign])

def calc_costs(data, medoids):
    clusters = []
    for i in medoids:
        clusters.append([])

    cost_sum = 0
    for k, v in data.items():
        best_index = 0
        best_dist = None
        for i, medoid in enumerate(medoids):
            dist = cosine_dist(data[medoid], v)
            if best_dist is None or best_dist > dist:
                best_index = i
                best_dist = dist
        clusters[best_index].append(k)
        cost_sum += best_dist

    return clusters, cost_sum

def get_new_centers(data, clusters, medoids):
    best_point = ''
    best_dist = None

    for i, cluster in enumerate(clusters):
        for point in cluster:
            dist = notranja_razdalja(point, clusters, data)
            if best_dist is None or dist < best_dist:
                best_dist = dist
                best_point = point
        if best_dist is not None:
            medoids[i] = best_point
        best_point = ''
        best_dist = None
    
    return medoids

def analiza(clusters, data):
    cluster_no = 0
    for cluster in clusters:
        scores = [silhouette(x, clusters, data) for x in cluster]
        fig, axs = plt.subplots(2, math.ceil(len(cluster)/2))
        fig.set_size_inches(12, 3)
        fig.tight_layout()
        cl = cluster.copy()
        for i in range(math.ceil(len(cluster)/2)):
            index = np.argmax(scores)
            img = cl.pop(index)
            score = scores.pop(index)

            axs[0, i].imshow(images[img])
            axs[0, i].set_title(str(round(score, 4)))

            if len(cl) > 0:
                index = np.argmax(scores)
                img = cl.pop(index)
                score = scores.pop(index)

                axs[1, i].imshow(images[img])
                axs[1, i].set_title(str(round(score, 4)))
        plt.savefig('cluster_'+str(cluster_no)+'.png', dpi=400)
        cluster_no += 1

if __name__ == "__main__":
    k = 3
    path = 'pictures'
    if len(sys.argv) == 3:
        k = int(sys.argv[1])
        path = sys.argv[2]

    data = read_data(path)
    medoids = []
    for i in range(100):
        medoids.append(random.sample(list(data.keys()), k=k))


    best_clusters = []
    best_sil = None
    for sample in medoids:
        clusters = k_medoids(data, sample)
        sil = silhouette_average(data, clusters)
        if best_sil is None or sil > best_sil:
            best_clusters = clusters
            best_sil = sil

    analiza(best_clusters, data)

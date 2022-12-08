import math
import csv
import random
import numpy as np
import matplotlib.pyplot as plt


def load_data(filepath):
    data = []
    with open(filepath, newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        for line, i in zip(reader, range(20)):
            line.pop('Generation')
            line.pop('Legendary')
            for value in line:
                if (value != 'Name') & (value != 'Type 1') & (value != 'Type 2'):
                    line[value] = int(line[value])
            data.append(line)
    return data


def calculate_x_y(stats):
    return stats['Attack'] + stats['Sp. Atk'] + stats['Speed'],\
           stats['HP'] + stats['Sp. Def'] + stats['Defense']


def hac(dataset):
    for i in range(len(dataset)):
        if (math.inf in dataset[i]) | (math.nan in dataset[i]):
            dataset.pop(i)
    m = len(dataset)
    pairs = []
    distance = []
    for i in range(len(dataset)):
        for j in range(len(dataset))[i+1:]:
            pairs.append([dataset[i], dataset[j]])
    for pair in pairs:
        distance.append(math.sqrt(math.fabs(math.pow(pair[0][0] - pair[1][0], 2) +
                                            math.pow(pair[0][1] - pair[1][1], 2))))
    linkage_table = np.empty([m-1, 4])
    clusters = []
    for point in dataset:
        clusters.append([point])
    for i in range(m-1):
        for j in range(len(distance)):
            closest_distance_index = distance.index(min(distance))
            point1 = pairs[closest_distance_index][0]
            point2 = pairs[closest_distance_index][1]
            for cluster in clusters:
                if point1 in cluster:
                    cluster1 = cluster
                if point2 in cluster:
                    cluster2 = cluster
            if cluster1 == cluster2:
                distance.pop(closest_distance_index)
                pairs.pop(closest_distance_index)
            else:
                break
        smaller_index = min(clusters.index(cluster1), clusters.index(cluster2))
        larger_index = max(clusters.index(cluster1), clusters.index(cluster2))
        linkage_table[i] = [smaller_index, larger_index, distance[closest_distance_index],
                            len(cluster1) + len(cluster2)]
        clusters.append(cluster1+cluster2)
        cluster1.clear()
        cluster2.clear()
        distance.pop(closest_distance_index)
        pairs.pop(closest_distance_index)
    return linkage_table


def random_x_y(m):
    random_value = []
    for i in range(m):
        random_value.append([random.randint(1, 359), random.randint(1, 359)])
    return random_value


def imshow_hac(dataset):
    for i in range(len(dataset)):
        if (math.inf in dataset[i]) | (math.nan in dataset[i]):
            dataset.pop(i)
    for i in range(len(dataset)):
        plt.scatter(dataset[i][0], dataset[i][1])
    process = hac(dataset)
    pairs = []
    distance = []
    for i in range(len(dataset)):
        for j in range(len(dataset))[i+1:]:
            pairs.append([dataset[i], dataset[j]])
    for pair in pairs:
        distance.append(math.sqrt(math.fabs(math.pow(pair[0][0] - pair[1][0], 2) +
                                            math.pow(pair[0][1] - pair[1][1], 2))))
        print(pair)
    for step in process:
        points_index = distance.index(step[2])
        points = pairs[points_index]
        x = [points[0][0], points[1][0]]
        y = [points[0][1], points[1][1]]
        pairs.pop(points_index)
        distance.pop(points_index)
        plt.plot(x, y)
        plt.pause(0.1)
    plt.show()


if __name__ == "__main__":
    # pokemons_x_y = []
    # pokemons = load_data("Pokemon.csv")
    # print(pokemons)
    # for row in pokemons:
    #     pokemons_x_y.append(calculate_x_y(row))
    # # print(hac(pokemons_x_y))
    # imshow_hac(pokemons_x_y)



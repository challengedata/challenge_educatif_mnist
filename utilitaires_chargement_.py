# Import des librairies utilisées dans le notebook
#import basthon
import requests
import numpy as np
import matplotlib.pyplot as plt
import pickle
from zipfile import ZipFile
from io import BytesIO, StringIO
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches
from scipy.spatial import Voronoi, voronoi_plot_2d


# Téléchargement et extraction des inputs contenus dans l'archive zip
inputs_zip_url = "https://raw.githubusercontent.com/challengedata/challenge_educatif_mnist/main/inputs.zip"
inputs_zip = requests.get(inputs_zip_url)
zf = ZipFile(BytesIO(inputs_zip.content))
zf.extractall()
zf.close()


# Téléchargement des outputs d'entraînement de MNIST-10 contenus dans le fichier y_train_10.csv
output_train_url = "https://raw.githubusercontent.com/challengedata/challenge_educatif_mnist/main/y_train_10.csv"
output_train = requests.get(output_train_url)

# Création des variables d'inputs, outputs et indices pour les datasets MNIST-2, MNIST-4 et MNIST-10

# MNIST-10

# Inputs and indices
with open('mnist_10_x_train.pickle', 'rb') as f:
    ID_train_10, x_train_10 = pickle.load(f).values()

with open('mnist_10_x_test.pickle', 'rb') as f:
    ID_test_10, x_test_10 = pickle.load(f).values()

# Outputs
_, y_train_10 = [np.loadtxt(StringIO(output_train.content.decode('utf-8')),
                                dtype=int, delimiter=',')[:,k] for k in [0,1]]

# Les challenges MNIST-2 et MNIST-4 sont des sous-ensembles de MNIST-10.
# MNIST-4

# Inputs
with open('mnist_4_x_train.pickle', 'rb') as f:
    ID_train_4, x_train_4 = pickle.load(f).values()

with open('mnist_4_x_test.pickle', 'rb') as f:
    ID_test_4, x_test_4 = pickle.load(f).values()

# Outputs
y_train_4 = y_train_10[np.isin(y_train_10, [0,1,4,8])]

# MNIST-2

# Inputs
with open('mnist_2_x_train.pickle', 'rb') as f:
    ID_train_2, x_train_2 = pickle.load(f).values()

with open('mnist_2_x_test.pickle', 'rb') as f:
    ID_test_2, x_test_2 = pickle.load(f).values()

# Outputs
y_train_2 = y_train_10[np.isin(y_train_10, [0,1])]

# Example image
image_url = "https://raw.githubusercontent.com/challengedata/challenge_educatif_mnist/main/x.npy"
x = np.load(BytesIO(requests.get(image_url).content))

# Pour MNIST-4 :
chiffres = [0,1,4,8]
x_train_par_population = [x_train_4[y_train_4==k] for k in chiffres]

# Affichage d'une image
def affichage(image):
    plt.imshow(image, cmap='gray')
    plt.show()
    plt.close()

# Affichage de dix images
def affichage_dix(images):
    fig, ax = plt.subplots(1, 10, figsize=(10,1))
    for k in range(10):
        ax[k].imshow(images[k], cmap='gray')
        ax[k].set_xticks([])
        ax[k].set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    plt.show()
    plt.close()

# Affichage de vingt images
def affichage_vingt(images):
    fig, ax = plt.subplots(2, 10, figsize=(10,2))
    for k in range(20):
        ax[k//10,k%10].imshow(images[k], cmap='gray')
        ax[k//10,k%10].set_xticks([])
        ax[k//10,k%10].set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    plt.show()
    plt.close()

# Affichage de trente images
def affichage_trente(images):
    fig, ax = plt.subplots(3, 10, figsize=(10,3))
    for k in range(30):
        ax[k//10,k%10].imshow(images[k], cmap='gray')
        ax[k//10,k%10].set_xticks([])
        ax[k//10,k%10].set_yticks([])
    plt.subplots_adjust(left=0, right=1, top=1, bottom=0, wspace=0.05, hspace=0.05)
    plt.show()
    plt.close()

# Sauver le .csv

def sauver_et_telecharger_mnist_2(y_est_test, nom_du_fichier):
    np.savetxt(nom_du_fichier, np.stack([ID_test_2, y_est_test], axis=-1), fmt='%d', delimiter=',', header='ID,targets')
    basthon.download(nom_du_fichier)

def sauver_et_telecharger_mnist_4(y_est_test, nom_du_fichier):
    np.savetxt(nom_du_fichier, np.stack([ID_test_4, y_est_test], axis=-1), fmt='%d', delimiter=',', header='ID,targets')
    basthon.download(nom_du_fichier)

def sauver_et_telecharger_mnist_10(y_est_test, nom_du_fichier):
    np.savetxt(nom_du_fichier, np.stack([ID_test_10, y_est_test], axis=-1), fmt='%d', delimiter=',', header='ID,targets')
    basthon.download(nom_du_fichier)

# Visualiser les histogrammes
def visualiser_histogrammes_mnist_2(c_train):
    digits = [0,1]
    nb_digits = 2

    c_train_par_population = [np.array(c_train)[y_train_2==k] for k in digits]

    # Visualisation des histogrammes
    for k in range(nb_digits):
        plt.hist(c_train_par_population[k], bins=60, alpha=0.7, label=k)

    plt.gca().set_xlim(xmin=0)
    plt.gca().set_title("Histogrammes de la caractéristique")
    plt.legend(loc='upper right')
    plt.show()
    plt.close()

# Visualiser les histogrammes
def visualiser_histogrammes_mnist_4(c_train_par_population):
    digits = [0,1,4,8]
    nb_digits = 4

    # Visualisation des histogrammes
    for k in range(nb_digits):
        plt.hist(np.array(c_train_par_population[k]), bins=60, alpha=0.7, label=digits[k])

    plt.gca().set_xlim(xmin=0)
    plt.gca().set_title("Histogrammes de la caractéristique")
    plt.legend(loc='upper right')
    plt.show()
    plt.close()

# Visualiser les histogrammes 2D
def visualiser_histogrammes_2d_mnist_4(c_train):

    c_train_par_population = par_population(c_train)

    digits = [0,1,4,8]
    nb_digits = 4

    # Moyennes
    N = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N[i] for i in range(nb_digits)]

    # Quatre premières couleurs par défaut de Matplotlib
    colors = {0:'C0', 1:'C1', 4:'C2', 8:'C3'}
    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", colors[i]], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))
    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(min(0, mins_[0]),maxs_[0],100), np.linspace(min(0, mins_[1]),maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor=colors[list(colors.keys())[i]])

    patches = [mpatches.Patch(color=colors[i], label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()


# Visualiser les histogrammes 2D avec les domaines de Voronoi
def visualiser_histogrammes_2d_mnist_4_vor(c_train):
    c_train_par_population = par_population(c_train)

    digits = [0,1,4,8]
    nb_digits = 4

    # Moyennes
    N = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N[i] for i in range(nb_digits)]
    theta = [np.mean(c_train_par_population[i], axis = 0) for i in range(4)]

    # Quatre premières couleurs par défaut de Matplotlib
    colors = {0:'C0', 1:'C1', 4:'C2', 8:'C3'}
    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", colors[i]], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)

    fig, ax = plt.subplots(figsize=(10,10))

    # Voronoi
    vor = Voronoi(theta)
    fig = voronoi_plot_2d(vor, ax=ax, show_points=False)

    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(min(0, mins_[0]),maxs_[0],100), np.linspace(min(0, mins_[1]),maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor=colors[list(colors.keys())[i]])

    patches = [mpatches.Patch(color=colors[i], label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()

# Visualiser les histogrammes 2d avec les argmax des 4 distributions
def visualiser_histogrammes_2d_mnist_4_max(c_train):

    c_train_par_population = par_population(c_train)

    digits = [0,1,4,8]
    nb_digits = 4

    # Moyennes
    N = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N[i] for i in range(nb_digits)]

    max_list = max_hist_2d_mnist4(c_train)

    # Quatre premières couleurs par défaut de Matplotlib
    colors = {0:'C0', 1:'C1', 4:'C2', 8:'C3'}
    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", colors[i]], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))
    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(0,maxs_[0],100), np.linspace(0,maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(max_list[i][0], max_list[i][1], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor=colors[list(colors.keys())[i]])

    patches = [mpatches.Patch(color=colors[i], label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()


# Visualiser les histogrammes 2D pour MNIST-10
def visualiser_histogrammes_2d_mnist_10(c_train):
    c_train_par_population = par_population_10(c_train)
    digits = np.arange(10).tolist()
    nb_digits = 10

    # Moyennes
    N_ = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N_[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N_[i] for i in range(nb_digits)]

    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", 'C'+str(i)], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))
    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(min(0, mins_[0]),maxs_[0],100), np.linspace(min(0, mins_[1]),maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor='C'+str(i))

    patches = [mpatches.Patch(color='C'+str(i), label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()

# Visualiser les 10 hist 2D de MNIST-10 avec les domaines de voronoi
def visualiser_histogrammes_2d_mnist_10_vor(c_train):
    c_train_par_population = par_population_10(c_train)
    digits = np.arange(10).tolist()
    nb_digits = 10

    # Moyennes
    N_ = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N_[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N_[i] for i in range(nb_digits)]
    theta = [np.mean(c_train_par_population[i], axis = 0) for i in range(nb_digits)]

    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", 'C'+str(i)], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))

    # Voronoi
    vor = Voronoi(theta)
    fig = voronoi_plot_2d(vor, ax=ax, show_points=False)

    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(min(0, mins_[0]),maxs_[0],100), np.linspace(min(0, mins_[1]),maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor='C'+str(i))

    patches = [mpatches.Patch(color='C'+str(i), label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()

def max_hist_2d_mnist4(c_train):
    # Trouve les coordonnées qui réalisent le max de chaque hist 2D de la caractéristique

    c_train_par_population = par_population(c_train)

    digits = [0,1,4,8]
    nb_digits = 4
    max_list = []

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    for i in range(nb_digits):
        H, xedges, yedges = np.histogram2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
            bins=[np.linspace(0,maxs_[0],60), np.linspace(0,maxs_[1],60)])
        x, y = np.argwhere(H == H.max())[0]
        max_list.append([np.average(xedges[x:x + 2]), np.average(yedges[y:y + 2])])

    return max_list


# Visualiser dans le plan dix caractéristiques 2D pour chaque population
def visualiser_caracteristiques_2d_dix(c_train_par_population):
    digits = [0,1,4,8]
    for k in range(4):
        plt.scatter(c_train_par_population[k][:10,0], c_train_par_population[k][:10,1],label=digits[k])
    plt.legend(loc='upper left')
    plt.show()
    plt.close()

# Fonction de score
def score(y_est, y_vrai):
    # Initialisation de la valeur de l'erreur
    somme = 0
    # On lance une erreur si y_est et y_vrai ne sont pas de la même taille
    if len(y_est)!=len(y_vrai) :
        raise ValueError("Les sorties comparées ne sont pas de la même taille.")
    else :   # on incrémente la somme pour toute mauvaise estimation
        for i in range(len(y_vrai)):
            if y_est[i] != y_vrai[i]:
                somme = somme + 1
    return somme / len(y_vrai)

# Moyenne
def moyenne(liste_car):
    somme = 0.
    for car in liste_car:
        somme = somme + car.astype(float)
    return somme / len(liste_car)

def par_population(liste):
    chiffres = [0,1,4,8]
    # Créer une liste de liste qui divise par population, comme par exemple pour liste = c_train
    return [np.array(liste)[y_train_4==k] for k in chiffres]

def par_population_10(liste):
    # Créer une liste de liste qui divise par population, comme par exemple pour liste = c_train
    return [np.array(liste)[y_train_10==k] for k in range(10)]

def par_population_mnist2(liste):
    chiffres = [0,1]
    # Créer une liste de liste qui divise par population, comme par exemple pour liste = c_train
    return [np.array(liste)[y_train_2==k] for k in chiffres]

def distance_carre(a,b):
    # a et b sont supposés être des points en deux dimensions contenus dans des listes de longueur deux
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

def distance_carre_gen(A, B):
    return np.sum((A-B)**2)

def classification_2d_MNIST4(c, theta):

    #c_train_moyennes_par_population = [moyenne(liste_car) for liste_car in c_train_par_population]

    # On définit d'abord les différentes estimations possibles
    chiffres = [0,1,4,8]
    # On calcule le carré des distances entre la caractéristique c et les caractéristiques moyennes
    dist = [distance_carre(c, theta_i) for theta_i in theta]
    # On extrait l'indice minimisant cette distance
    index_min = dist.index(min(dist))
    # On renvoie le chiffre correspondant
    return chiffres[index_min]

# Algorithme de classification pour les 10 catégories de chiffres
def classification_dist_moy(c, theta_):
    # On définit d'abord les différentes estimations possibles
    chiffres = np.arange(10).tolist()
    # On calcule le carré des distances entre la caractéristique c et les caractéristiques moyennes
    dist = [distance_carre_gen(np.array(c).flatten(), np.array(theta).flatten()) for theta in theta_]
    # On extrait l'indice minimisant cette distance
    index_min = dist.index(min(dist))
    # On renvoie le chiffre correspondant
    return chiffres[index_min]

def visualiser_histogrammes_2d_mnist_2(c_train):

    c_train_par_population = par_population_mnist2(c_train)

    digits = [0,1]
    nb_digits = 2

    # Moyennes
    N = [len(c_train_par_population[i][:,0]) for i in range(nb_digits)]
    M_x = [sum(c_train_par_population[i][:,0])/N[i] for i in range(nb_digits)]
    M_y = [sum(c_train_par_population[i][:,1])/N[i] for i in range(nb_digits)]

    # Quatre premières couleurs par défaut de Matplotlib
    colors = {0:'C0', 1:'C1', 4:'C2', 8:'C3'}
    # Palette de couleurs interpolant du blanc à chacune de ces couleurs, avec N=100 nuances
    cmaps = [LinearSegmentedColormap.from_list("", ["w", colors[i]], N=100) for i in digits]
    # Ajout de transparence pour la superposition des histogrammes :
    # plus la couleur est proche du blanc, plus elle est transparente
    cmaps_alpha = []
    for cmap in cmaps:
        cmap._init()
        cmap._lut[:-3,-1] = np.linspace(0, 1, cmap.N)  # la transparence va de 0 (complètement transparent) à 1 (opaque)
        cmaps_alpha += [ListedColormap(cmap._lut[:-3,:])]

    maxs_ = np.concatenate(c_train_par_population).max(axis=0)
    mins_ = np.concatenate(c_train_par_population).min(axis=0)
    fig, ax = plt.subplots(figsize=(10,10))
    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(mins_[0],maxs_[0],100), np.linspace(mins_[1],maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor=colors[list(colors.keys())[i]])

    patches = [mpatches.Patch(color=colors[i], label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()

# --------   POUR MNIST 4 STOCHASTIC DESCENT --------------
# Obtenir le score pour MNIST4 en mode ML

def caracteristique(x, w):
    N = 28*28
    return (np.sum(x*w[:-1]) + w[-1])/(N+1)

def get_c_train(w1, w2, x_train_scaled):
    c_train = []
    for x in x_train_scaled:
        c1 = caracteristique(x, w1)
        c2 = caracteristique(x, w2)
        c_train.append([c1, c2])
    return c_train

def get_x_train4_scaled(x_train_4):
    N = 28*28

    long_4 = len(x_train_4)
    x_train_4_scaled = x_train_4.reshape((long_4, N))/255

    return x_train_4_scaled

def get_score(c_train):
    c_train_par_population = par_population(c_train)
    theta = [np.mean(c_train_par_population[i], axis = 0) for i in range(4)]

    y_est_train = []
    for c in c_train:
        y_est_train.append(classification_2d_MNIST4(c, theta))

    return 100*score(y_est_train, y_train_4)

def tableau_aleatoire(long):
    return np.random.rand(long)*2-1

x_train_4_scaled = get_x_train4_scaled(x_train_4)
def faire_K_pas(K, w1_0, w2_0, taille_pas, best_score, best_c_train):
    # Essai K pas à partir de (w1_0, w2_0)
    global x_train_4_scaled

    N = 28*28

    for i in range(K):
        eps1 = (np.random.rand(N+1)*2-1)*taille_pas
        eps2 = (np.random.rand(N+1)*2-1)*taille_pas

        w1_1 = w1_0 + eps1
        w2_1 = w2_0 + eps2

        c_train1 = get_c_train(w1_1, w2_1, x_train_4_scaled)
        score_1 = get_score(c_train1)

        if score_1 < best_score:
            best_score = score_1
            best_c_train = c_train1
            w1_0 = w1_1
            w2_0 = w2_1

            print("Current score =", best_score)
            visualiser_histogrammes_2d_mnist_4_vor(best_c_train)

    return best_score, best_c_train, w1_0, w2_0

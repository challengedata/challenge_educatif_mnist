# Import des librairies utilisées dans le notebook
import basthon
import requests
import numpy as np
import matplotlib.pyplot as plt
import pickle
from zipfile import ZipFile
from io import BytesIO, StringIO
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches

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
    fig, ax = plt.subplots(figsize=(10,10))
    for i in reversed(range(nb_digits)):  # ordre inversé pour un meilleur rendu
        ax.hist2d(c_train_par_population[i][:,0], c_train_par_population[i][:,1],
                  bins=[np.linspace(0,maxs_[0],100), np.linspace(0,maxs_[1],100)], cmap=cmaps_alpha[i])

    for i in reversed(range(nb_digits)):
        ax.scatter(M_x[i], M_y[i], marker = 'o', s = 70, edgecolor='black', linewidth=1.5, facecolor=colors[list(colors.keys())[i]])

    patches = [mpatches.Patch(color=colors[i], label=i) for i in digits]
    ax.legend(handles=patches,loc='upper left')

    plt.show()
    plt.close()

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
    # Créer une liste de liste qui divise par population, comme par exemple pour liste = c_train
    return [np.array(liste)[y_train_4==k] for k in chiffres]

def distance_carre(a,b):
    # a et b sont supposés être des points en deux dimensions contenus dans des listes de longueur deux
    return (a[0]-b[0])**2 + (a[1]-b[1])**2

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

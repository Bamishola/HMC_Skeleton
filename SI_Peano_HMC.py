"""
SI_Peano_HMC.py
Segmentation non supervisée d'une image par chaîne de Markov cachée (HMC)
avec parcours de Peano (courbe de Hilbert) et algorithme EM.

Auteur : Bamishola LOKE
"""

import sys
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image
from scipy.stats import norm
from sklearn.cluster import KMeans
from skimage.filters import threshold_otsu

# Ajouter le chemin vers le dossier Peano
sys.path.append('./Peano')
from PeanoImage import Peano, getPowerOfTwo
from InvPeanoImage import PeanoInverse

sys.path.insert(0, '.')
from func import (getSteadyState, getAlpha, getBeta, getGamma,
                  getMPMClassif, getProbaMarkov, InitParam,
                  getCtilde, UpdateParameters, EM_Iter,
                  DrawCurvesParam, DrawCurvesError)

# ─────────────────────────────────────────────────────────────────────────────
# Paramètres
# ─────────────────────────────────────────────────────────────────────────────
# IMAGE_PATH = './Peano/sources/cible_64_bruit.png'
# IMAGE_PATH = './Peano/sources/image_seg.pgm' # nouvelle image  
IMAGE_PATH = './Peano/sources/image3_64.pgm' # nouvelle image 
RESULT_DIR = './results'
os.makedirs(RESULT_DIR, exist_ok=True)

K      = 2    # nombre de classes (segments)
nbIter = 30   # itérations EM

# ─────────────────────────────────────────────────────────────────────────────
# 1. Chargement et conversion de l'image en vecteur 1D (parcours de Peano)
# ─────────────────────────────────────────────────────────────────────────────
img   = Image.open(IMAGE_PATH).convert('L')
image = np.array(img, dtype=float)
L, C  = image.shape
print(f"Image chargée : {L}x{C} pixels, min={image.min():.0f}, max={image.max():.0f}")

# Conversion 2D → 1D via le parcours de Hilbert/Peano
Y = Peano(image)
N = len(Y)
print(f"Vecteur Peano : {N} échantillons")

# Vérification aller-retour (Peano puis PeanoInverse doit redonner l'image)
image_recon = PeanoInverse(Y)
assert np.allclose(image, image_recon), "Erreur : l'aller-retour Peano n'est pas exact !"
print("Vérification aller-retour Peano : OK")

# ─────────────────────────────────────────────────────────────────────────────
# 2. Initialisation des paramètres EM
# ─────────────────────────────────────────────────────────────────────────────
meanTabIter = np.zeros((nbIter, K))
varTabIter  = np.zeros((nbIter, K))
cTabIter    = np.zeros((nbIter, K, K))
tTabIter    = np.zeros((nbIter, K, K))
ITabIter    = np.zeros((nbIter, K))

MeanErrorRateTab        = np.zeros(nbIter)
MeanErrorRateTabbyClass = np.zeros((nbIter, K))

iteration = 0
meanTabIter[0], sigma0, cTabIter[0] = InitParam(K, Y)
varTabIter[0] = sigma0 ** 2
tTabIter[0], ITabIter[0] = getProbaMarkov(cTabIter[0])

print(f"\nParamètres initiaux :")
print(f"  mu    = {meanTabIter[0]}")
print(f"  sigma = {np.sqrt(varTabIter[0])}")
print(f"  t     =\n{tTabIter[0]}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. Itérations EM
# ─────────────────────────────────────────────────────────────────────────────
print(f"\nDébut des {nbIter} itérations EM...")
for iteration in range(1, nbIter):
    if iteration % 5 == 1 or iteration == nbIter-1:
        print(f"  --> itération {iteration}")
    gamma = EM_Iter(iteration, Y, meanTabIter, varTabIter,
                    cTabIter, tTabIter, ITabIter)

print("EM terminé.")
print(f"\nParamètres finaux estimés :")
print(f"  mu    = {meanTabIter[nbIter-1]}")
print(f"  sigma = {np.sqrt(varTabIter[nbIter-1])}")
print(f"  t     =\n{tTabIter[nbIter-1]}")

# ─────────────────────────────────────────────────────────────────────────────
# 4. Segmentation finale (critère MPM)
# ─────────────────────────────────────────────────────────────────────────────
alpha_f, S_f = getAlpha(Y, meanTabIter[nbIter-1], varTabIter[nbIter-1],
                         ITabIter[nbIter-1], tTabIter[nbIter-1])
beta_f       = getBeta (Y, meanTabIter[nbIter-1], varTabIter[nbIter-1],
                         ITabIter[nbIter-1], tTabIter[nbIter-1], S_f)
gamma_f      = getGamma(alpha_f, beta_f)
X_MPM        = getMPMClassif(gamma_f)

# Reconstruction de l'image segmentée (vecteur 1D → image 2D)
image_seg = PeanoInverse(X_MPM)

# ─────────────────────────────────────────────────────────────────────────────
# 5. Comparaison : Otsu et K-means
# ─────────────────────────────────────────────────────────────────────────────
# Seuillage d'Otsu
thresh      = threshold_otsu(image)
image_otsu  = (image > thresh).astype(float)

# K-Means
pixels_flat  = image.reshape(-1, 1)
kmeans       = KMeans(n_clusters=K, random_state=42, n_init=10)
kmeans.fit(pixels_flat)
image_kmeans = kmeans.labels_.reshape(L, C).astype(float)

# ─────────────────────────────────────────────────────────────────────────────
# 6. Sauvegarde des courbes de convergence EM
# ─────────────────────────────────────────────────────────────────────────────
basename   = os.path.splitext(os.path.basename(IMAGE_PATH))[0]
pathToSave = os.path.join(RESULT_DIR, basename)
DrawCurvesParam(nbIter, pathToSave, meanTabIter, varTabIter, tTabIter)

# ─────────────────────────────────────────────────────────────────────────────
# 7. Figure principale du CR :
#    image originale | image segmentée HMC | Otsu | K-Means
# ─────────────────────────────────────────────────────────────────────────────
fig, axes = plt.subplots(1, 4, figsize=(14, 4))

axes[0].imshow(image,      cmap='gray', vmin=0, vmax=255)
axes[0].set_title('Image originale',     fontsize=11)
axes[0].axis('off')

axes[1].imshow(image_seg,  cmap='gray')
axes[1].set_title('Segmentation HMC\n(EM + MPM + Peano)', fontsize=11)
axes[1].axis('off')

axes[2].imshow(image_otsu, cmap='gray')
axes[2].set_title("Seuillage d'Otsu",   fontsize=11)
axes[2].axis('off')

axes[3].imshow(image_kmeans, cmap='gray')
axes[3].set_title('K-Means (K=2)',       fontsize=11)
axes[3].axis('off')

plt.suptitle("Segmentation non supervisée – HMC vs méthodes classiques", fontsize=12)
plt.tight_layout()
plt.savefig(pathToSave + '_segmentation.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"\nFigure segmentation sauvegardée : {pathToSave}_segmentation.png")

# ─────────────────────────────────────────────────────────────────────────────
# 8. Figure bonus : histogramme + mélange gaussien estimé
# ─────────────────────────────────────────────────────────────────────────────
mu_est  = meanTabIter[nbIter-1]
var_est = varTabIter[nbIter-1]
I_est   = ITabIter[nbIter-1]

yvals = np.linspace(Y.min() - 10, Y.max() + 10, 500)
mix   = np.zeros_like(yvals)
for k in range(K):
    mix += I_est[k] * norm.pdf(yvals, loc=mu_est[k], scale=np.sqrt(var_est[k]))

fig2, ax = plt.subplots(figsize=(7, 4))
ax.hist(Y, bins=50, density=True, color='steelblue', alpha=0.6, label='Histogramme normalisé')
ax.plot(yvals, mix, 'r-', lw=2, label='Mélange gaussien estimé (EM)')
for k in range(K):
    comp = I_est[k] * norm.pdf(yvals, loc=mu_est[k], scale=np.sqrt(var_est[k]))
    ax.plot(yvals, comp, '--', lw=1.5,
            label=f'Classe {k} : μ={mu_est[k]:.1f}, σ={np.sqrt(var_est[k]):.1f}')
ax.set_xlabel('Niveau de gris', fontsize=11)
ax.set_ylabel('Densité', fontsize=11)
ax.set_title('Histogramme de l\'image et mélange gaussien estimé par EM', fontsize=11)
ax.legend(fontsize=9)
plt.tight_layout()
plt.savefig(pathToSave + '_histogramme.png', dpi=150, bbox_inches='tight')
plt.close()
print(f"Figure histogramme sauvegardée : {pathToSave}_histogramme.png")

print("\nTout est terminé. Fichiers dans ./results/")

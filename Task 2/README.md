# Analyse des propriétés optiques des métaux

Ce code Python permet d'analyser les propriétés optiques de différents métaux (or, cuivre, argent, fer, nickel) en fonction de l'épaisseur du film métallique et de l'angle d'incidence de la lumière. Il calcule et visualise la réflectivité (R), la transmissivité (T) et l'absorbance (A) en fonction de la longueur d'onde, ainsi que les pourcentages de lumière réfléchie, transmise et absorbée dans différentes plages spectrales (UV, visible, IR). De plus, il permet de comparer les résultats théoriques avec des données expérimentales issues de l'ellipsométrie.

## Concepts clés

### Réflectivité, Transmissivité, Absorbance (RTA)
Ces quantités décrivent comment un matériau interagit avec la lumière incidente :
- **Réflectivité (R)** : fraction de lumière réfléchie.
- **Transmissivité (T)** : fraction de lumière transmise.
- **Absorbance (A)** : fraction de lumière absorbée.

### Ellipsométrie
Technique expérimentale utilisée pour mesurer les propriétés optiques des matériaux, en particulier les angles Psi (Ψ) et Delta (Δ), qui décrivent l'état de polarisation de la lumière réfléchie.

### Interpolation
Les données optiques (indices de réfraction et coefficients d'extinction) sont interpolées pour permettre des calculs sur une plage continue de longueurs d'onde.

### Optimisation
Le code permet de trouver l'épaisseur optimale du film métallique qui minimise l'erreur entre les résultats théoriques et expérimentaux pour les angles Psi et Delta.

## Utilisation du code

### 1. Préparation des fichiers de données

#### Fichiers n/k
Les fichiers `n_k_glass.txt`, `n_k_gold.txt`, `n_k_copper.txt`, etc., contiennent les données des indices de réfraction (n) et des coefficients d'extinction (k) pour le verre et les métaux respectifs. Ces fichiers doivent être au format texte avec trois colonnes :
```
longueur d'onde (µm)  n  k
```

#### Fichiers d'ellipsométrie
Les fichiers `Ellipsometrie_45.txt` et `Ellipsometrie_65.txt` contiennent les données expérimentales de Psi et Delta pour des angles d'incidence de 45° et 65° respectivement.

### 2. Configuration des options dans la fonction `main`

Vous pouvez configurer les options suivantes :
```python
# Options de plot
RTA = True  # Trace R, T, A en fonction de la longueur d'onde
percentage_thikness = True  # Calcule les pourcentages de lumière réfléchie, transmise et absorbée
RvsWavelenth = False  # Ne trace pas R en fonction de la longueur d'onde
ellipsometry = True  # Compare les résultats théoriques et expérimentaux pour l'ellipsométrie
plot = True  # Active le tracé des graphiques
```

### 3. Exécution du code
Lancer simplement le script Python pour obtenir les graphiques et les résultats.

### 4. Modification des variables

#### Ajout de nouveaux métaux
Ajoutez un fichier `n_k_<nom_du_metal>.txt` et modifiez le dictionnaire `metals` :
```python
metals = {
    "Gold": {
        "file_nk": 'n_k_gold.txt',
        "n_interp": None,
        "k_interp": None,
    },
    "NewMetal": {  # Nouveau métal
        "file_nk": 'n_k_newmetal.txt',
        "n_interp": None,
        "k_interp": None,
    },
}
```

#### Modification des épaisseurs de film
```python
d_metal_values = [0.001, 0.01, 0.1, 1, 10, 100]  # Épaisseurs en nm
d_metal_values_log = np.logspace(-3, 4, 50)  # De 0.001 nm à 10000 nm
```

#### Changement des angles d'incidence
```python
angles_incidence = [27.1, 45, 65]  # Angles en degrés
ellipsometry_angles = [45, 65]  # Angles pour l'ellipsométrie
```

#### Ajout de nouvelles plages spectrales
```python
visible_range = (0.4, 0.7)  # En micromètres (µm)
non_visible_UV_range = (0.01, 0.4)  # UV en µm
non_visible_IR_range = (0.7, 1000)  # IR en µm
```

#### Modification de la plage de longueurs d'onde
```python
lambda_common = np.linspace(0.2, 20, num=1000)  # De 0.2 µm à 20 µm
```

#### Activation ou désactivation des options
```python
RTA = True  # Trace R, T, A
percentage_thikness = True  # Calcule les pourcentages de lumière
RvsWavelenth = False  # Ne trace pas R en fonction de la longueur d'onde
ellipsometry = True  # Compare les données d'ellipsométrie
plot = True  # Active l'affichage des graphiques
```

## Conclusion
Ce code est un outil puissant pour analyser les propriétés optiques des métaux en fonction de l'épaisseur du film et de l'angle d'incidence de la lumière. Il permet de visualiser les résultats sous forme de graphiques et de trouver les épaisseurs optimales pour différentes applications optiques. Sa flexibilité permet l'ajout de nouveaux métaux, épaisseurs, angles d'incidence ou plages spectrales.

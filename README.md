#  🚗 Apprentissage par renforcement - Traffic Signal Control  🚗

*Réalisé par Julie CIESLA, Clémence DUMOULIN et Pauline HOSTI*
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

Simulation et optimisation du trafic dans une intersection à l'aide d'algorithmes de Q-learning. L'objectif des simulations est de réduire au maximum le temps d'attente des véhicules aux feux de signalisation.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## 🛠️ Manuel technique

### Prérequis 
 - **Python** : 3.11

### Librairies
 - **numpy**
 - **random**
 - **matplotlib**
 - **heapq**

 Pour installer une librairie faire
 ```
pip install <nom librairie>
```
![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

### Organisation

Nous avons organisé notre travail en quatre fichiers : 
- utils.py
- RL_TSC_Phased_based_action.py
- RL_TSC_Step_based_action.py
- analyses.py

**UTILS**

Ce fichier contient toutes les fonctions utilitaires communes aux deux types de simulations. On y retrouve les fonctions pour :
  - Générer les arrivées de véhicules (avec des lois de Poisson) pour chaque direction.
  - Calculer le taux d'exploration (_epsilon_) et le taux d'apprentissage (_alpha_).
  - Mettre à jour les files d'attente des véhicules.
  - Calculer les récompenses.
  - Mapper les états (avec et sans durée) sur des entiers pour exploiter les Q-tables.


**RL_TSC_PHASED_BASED_ACTION**

Ce fichier contient l'implémentation de la simulation phased based décrite dans l'article.

**RL_TSC_STEP_BASED_ACTION**

Ce fichier contient l'implémentation de la simulation step based décrite dans l'article.

**ANALYSES**

Ce fichier permet de générer les divers résultats obtenus dans l'article.

![-----------------------------------------------------](https://raw.githubusercontent.com/andreasbm/readme/master/assets/lines/rainbow.png)

## 📑 Manuel utilisateur

En lançant le fichier *analyses.py* on obtient notre résultat pour la figure 1 de l'article ( pour un $\tau$ = 0.1, l'attente cumulée à chaque itération en utilisant l'espace d'action phased based, step based avec n = 5 et n = 15).

> pour changer les paramètres ou le résultat que l'on souhaite obtenir, aller directement dans le main du fichier *analyses.py*
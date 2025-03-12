####################### FONCTIONS UTILES #######################

###### LIBRAIRIES ######

import numpy as np
from heapq import heappush

###### CONSTANTES ######

DMIN = 5                  # Durée minimul d'un feu vert      
DMAX = 45                 # Durée maximum d'un feu vert
T_DUREE = 5               # Durée de la phase de transition
TPS_SIMULATION = 10000    # Durée d'une simulation
ITERATION = 100           # Nombre de simulation à faire      


def arrivees_voitures(param1, param2):
    """
    Génère les arrivées de véhicules dans chaque direction (Nord, Sud, Ouest, Est)
    sur toute la durée de la simulation, en s'appuyant sur une distribution de Poisson.
    
    Paramètres :
      - param1 : float
          Taux d'arrivée des véhicule dans toutes les directions
      - param2 : float
          Taux d'arrivée additionnel des véhicules pour les directions Nord et Sud.
    
    Renvoie :
      Tuple contenant 4 tableaux (ndarray) correspondant aux arrivées en 
      direction Nord, Sud, Ouest et Est, respectivement pour chaque top de temps.
    """
    arrivees_N = np.random.poisson(param1/4 + param2/2, TPS_SIMULATION)
    arrivees_S = np.random.poisson(param1/4 + param2/2, TPS_SIMULATION)
    arrivees_W = np.random.poisson(param1/4, TPS_SIMULATION)
    arrivees_E = np.random.poisson(param1/4, TPS_SIMULATION)
    
    return arrivees_N, arrivees_S, arrivees_W, arrivees_E


def get_epsilon(iteration):
    """
    Calcule le taux d'exploration epsilon, qui décroît de manière exponentielle 
    avec le nombre d'itérations. Cela permet de réduire progressivement l'exploration 
    au profit de l'exploitation au cours de l'apprentissage.
    
    Paramètre :
      iteration : int
          Numéro de l'itération courante de l'apprentissage.

    Renvoie :
      float 
    """

    return max(np.exp(-0.05 * iteration), 0.05)


def learning_rate(visites):
    """
    Détermine le taux d'apprentissage en fonction du nombre de visites 
    d'une action. Un nombre plus élevé de visites conduit à un taux plus faible, 
    garantissant ainsi une mise à jour plus fine des estimations.

    Paramètre :
      visites : int
          Nombre de fois qu'une action a été sélectionnée dans un état donné.

    Renvoie :
      float
    """

    return max(1 / visites, 0.05)


def ajout_voitures(t1, t2, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E):
    """
    Met à jour les files d'attente des véhicules pour chaque direction en ajoutant 
    les voitures arrivant entre t1 et t2. 

    Paramètres :
      t1, t2 : int
          Intervalle de temps pendant lequel les arrivées sont prises en compte.
      v_N, v_S, v_W, v_E : ndarray
          Tableaux contenant les arrivées de véhicules pour chaque direction en fontion du temps.
      q_N, q_S, q_W, q_E : list
          Files d'attente actuelles pour chaque direction.

    Renvoie :
      Tuple mis à jour des files d'attente (q_N, q_S, q_W, q_E)
    """
    for i in range(t1, min(t2, TPS_SIMULATION - 1)):
        N = v_N[i]
        S = v_S[i]
        W = v_W[i]
        E = v_E[i]
        if N != 0:
            for _ in range(N):
                heappush(q_N, i)
        if S != 0:
            for _ in range(S):
                heappush(q_S, i)
        if W != 0:
            for _ in range(W):
                heappush(q_W, i)
        if E != 0:
            for _ in range(E):
                heappush(q_E, i)
    return q_N, q_S, q_W, q_E


def calcul_attente(t, q_N, q_S, q_W, q_E):
    """
    Calcule le temps d'attente cumulé de toutes les voitures dans les files d'attente à l'instant t.

    Paramètres :
      t : int
          Instant de la simulation.
      q_N, q_S, q_W, q_E : list
          Files d'attente pour chaque direction, contenant l'instant d'arrivée des véhicules.

    Renvoie :
      int : Somme des temps d'attente de toutes les voitures.
    """
    attente_total = 0
    for q in (q_N, q_S, q_W, q_E):
        for arrivee in q:
            attente_total += (t - arrivee)
    
    return attente_total


def mapping_etat_phased():
    """
    Crée un dictionnaire de mapping pour la simulation "phased based". Chaque 
    état est représenté par un tuple (phase, c1, c2) où :
        - phase : indique la phase (0=Nord-Sud ou 1=Est-Ouest)
        - c1 : congestion sur l'axe Nord-Sud
        - c2 : congestion sur l'axe Est-Ouest
    Ce dictionnaire associe à chaque état un indice unique, utilisé dans le Q-learning.

    Renvoie :
      dict : Dictionnaire de mapping état -> indice.
    """
    mapping = {}
    indice = 0
    for c2 in range(3):
        for c1 in range(3):
            for phase in range(2):
                mapping[(phase, c1, c2)] = indice
                indice += 1
    
    return mapping


def mapping_etat_step():
    """
    Crée un dictionnaire de mapping pour la simulation "step based". Chaque état 
    est représenté par un tuple (phase, duree, c1, c2) où :
        - phase : indique la phase (0=Nord-Sud ou 1=Est-Ouest)
        - duree : durée actuelle du feu vert (entre DMIN et DMAX)
        - c1 : congestion sur l'axe Nord-Sud 
        - c2 : congestion sur l'axe Est-Ouest
    Ce dictionnaire associe à chaque état un indice unique, nécessaire pour 
    l'implémentation du Q-learning.

    Renvoie :
      dict : Dictionnaire de mapping état -> indice.
    """
    mapping = {}
    indice = 0
    for c2 in range(3):
        for c1 in range(3):
            for delai in range(DMIN, DMAX + 1):
                for phase in range(2):
                    mapping[(phase, delai, c1, c2)] = indice
                    indice += 1
    
    return mapping
####################### PHASED BASED ACTION #######################

###### LIBRAIRIES ######

import numpy as np
import random
from heapq import heappop
import matplotlib.pyplot as plt

from utils import *


###### SIMULATION ######

def phased_based_simulation(tau, facteur_discount, passage):
    """
    Réalise une simulation de contrôle d'intersection en mode "phased based" en 
    appliquant une stratégie de Q-learning.

    Paramètres :
      tau : float
          Paramètre influençant le taux additionnel d'arrivée des véhicules sur l'axe NORD-SUD
      facteur_discount : float
          Facteur de discount utilisé dans la mise à jour du Q-learning.
      passage : int
          Le temps que met une voiture à passer au feu vert.
    
    Renvoie :
      list : Liste de tuples (délai cumulé, nombre de véhicules passés) pour chaque itération.
    """

    # la somme des paramètres doit faire 0.5
    lam = 0.5 - tau

    # ETAPE 1 : ON INITIALISE LA TABLE Q
    # on initialise aussi une table pour laquelle on voit combien de fois on visite chaque couple (etat,action)
    espace_etats = 2 * 3 * 3
    espace_actions = DMAX - DMIN + 1

    Q = np.zeros((espace_etats, espace_actions))
    V = np.zeros((espace_etats, espace_actions))

    # dictionnaire qui nous permet de passer de tuple à indice dans la table Q
    mapping_etat_nbr = mapping_etat_phased()

    # pour chaque itération, on ajoute un tuple de la forme (attente totale, nbr voitures)
    res = []

    # ETAPE 2 : ON ENTRAINE LE CONTROLEUR D'INTERSECTION SUR 100 REPETITIONS
    for iteration in range(ITERATION):
        
        # génération arrivées des voitures au sein de l'intersection
        v_N, v_S, v_W, v_E = arrivees_voitures(lam, tau)

        # INITIALISATION

        # on commence par un feu rouge partout
        t = T_DUREE

        # nombre de voitures au niveau du carrefour au début en fonction des voies
        q_N, q_S, q_W, q_E = [], [], [], []
        q_N, q_S, q_W, q_E = ajout_voitures(0, t, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E)
        
        # etat initial
        phase = np.random.randint(0, 2)
        c_0 = (len(q_N) + len(q_S)) % 3
        c_1 = (len(q_W) + len(q_E)) % 3

        etat = (phase, c_0, c_1)

        # variable pour nos résultats
        delai_passage_cumule = 0
        voitures = 0

        # pour la récompense
        attente = calcul_attente(t, q_N, q_S, q_W, q_E)

        # UNE SIMULATION DURE TPS_SIMULATION TEMPS
        while t < TPS_SIMULATION : 
            
            # ETAPE 2 : EXPLORATION OU EXPLOITATION AVEC ALGORITHME EPSILON GLOUTON
            
            # au hasard on tire une probabilité
            exp_exp = random.uniform(0, 1)
            # calcul du epsilon qui diminue avec le nombre d'itérations
            epsilon = get_epsilon(iteration)
            
            # cas de l'exploitation
            if exp_exp > epsilon : 
                etat_ind = mapping_etat_nbr[etat]
                action  = np.argmax(Q[etat_ind,:]) + DMIN

            # cas de l'exploration
            else : 
                action = np.random.randint(DMIN, DMAX)

            # ETAPE 3 : ON APPLIQUE L'ACTION A REALISER

            # ACTION FEU VERT + PHASE DE TRANSITION

            new_phase = (etat[0] + 1) % 2
            new_t = t + action

            # on met à jour le nombre de voiture dans chaque voie
            # pendant que le feu est vert
            q_N, q_S, q_W, q_E = ajout_voitures(t, new_t, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E)

            # on enlève les voitures qui passent au feu vert
            for tps in np.arange(t, new_t, passage):
                # AXE NORD SUD
                if etat[0] == 0 : 

                    if q_N : 

                        v = heappop(q_N)
                        voitures += 1
                        delai_passage_cumule += (int(tps) - v)

                    if q_S : 

                        v = heappop(q_S)
                        voitures += 1
                        delai_passage_cumule += (int(tps) - v)

                # AXE OUEST-EST
                else : 

                    if q_W : 

                        v = heappop(q_W)
                        voitures += 1
                        delai_passage_cumule += (int(tps) - v)

                    if q_E : 

                        v = heappop(q_E)
                        voitures += 1
                        delai_passage_cumule += (int(tps) - v)              
            
            # on ajoute les voitures qui arrivent pendant le temps de transition
            q_N, q_S, q_W, q_E = ajout_voitures(new_t, new_t + T_DUREE, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E)
            new_t += T_DUREE

            c_0 = (len(q_N) + len(q_S)) % 3
            c_1 = (len(q_W) + len(q_E)) % 3

            new_etat = (new_phase, c_0, c_1)

            # ETAPE 5: ON MET A JOUR LA TABLE Q
            
            action -= DMIN

            # on calcule notre récompense
            new_attente = calcul_attente(new_t, q_N, q_S, q_W, q_E)
            reward = attente - new_attente
            attente = new_attente

            # on récupère les valeurs des indices pour nos calculs
            etat_ind = mapping_etat_nbr[etat]
            new_etat_ind = mapping_etat_nbr[new_etat]

            # on met à jour V
            V[etat_ind, action] += 1
            alpha = learning_rate(V[etat_ind, action])

            # on met à jour la table Q
            Q[etat_ind, action] = Q[etat_ind, action] + alpha * (reward + facteur_discount * np.max(Q[new_etat_ind, :]) - Q[etat_ind, action]) 
            
            etat = new_etat
            t = new_t + T_DUREE

        res.append((delai_passage_cumule, voitures))

    return res


def phased_based_simulation_repetee(tau, facteur_discount, passage, nbr_episode):
    """
    Exécute la simulation sur plusieurs épisodes et calcule la 
    moyenne des résultats (le délai cumulé) pour chaque itération.
    
    Paramètres :
      tau, facteur_discount, passage : paramètres de la simulation.
      nbr_episode : int
          Nombre d'épisodes sur lesquels lisser les résultats.

    Renvoie :
      list : Moyenne des résultats par itération
    """

    res = [0 for _ in range(ITERATION)]

    for _ in range(nbr_episode):

        res_episode = [r[0] for r in phased_based_simulation(tau, facteur_discount, passage)]
        
        for i in range(ITERATION) : 
            res[i] += res_episode[i]

    return [element / nbr_episode for element in res]
####################### STEP BASED ACTION #######################

###### LIBRAIRIES ######

import numpy as np
import random
from heapq import heappop
import matplotlib.pyplot as plt

from utils import *
from utils import *


###### SIMULATION ######

def step_based_simulation(tau, facteur_discount, n, passage):
    """
    Effectue une simulation "step based" où, à chaque pas de temps n, l'algorithme décide 
    s'il faut prolonger la durée du feu vert actuel ou bien changer de phase. Le choix 
    se fait à l'aide d'une stratégie epsilon-greedy et les mises à jour sont réalisées 
    via un algorithme de Q-learning.

    Paramètres :
      tau : float
          Paramètre influençant le taux additionnel d'arrivée des véhicules sur l'axe NORD-SUD
      facteur_discount : float
          Facteur de discount utilisé dans la mise à jour du Q-learning.
      passage : int
          Le temps que met une voiture à passer au feu vert.
      n : int
          intervalle de temps pour effectuer une action.
    
    Renvoie :
      list : Liste de tuples (délai cumulé, nombre de véhicules passés) pour chaque itération.
    """

    # la somme des paramètres doit faire 0.5
    lam = 0.5 - tau

    # ETAPE 1 : ON INITIALISE LA TABLE Q
    # on initialise aussi une table pour laquelle on voit combien de fois on visite chaque couple (etat,action)
    espace_etats = 2 * (DMAX - DMIN + 1) * 3 * 3
    espace_actions = 2

    Q = np.zeros((espace_etats, espace_actions))
    V = np.zeros((espace_etats, espace_actions))

    # dictionnaire qui nous permet de passer de tuple à indice dans la table Q
    mapping_etat_nbr = mapping_etat_step()

    # pour chaque itération, on ajoute un tuple de la forme (attente totale, nbr voitures)
    res = []

    # ETAPE 2 : ON ENTRAINE LE CONTROLEUR D'INTERSECTION SUR 100 REPETITIONS
    for iteration in range(ITERATION):
        
        # génération arrivées des voitures au sein de l'intersection
        v_N, v_S, v_W, v_E = arrivees_voitures(lam, tau)

        # INITIALISATION

        # temps initial (au min le feu est vert un temps DMIN)
        t = DMIN

        # nombre de voitures au niveau du carrefour au début en fonction des voies
        q_N, q_S, q_W, q_E = [], [], [], []
        q_N, q_S, q_W, q_E = ajout_voitures(0, t, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E)
        #print(f" voitures queue nord entre 0 et {t}: {q_N}")
        
        # etat initial
        phase = np.random.randint(0, 2)
        duree = t
        c_0 = (len(q_N) + len(q_S)) % 3
        c_1 = (len(q_W) + len(q_E)) % 3

        etat = (phase, duree, c_0, c_1)

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
                action  = np.argmax(Q[etat_ind,:])

            # cas de l'exploration
            else : 
                action = np.random.randint(0, 2)

            # ETAPE 3 : ON APPLIQUE L'ACTION A REALISER

            # CAS 1 : ON RESTE DANS LA PHASE ACTUELLE
            # si on choisit l'action 0 et que l'on a pas dépassé la borne supérieure
            if (action == 0) and (etat[1] + n <= DMAX) :

                new_phase = etat[0]
                new_delai = etat[1] + n
                new_t = t + n
                # à partir de quand débute le feu vert
                t_feu_vert = t

                # on met à jour le nombre de voiture dans chaque voie
                # pendant que le feu est vert
                q_N, q_S, q_W, q_E = ajout_voitures(t, new_t, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E)

            # CAS 2 : ON CHANGE DE PHASE
            # si on change de phase ou doit attendre T_DUREE avant qu'un feu passe au vert et au min le feu de l'autre phase dure DMIN temps
            else :

                new_phase = (etat[0] + 1) % 2
                new_delai = DMIN
                new_t = t + T_DUREE + DMIN
                # à partir de quand débute le feu vert
                t_feu_vert = new_t - DMIN

                # on met à jour le nombre de voiture dans chaque voie 
                # phase de transition + phase de feu vert
                q_N, q_S, q_W, q_E = ajout_voitures(t, new_t, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E)

            # POUR LES DEUX CAS, ON ENLEVE LES VOITURES LORS DU FEU VERT
            for tps in np.arange(t_feu_vert, new_t, passage):
                # AXE NORD SUD
                if new_phase == 0 : 

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
            
            c_0 = (len(q_N) + len(q_S)) % 3
            c_1 = (len(q_W) + len(q_E)) % 3
                    
            new_etat = (new_phase, new_delai, c_0, c_1)

            # ETAPE 5: ON MET A JOUR LA TABLE Q
            
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
            t = new_t

        res.append((delai_passage_cumule, voitures))

    return res


def step_based_simulation_repetee(tau, facteur_discount, n, passage, nbr_episode):
    """
    Exécute la simulation sur plusieurs épisodes et calcule la 
    moyenne des résultats (le délai cumulé) pour chaque itération.
    
    Paramètres :
      tau, facteur_discount, n, passage : paramètres de la simulation.
      nbr_episode : int
          Nombre d'épisodes sur lesquels lisser les résultats.

    Renvoie :
      list : Moyenne des résultats par itération
    """

    res = [0 for _ in range(ITERATION)]

    for _ in range(nbr_episode):

        res_episode = [r[0] for r in step_based_simulation(tau, facteur_discount, n, passage)]
        
        for i in range(ITERATION) : 
            res[i] += res_episode[i]

    return [element / nbr_episode for element in res]

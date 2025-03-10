####################### PHASED BASED ACTION #######################

###### LIBRAIRIES ######

import numpy as np
import random
from heapq import heappush, heappop
import matplotlib.pyplot as plt

###### CONSTANTES ######

DMIN = 5
DMAX = 45
T_DUREE = 5

TPS_SIMULATION = 10000
ITERATION = 100

###### FONCTIONS UTILES ######

def _arrivees_voitures(param1, param2):

    arrivees_N = np.random.poisson(param1/4 + param2/2, TPS_SIMULATION)
    arrivees_S = np.random.poisson(param1/4 + param2/2, TPS_SIMULATION)
    arrivees_W = np.random.poisson(param1/4, TPS_SIMULATION)
    arrivees_E = np.random.poisson(param1/4, TPS_SIMULATION)

    return arrivees_N, arrivees_S, arrivees_W, arrivees_E


def _get_epsilon(iteration):

    return max(np.exp(-0.05*iteration), 0.05)


def _learning_rate(visites):

    return max(1/visites, 0.05)


def _mapping_etat():

    dictionnaire_etat_nbr = {}
    nbr = 0

    for c2 in range(3):

        for c1 in range(3):

                for phase in range(2):
                    
                    dictionnaire_etat_nbr[(phase, c1, c2)] = nbr
                    nbr += 1
    
    return dictionnaire_etat_nbr


def _ajout_voitures(t1, t2, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E):
    
    for i in range(t1, min(t2, TPS_SIMULATION-1)):
        N = v_N[i]
        S = v_S[i]
        W = v_W[i]
        E = v_E[i]

        if N != 0:
            for elem in range(N): 
                heappush(q_N, i)
        
        if S != 0:
            for elem in range(S): 
                heappush(q_S, i)
        
        if W != 0:
            for elem in range(W): 
                heappush(q_W, i)
        
        if E != 0:
            for elem in range(E): 
                heappush(q_E, i)

    return q_N, q_S, q_W, q_E


def _calcul_attente(t, q_N, q_S, q_W, q_E):

    res = 0
    for i in q_N : 
        res += (t - i)
    for i in q_S : 
        res += (t - i)
    for i in q_W : 
        res += (t - i)
    for i in q_E : 
        res += (t - i)
    
    return res


###### SIMULATION ######

def _phased_based_simulation(tau, facteur_discount, passage):

    # la somme des paramètres doit faire 0.5
    lam = 0.5 - tau

    # ETAPE 1 : ON INITIALISE LA TABLE Q
    # on initialise aussi une table pour laquelle on voit combien de fois on visite chaque couple (etat,action)
    espace_etats = 2 * 3 * 3
    espace_actions = DMAX - DMIN + 1

    Q = np.zeros((espace_etats, espace_actions))
    V = np.zeros((espace_etats, espace_actions))

    # dictionnaire qui nous permet de passer de tuple à indice dans la table Q
    mapping_etat_nbr = _mapping_etat()

    # pour chaque itération, on ajoute un tuple de la forme (attente totale, nbr voitures)
    res = []

    # ETAPE 2 : ON ENTRAINE LE CONTROLEUR D'INTERSECTION SUR 100 REPETITIONS
    for iteration in range(ITERATION):
        
        # génération arrivées des voitures au sein de l'intersection
        v_N, v_S, v_W, v_E = _arrivees_voitures(lam, tau)

        # INITIALISATION

        # on commence par un feu rouge partout
        t = T_DUREE

        # nombre de voitures au niveau du carrefour au début en fonction des voies
        q_N, q_S, q_W, q_E = [], [], [], []
        q_N, q_S, q_W, q_E = _ajout_voitures(0, t, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E)
        
        # etat initial
        phase = np.random.randint(0, 2)
        c_0 = (len(q_N) + len(q_S)) % 3
        c_1 = (len(q_W) + len(q_E)) % 3

        etat = (phase, c_0, c_1)

        # variable pour nos résultats
        delai_passage_cumule = 0
        voitures = 0

        # pour la récompense
        attente = _calcul_attente(t, q_N, q_S, q_W, q_E)

        # UNE SIMULATION DURE TPS_SIMULATION TEMPS
        while t < TPS_SIMULATION : 
            
            # ETAPE 2 : EXPLORATION OU EXPLOITATION AVEC ALGORITHME EPSILON GLOUTON
            
            # au hasard on tire une probabilité
            exp_exp = random.uniform(0, 1)
            # calcul du epsilon qui diminue avec le nombre d'itérations
            epsilon = _get_epsilon(iteration)
            
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
            q_N, q_S, q_W, q_E = _ajout_voitures(t, new_t, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E)

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
            q_N, q_S, q_W, q_E = _ajout_voitures(new_t, new_t + T_DUREE, v_N, v_S, v_W, v_E, q_N, q_S, q_W, q_E)
            new_t += T_DUREE

            c_0 = (len(q_N) + len(q_S)) % 3
            c_1 = (len(q_W) + len(q_E)) % 3

            new_etat = (new_phase, c_0, c_1)

            # ETAPE 5: ON MET A JOUR LA TABLE Q
            
            action -= DMIN

            # on calcule notre récompense
            new_attente = _calcul_attente(new_t, q_N, q_S, q_W, q_E)
            reward = attente - new_attente
            attente = new_attente

            # on récupère les valeurs des indices pour nos calculs
            etat_ind = mapping_etat_nbr[etat]
            new_etat_ind = mapping_etat_nbr[new_etat]

            # on met à jour V
            V[etat_ind, action] += 1
            alpha = _learning_rate(V[etat_ind, action])

            # on met à jour la table Q
            Q[etat_ind, action] = Q[etat_ind, action] + alpha * (reward + facteur_discount * np.max(Q[new_etat_ind, :]) - Q[etat_ind, action]) 
            
            etat = new_etat
            t = new_t + T_DUREE

        res.append((delai_passage_cumule, voitures))

    return res


###### ANALYSES ######

def _phased_based_simulation_lissee(tau, facteur_discount, passage, nbr_episode):

    res = [0 for _ in range(ITERATION)]

    for _ in range(nbr_episode):

        res_episode = [r[0] for r in _phased_based_simulation(tau, facteur_discount, passage)]
        
        for i in range(len(res_episode)) : 
            res[i] += res_episode[i]

    return [element / nbr_episode for element in res]


def _convergence(data, fenetre, seuil) :

    # initialisation
    moyenne = sum(data[:fenetre])/fenetre

    for i in range(1, ITERATION-fenetre-1):

        new_moyenne = sum(data[i:i+fenetre])/fenetre

        if abs(new_moyenne - moyenne) / moyenne <= seuil : 

            print(f"converge à partir de l'itération {i}")

            break

        else : 

            moyenne = new_moyenne


def _affichage(res):

    iterations = range(len(res))
    delais = [r[0] for r in res]

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, delais, label="Délai cumulé")
    plt.xlabel("Itération")
    plt.ylabel("Valeur")
    plt.title("Évolution du résultat en fonction de l'itération")
    plt.legend()
    plt.grid(True)
    plt.show()

def _affichage1(res):

    iterations = range(len(res))
    delais = res

    plt.figure(figsize=(10, 5))
    plt.plot(iterations, delais, label="Délai cumulé")
    plt.xlabel("Itération")
    plt.ylabel("Valeur")
    plt.title("Évolution du résultat en fonction de l'itération")
    plt.legend()
    plt.grid(True)
    plt.show()

###### MAIN ######

if __name__ == '__main__':

    tau = 0.1
    facteur = 0.9
    passage = 1

    resultat = _phased_based_simulation_lissee(tau, facteur, passage, 5)
    _affichage1(resultat)
    #_convergence(resultat, 5, 0.01)

    #moyenne_step(facteur, passage)
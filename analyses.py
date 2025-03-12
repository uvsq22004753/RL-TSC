####################### ANALYSES #######################

###### LIBRAIRIES ######

import matplotlib.pyplot as plt
from utils import *
from RL_TSC_Phased_based_action import *
from RL_TSC_Step_based_action import *


###### ANALYSES ######

def affichage_une_courbe(resultat):
    """
    Cette fonction affiche un graphique représentant l'évolution du temps d'attente
    cumulé en fonction des itérations de la simulation.

    Paramètre :
      - resultat : list
          Liste contenant les résultats de la simulation. Chaque élément peut être soit 
          un float représentant directement le temps d'attente cumulé, soit un tuple où 
          le premier élément correspond à ce temps.
    """

    x = range(ITERATION)
    
    if type(resultat[0])  == float : 
        y = resultat
    else : 
        y = [r[0] for r in resultat]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y)
    plt.xlabel("Itération")
    plt.ylabel("Temps d'attente cumulé")
    plt.title("Temps d'attente cumulé au fil des itérations")
    plt.legend()
    plt.grid(True)
    plt.show()


# pour retrouver la Figure 1 du document
def affichage_figure_une(resultat_phased, resultat_n5, resultat_n15):
    """
    Cette fonction affiche un graphique représentant l'évolution du temps d'attente
    cumulé en fonction des itérations de la simulation pour phased based et deux step
    based : 5 et 15.

    Paramètre :
      - resultat_phases : list
      - resultat_n5 : list
      - resultat_n15 : list
    """
    x = range(ITERATION)
    
    if type(resultat_phased[0])  == float : 
        y1 = resultat_phased
    else : 
        y1 = [r[0] for r in resultat_phased]

    if type(resultat_n5[0])  == float : 
        y2 = resultat_n5
    else : 
        y2 = [r[0] for r in resultat_n5]

    if type(resultat_n15[0])  == float : 
        y3 = resultat_n15
    else : 
        y3 = [r[0] for r in resultat_n15]

    plt.figure(figsize=(10, 5))
    plt.plot(x, y1, label="phase")
    plt.plot(x, y2, label="step n=5")
    plt.plot(x, y3, label="step n=15")
    plt.xlabel("Itération")
    plt.ylabel("Temps d'attente cumulé")
    plt.title("Temps d'attente cumulé au fil des itérations")
    plt.legend()
    plt.grid(True)
    plt.show()


# pour les résultats de la table 1
def moyenne_step_phased(facteur, passage, repetition):
    """
    Cette fonction renvoie pour chaque tau le temps moyen d'attente cumulé en considérant
    que l'on converge dès l'itération 50.

    Paramètre :
      - facteur : facteur de discount
      - passage : temps que met une voiture pour passer
    """
    for tau in [0.0, 0.1, 0.2 ,0.3, 0.4, 0.5] :

        # calcul des performances
        performances = {}

        # pour les steps

        for step in [1, 5, 10, 15, 20] : 
            res = step_based_simulation_repetee(tau, facteur, step, passage, repetition)
            m = [r for r in res[50:]]
            performances[step] = sum(m) / (ITERATION//2)
        
        # pour la phase
        res = phased_based_simulation_repetee(tau, facteur, passage, repetition)
        m = [r for r in res[50:]]
        performances['phased'] = sum(m) / (ITERATION//2)



        print(f"pour le taux {tau} on a les performances {performances}")



# pour les résultats de la table 2
def moyenne_step(facteur, passage):
    """
    Cette fonction renvoie pour chaque tau le temps moyen d'attente cumulé en considérant
    que l'on converge dès l'itération 50.

    Paramètre :
      - facteur : facteur de discount
      - passage : temps que met une voiture pour passer
    """

    for tau in [0.0, 0.1, 0.2 ,0.3, 0.4, 0.5] :

        # calcul des performances
        performances = {}

        for step in [1, 5, 10, 15, 20] : 
            res = step_based_simulation_repetee(tau, facteur, step, passage)
            m = [r for r in res[50:]]
            performances[step] = sum(m) / (ITERATION//2)

        print(f"pour le taux {tau} on a les performances {performances}")

###### SIMULATION EN COURS ######

if __name__ == '__main__':

    tau = 0.1
    facteur = 0.8
    n = 20
    passage = 1
    repetition = 5

    #resultat = step_based_simulation_repetee(tau, facteur, n, passage, repetition)
    #affichage_une_courbe(resultat)

    resultat_phased = phased_based_simulation_repetee(0.1, facteur, passage, repetition)
    resultat_n5 = step_based_simulation_repetee(0.1, facteur, 5, passage, repetition)
    resultat_n15 = step_based_simulation_repetee(0.1, facteur, 15, passage, repetition)
    affichage_figure_une(resultat_phased, resultat_n5, resultat_n15)

    #moyenne_step_phased(facteur, passage, repetition)


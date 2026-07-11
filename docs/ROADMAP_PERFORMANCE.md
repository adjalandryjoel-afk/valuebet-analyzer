# Feuille de route performance — Value Bet Analyzer

*Recherche documentée du 2026-07-11 (3 agents : modèles de prédiction,
pratique professionnelle, sources de données). Chaque point cite ses
sources. Priorisation en fin de document.*


## Axe 1 — Modèles de prédiction

*L'architecture de l'app (Poisson ancré marché + blend Elo) est conforme à l'état de l'art dans sa STRUCTURE, mais quatre corrections peu coûteuses la sépareraient d'un modèle amateur : (1) la correction tau de Dixon-Coles sur les scores 0-0/1-0/0-1/1-1 — gain faible sur le 1X2 mais net sur la calibration des marchés O/U, BTTS et totaux, cœur de l'app ; (2) la pondération temporelle exponentielle exp(-0.002*jours), critique vu les saisons anciennes d'API-Football ; (3) la méthode de Shin à la place de la normalisation proportionnelle du vig (odds_collector.py) — la normalisation actuelle biaise l'ancre marché et fabrique de la fausse value sur les outsiders, aggravée par les fortes marges Betclic CI ; (4) un vrai dispositif de calibration (Brier + reliability diagrams sur TOUTES les prédictions, pas seulement les paris placés comme dans backtester.py actuellement) — c'est le critère décisif car value = p*cote-1 transforme toute surconfiance en value fictive. Sur l'ancrage : 60% marché est plutôt trop faible qu'excessif pour un modèle aux données dégradées (la littérature ne bat pas les cotes de clôture ; optimiser ce poids par backtest, et ancrer sur Pinnacle/consensus plutôt que sur Betclic pour que la value signifie 'Betclic s'écarte du marché sharp'). Les constantes empiriques de l'app sont validées : 45/55 par mi-temps est correct (à moduler par ligue, Bundesliga ~51/49) et 3.3 SOT/but est dans la fourchette documentée 2.9-3.3 (3.1 plus central). Le plus gros gain d'entrée de données serait le passage aux npxG (meilleur prédicteur que les buts), mais il reste bloqué par l'accès aux données depuis l'environnement actuel — à traiter après les corrections 1-4.*


### Correction Dixon-Coles (rho) sur les scores faibles : le gain est réel là où l'app parie le plus (O/U, BTTS, nul)
**Difficulté : facile**

Le Poisson indépendant sous-estime systématiquement les 0-0 et 1-1 et surestime les 1-0/0-1. Dixon & Coles (1997) corrigent via un facteur tau appliqué aux 4 cases (0,0),(0,1),(1,0),(1,1) : tau = 1 - lambda*mu*rho pour (0,0) ; 1 + lambda*rho pour (0,1) ; 1 + mu*rho pour (1,0) ; 1 - rho pour (1,1) ; 1 ailleurs. Avec la convention originale, rho estimé est négatif (|rho| typique 0.05-0.15), ce qui gonfle 0-0 et 1-1 et dégonfle 1-0/0-1 (attention : certaines implémentations inversent le signe). Gain documenté : +1 à 3% de log-vraisemblance, quasi nul sur la précision 1X2, mais amélioration nette de la CALIBRATION des scorelines exactes — donc des marchés Under 2.5, BTTS Non, nul et totaux par équipe. L'alternative bivariée (Karlis & Ntzoufras 2003, inflation de la diagonale) vise le même problème mais est plus lourde ; la correction tau suffit.

**Pour l'app :** Quelques lignes dans poisson_model.py (_fill_probabilities) : après construction de la matrice de scores Poisson, multiplier les 4 cases basses par tau avec un rho fixe (~ -0.10 en convention originale, à défaut de pouvoir l'estimer par MLE) puis renormaliser la matrice. Cela corrige d'un coup 1X2 (nul), O/U 2.5, BTTS et les totaux par équipe 0.5 — les marchés cœur de l'app. Sans cette correction, l'app voit de la fausse value sur Over/BTTS Oui et rate de la vraie value sur Under/0-0.

Sources : https://grokipedia.com/page/DixonColes_model · https://statsultra.com/dixon-coles-model/ · https://football-bet-prediction.com/articles/dixoncoles-model-explained-improving-poisson/ · https://www.emergentmind.com/topics/bivariate-dixon-and-coles-model


### Pondération temporelle exponentielle des matchs (time decay) : indispensable vu les données anciennes de l'app
**Difficulté : facile**

Deuxième apport de Dixon-Coles : pondérer chaque match passé par w = exp(-xi * t) où t est l'ancienneté en jours. Valeurs optimales documentées par backtesting sur les grands championnats : xi ≈ 0.0018-0.0023 par jour (demi-vie ~300 jours ; le 0.0065 original était en demi-semaines, soit ~0.00186/jour). Un xi trop agressif (demi-vie 60 j) réduit l'échantillon effectif et augmente le bruit ; xi=0 (poids uniforme) laisse des équipes surévaluées sur la base de saisons révolues. Le paramètre doit être recalé sur ses propres données, pas repris tel quel.

**Pour l'app :** L'app agrège des stats API-Football de saisons ANCIENNES (plan gratuit) : c'est exactement le cas où l'absence de décote temporelle fait le plus mal. À implémenter dans l'agrégation des stats d'équipe (data_collector/api_football) : moyennes de buts marqués/encaissés pondérées par exp(-0.002 * jours_écoulés) au lieu de moyennes simples. Bonus : cela dévalorise automatiquement les saisons trop vieilles et rend visible quand les données sont trop périmées pour être utiles (somme des poids faible => basculer 100% marché).

Sources : https://opisthokonta.net/?p=1013 · https://dashee87.github.io/football/python/predicting-football-results-with-statistical-modelling-dixon-coles-and-time-weighting/ · https://artiebits.com/blog/improving-poisson-model-using-time-weighting/


### La calibration (Brier, courbes de fiabilité) est LE critère de validation — et le Brier actuel de l'app est biaisé
**Difficulté : moyen**

La littérature recommande log-loss et Brier score plutôt que la précision : value = p*cote - 1 est une fonction directe de p, donc toute surconfiance du modèle fabrique de la value fictive. Un modèle peut bien classer les issues et être mal calibré (parmi les événements notés p=0.60, il faut qu'environ 60% se réalisent). L'outil standard : reliability diagram (probabilités prédites par déciles vs fréquences observées) + Brier décomposé, calculé sur TOUTES les prédictions, pas seulement les paris placés — un Brier restreint aux paris sélectionnés souffre d'un biais de sélection (on ne garde que les cas où le modèle diverge du marché, précisément ceux où il est le plus souvent surconfiant). Recalibration possible ensuite (Platt/isotonique) dès ~200-300 observations.

**Pour l'app :** backtester.py calcule déjà un Brier mais uniquement sur les BetRecord (paris placés) : le biais de sélection le rend trompeur. Actions : (1) logger dans SQLite les probabilités du modèle pour TOUS les marchés de TOUS les matchs analysés, même sans pari ; (2) page Streamlit avec reliability diagram par marché (1X2 / O/U / BTTS) et Brier vs Brier du marché no-vig (le benchmark à battre) ; (3) si la courbe montre une surconfiance systématique, appliquer un shrinkage vers le marché avant le calcul de value. C'est le meilleur investissement de l'app : il dit si les 5% de seuil de value sont réels ou du bruit.

Sources : https://exprysm.com/insights/methodology/model-calibration.html · https://www.truevalueengine.com/what-is-a-brier-score-in-sports-betting-analytics/ · https://www.sports-ai.dev/blog/ai-model-calibration-brier-score


### Remplacer la normalisation proportionnelle du no-vig par la méthode de Shin (biais favori-outsider)
**Difficulté : facile**

L'app retire la marge en divisant chaque probabilité brute par leur somme (odds_collector.py l.300-315). Or la marge des bookmakers n'est pas répartie proportionnellement : elle est chargée sur les outsiders (favourite-longshot bias). Štrumbelj (2014, International Journal of Forecasting) a montré que les probabilités Shin (modèle avec 'insiders') donnent les prévisions les plus précises dans 5 sports dont le football, devant la normalisation basique ; Clarke et al. confirment. Conséquence de la normalisation proportionnelle : les probabilités des favoris sont sous-estimées et celles des outsiders surestimées — l'ancre marché de l'app est donc systématiquement biaisée, surtout sur Betclic CI où les marges sont élevées (>8%), ce qui amplifie le biais.

**Pour l'app :** Impact direct et double : (1) l'ancre no-vig qui pèse 60% des lambdas est faussée, (2) la value détectée sur les outsiders (cotes hautes) est gonflée artificiellement — exactement le piège classique de l'amateur. Implémentation triviale : package Python 'shin' (mberk/shin, pip install shin) ou méthode 'power'/'shin' du package implied. Remplacer raw[i]/total par shin.calculate_implied_probabilities(odds). Effet attendu : moins de faux signaux de value sur les cotes > 3.5.

Sources : https://journals.sagepub.com/doi/10.1177/1527002513519329 · https://www.researchgate.net/publication/264349990_On_determining_probability_forecasts_from_betting_odds · https://github.com/mberk/shin · https://cran.r-project.org/web/packages/implied/vignettes/introduction.html


### Ancrage marché 60% : plutôt trop FAIBLE que trop fort, et l'ancre devrait être Pinnacle/consensus, pas Betclic
**Difficulté : moyen**

La littérature est quasi unanime : les cotes de clôture des books sharp sont le meilleur prédicteur disponible, et les modèles simples ne les battent pas (Wilkens 2026 sur la Bundesliga : les modèles xG capturent des signaux structurels mais le marché reste mieux calibré ; la voie recommandée est l'ENSEMBLE modèle+marché). Egidi, Pauli & Torelli (2018) formalisent exactement l'approche de l'app — combinaison convexe des taux de but estimés sur l'historique et déduits des cotes — mais ESTIMENT le poids de mélange par inférence bayésienne au lieu de le fixer, et obtiennent des retours positifs en stratégie EV. Pour un modèle amateur avec données dégradées, le poids marché optimal est vraisemblablement 70-90%, pas 60%. Point logique en plus : détecter de la value contre Betclic alors que l'ancre EST la cote Betclic crée un auto-rétrécissement incohérent des edges.

**Pour l'app :** Deux actions : (1) traiter MARKET_WEIGHT non comme une constante mais comme un paramètre à optimiser dans backtester.py (grille 0.5-0.9, critère = Brier/log-loss out-of-sample) — vu la qualité des stats disponibles, s'attendre à un optimum > 0.60 ; (2) ancrer les lambdas sur le consensus/Pinnacle (odds_collector expose déjà pinnacle_prob) et calculer la value contre la cote Betclic : l'edge devient 'Betclic s'écarte du marché sharp + du modèle', la seule configuration où un amateur a une chance réaliste d'être gagnant à long terme.

Sources : https://arxiv.org/abs/1802.08848 · https://journals.sagepub.com/doi/10.1177/22150218261416681 · https://www.researchgate.net/publication/328490567_Combining_historical_data_and_bookmakers'_odds_in_modelling_football_scores


### Répartition mi-temps : 45/55 est correct en moyenne mais devrait être paramétré par ligue
**Difficulté : facile**

Les données confirment largement le 45/55 : ~44.3%/55.7% sur 5 saisons des 4 divisions anglaises, ~44/56 persistant dans les grands championnats européens (fatigue, prises de risque en fin de match ; pic de buts 76e-90e+). MAIS la répartition varie par ligue : Bundesliga ~51/49 (quasi équilibrée), Premier League plus équilibrée que la moyenne, La Liga/Serie A/Ligue 1 plus chargées en 2e mi-temps. Autre résultat utile (Grayson) : les buts de 1re mi-temps ne prédisent pas ceux de la 2e — les deux mi-temps sont quasi indépendantes, ce qui valide l'approche 'deux Poisson séparés par mi-temps'.

**Pour l'app :** FIRST_HALF_SHARE = 0.45 est une bonne valeur par défaut : la garder. Amélioration à faible coût : la déplacer dans le dictionnaire par ligue de config.py (comme avg_goals/home_win_rate), avec ~0.49 pour la Bundesliga, ~0.47 pour la Premier League, ~0.44-0.45 pour Liga/Serie A/Ligue 1. Attention aussi à l'asymétrie des marchés : le marché 'but 1re mi-temps' de Betclic est structurellement plus piégeux car lambda_1MT ~0.45*lambda rend l'Under 0.5 1MT plus probable que l'intuition ne le suggère.

Sources : https://www.thestatsdontlie.com/1st-2nd-half-goals/ · https://www.sportingpedia.com/2024/10/17/first-vs-second-half-goal-distribution-across-europes-top-5-leagues-scoring-patterns-of-all-96-teams/ · https://jameswgrayson.wordpress.com/2013/12/31/are-goals-scored-in-the-first-half-predictive-of-goals-scored-in-the-second/


### Ratio tirs cadrés/buts : 3.3 est dans la fourchette haute ; le vrai problème est l'uniformité du ratio
**Difficulté : facile**

Les taux de conversion documentés : ~10-12% de tous les tirs deviennent des buts, ~30-35% des tirs cadrés — soit ~2.9 à 3.3 tirs cadrés par but. Le 3.3 de l'app correspond à une conversion de ~30%, la borne basse de la fourchette : légèrement pessimiste sur le nombre de SOT (donc l'app sous-estime un peu les Over SOT). Surtout, ce ratio varie fortement selon le profil d'équipe (équipes dominantes qui tirent de loin vs contre-attaquantes à grosses occasions) et un Poisson sur les SOT dérivé linéairement des buts attendus ignore cette hétérogénéité.

**Pour l'app :** Garder SOT_PER_GOAL entre 3.0 et 3.3 comme défaut (3.1 serait plus central). Amélioration simple : quand les stats API-Football sont disponibles, calculer le ratio SOT/buts propre à chaque équipe sur les matchs récents (pondérés par le time decay du finding 2) et le mélanger avec le prior 3.1 (shrinkage 50/50). Ajouter un garde-fou : si le ratio d'équipe sort de [2.5, 4.5], revenir au prior — c'est du bruit d'échantillon.

Sources : https://www.sofascore.com/news/a-statistical-breakdown-of-shots-shots-on-target-and-big-chances · https://www.statscore.com/news-center/sport/soccer/what-does-the-shot-conversion-rate-for-goals-means-in-soccer/ · https://www.sportmonks.com/blogs/shot-conversion-rate/


### xG/npxG comme entrée du modèle : le plus gros gain théorique, mais bloqué par l'accès aux données
**Difficulté : difficile**

La littérature converge : les xG (et surtout npxG, hors penalties) prédisent mieux la performance FUTURE que les buts réels, car un tir est un événement ~10x plus fréquent qu'un but — la variance d'échantillon est bien moindre et la sur/sous-performance de finition régresse vers la moyenne. Les études comparatives montrent qu'un Poisson alimenté par les xG estime mieux les scorelines qu'un Poisson alimenté par les buts, et Wilkens (2026) trouve que les modèles xG dégagent une profitabilité modeste mais consistante en simulation, là où les modèles à base de buts échouent. C'est LE remplacement d'entrée prioritaire pour les lambdas côté 'stats réelles'.

**Pour l'app :** L'app a déjà tenté (xg_scraper.py bloqué en 403 sur Understat). Pistes réalistes depuis la Côte d'Ivoire : (1) le package Python 'understat' (API async non officielle) ou les endpoints JSON d'Understat avec headers navigateur complets ; (2) FBref via export CSV manuel hebdomadaire (gratuit, top 5 ligues + autres) importé dans SQLite ; (3) FootyStats API (plan gratuit limité). Une fois les npxG pour/contre par équipe disponibles, remplacer buts marqués/encaissés par un mélange 70% npxG / 30% buts dans _lambdas_from_stats. À faire APRÈS les findings 1-5 : sans calibration ni Shin, de meilleures données ne se verront même pas.

Sources : https://www.ncbi.nlm.nih.gov/pmc/articles/PMC11524524/ · https://octosport.medium.com/the-power-of-football-expected-goals-0d4849a5ff1b · https://bet2invest.com/blog/Can-xG-be-used-to-predict-soccer-matches · https://journals.sagepub.com/doi/10.1177/22150218261416681


## Axe 2 — Pratique professionnelle du value betting

*La recherche sur les pratiques des value bettors professionnels révèle trois écarts majeurs entre l'app et l'état de l'art. (1) Méthode de détection : les pros comparent la cote du book soft à une référence sharp indépendante (Pinnacle no-vig) ; l'app s'ancre à 60% sur les cotes de Betclic lui-même, ce qui dilue mécaniquement le signal — c'est la correction la plus impactante. (2) Suivi : le KPI universel des pros est le Closing Line Value, significatif après ~100-300 paris là où le ROI exige des milliers de paris ; l'app ne le calcule pas, alors qu'un scraping des cotes proche du coup d'envoi suffirait (et fournirait en bonus la détection des lignes gelées, principal gisement de value chez un book soft). (3) Calibration des seuils : le biais favori-outsider impose un seuil de value croissant avec la cote (5% sous 2,50, 8-12% au-delà) et une dé-vigorisation proportionnelle plutôt qu'égale ; les marchés secondaires (BTTS, totaux équipe, mi-temps, tirs cadrés) portent des marges de 7-9% qui exigent des seuils relevés et un suivi ROI/CLV ventilé par marché. Côté bankroll, le quart de Kelly est validé par la littérature mais le plafond devrait passer de 5% à 2% tant que le CLV n'a pas confirmé l'edge, et un simulateur Monte Carlo de variance/drawdown (25% de chances de 10 défaites d'affilée même avec 5% d'edge) protégerait l'utilisateur des deux erreurs fatales des amateurs : abandonner pendant une série normale de pertes, ou sur-miser après une série chanceuse. Enfin, le risque n°1 chez un book soft n'est pas la variance mais la limitation du compte gagnant : un mode « discrétion » (mises arrondies, diversification des marchés) prolongerait la durée de vie du compte Betclic CI.*


### Tracker le Closing Line Value (CLV) : le KPI n°1 des pros, absent de l'app
**Difficulté : moyen**

Les professionnels jugent leur edge non pas sur le ROI (trop bruité) mais sur le CLV : CLV% = (cote prise / cote de clôture) − 1, la cote de clôture (idéalement Pinnacle) étant le prix le plus efficient du marché. Battre régulièrement la clôture de +2% est le signe reconnu d'un parieur gagnant, détectable après ~100-300 paris, alors que le ROI exige des milliers de paris pour être statistiquement fiable. Buchdahl affiche ~3,7% de profit sur 18 000 paris avec cette approche.

**Pour l'app :** Ajouter au module SQLite deux colonnes (cote_cloture, clv_pct) : re-scraper la cote Betclic (et si possible Pinnacle via un agrégateur) du marché parié juste avant le coup d'envoi, calculer le CLV no-vig, puis afficher dans le dashboard Streamlit le CLV moyen, le % de paris battant la clôture et la courbe CLV cumulée à côté du ROI. C'est le moyen le plus rapide de savoir si le modèle a un vrai edge, bien avant que le ROI ne soit significatif.

Sources : https://oddsjam.com/betting-education/closing-line-value · https://www.pinnacleoddsdropper.com/blog/closing-line-value--clv-demystified-by-expert-joseph-buchdahl · https://pikkit.com/blog/how-to-track-closing-line-value-clv-in-sports-betting · https://www.sharpfootballanalysis.com/sportsbook/clv-betting/


### Ancrage circulaire : le modèle doit s'ancrer sur une référence sharp, pas sur Betclic lui-même
**Difficulté : moyen**

La méthode standard des pros (RebelBetting, Trademate, méthode « Wisdom of the Crowd » de Buchdahl) consiste à comparer la cote d'un book soft à la probabilité no-vig d'une référence indépendante et efficiente (Pinnacle, Betfair). Or l'app ancre son Poisson à 60% sur les cotes no-vig de Betclic : la value détectée ne peut alors provenir que des 40% restants (Elo estimé + stats API-Football anciennes), la partie la plus faible du modèle, et le signal est mécaniquement dilué.

**Pour l'app :** Remplacer (ou compléter) l'ancre marché par les cotes Pinnacle no-vig du même match, récupérées via un agrégateur (ex. The Odds API, plan gratuit 500 req/mois) : value = p(ancrée sharp) × cote_Betclic − 1. Quand Pinnacle n'est pas disponible (petites ligues), le signaler et exiger un seuil de value plus élevé. C'est le changement de méthode au plus fort impact sur la fiabilité de la détection.

Sources : https://www.football-data.co.uk/wisdom_of_crowd_bets · https://www.rebelbetting.com/blog/difference-soft-sharp-bookmakers · https://smartsportstrader.com/best-value-betting-software/


### Biais favori-outsider : moduler le seuil de value (5%) selon la fourchette de cotes
**Difficulté : facile**

La marge du bookmaker n'est pas répartie uniformément : elle est concentrée sur les outsiders (espérance souvent < −10% sur les grosses cotes, contre −1/−2% sur les favoris). Une « value » apparente sur une cote élevée est donc très souvent une erreur de modèle amplifiée par la répartition de marge, tandis que la value sur favori est plus robuste. La normalisation proportionnelle simple du vig aggrave le problème car elle surestime la probabilité vraie des outsiders.

**Pour l'app :** Deux changements concrets : (1) retirer la marge avec une méthode proportionnelle aux cotes (odds ratio / logarithmique, à la Buchdahl) plutôt qu'une normalisation égale ; (2) rendre le seuil de value croissant avec la cote — garder 5% sous 2,50, exiger ~8% entre 2,50 et 4,00, ~12% au-delà, voire refuser les cotes > 5-6. Le garde-fou actuel « value > 40% rejetée » va dans le bon sens mais est trop grossier.

Sources : https://thewagertheorem.com/favorite-longshot-bias-betting/ · https://footballdotpy.medium.com/the-favourite-longshot-bias-in-sports-betting-ef9c5cfde38 · https://www.sciencedirect.com/science/article/abs/pii/S1062976916000041


### Marchés secondaires : mesurer la marge réelle par marché et suivre le ROI/CLV par marché
**Difficulté : facile**

Les marges sont bien plus élevées sur les marchés dérivés (BTTS médiane ~7-8,6%, totaux équipe, mi-temps, tirs cadrés) que sur le 1X2 (~5-8% chez les books grand public, moins chez les sharps). C'est là que les books softs sont les moins précis — donc où l'opportunité existe — mais aussi là où le vig mange l'edge et où les heuristiques de l'app (répartition 45/55 mi-temps fixe, tirs cadrés = 3,3 × buts attendus) sont les plus fragiles : la « value » détectée y est majoritairement du bruit de modèle.

**Pour l'app :** L'app scrape déjà les cotes : calculer et afficher la marge réelle de chaque marché (somme des probabilités implicites − 1), exiger seuil de value + marge (pas seulement value > 5%), et ventiler l'historique SQLite par type de marché (ROI, CLV, nb paris par marché) pour découvrir empiriquement où Betclic CI est réellement mal calibré au lieu de le supposer. Couper les marchés dont le CLV par marché est négatif après ~100 paris.

Sources : https://365bettingtips.com/bookmakers/odds/btts · https://topbookmakerfootball.com/football-betting-markets-explained-complete-bookmaker-guide/ · https://footiqo.com/betting-guide/bookmaker-analysis-for-smarter-football-betting/


### Variance et significativité : ajouter un simulateur Monte Carlo et un p-value au suivi
**Difficulté : moyen**

Les amateurs confondent variance et edge : avec 5% d'edge réel, il reste ~25% de chances de subir 10 défaites d'affilée sur 1000 paris, et un échantillon de 2 375 paris a montré +5,77% de ROI là où 17 717 paris du même système donnaient −0,63%. En dessous de plusieurs milliers de paris, le ROI seul ne prouve rien ; les pros s'appuient sur des simulations de distribution de profits, de drawdown attendu et sur le p-value de leurs résultats.

**Pour l'app :** Ajouter au dashboard un onglet « Réalité statistique » : simulation Monte Carlo (numpy) à partir des paris historiques réels (cotes, mises, edges estimés) affichant la distribution des profits possibles, le drawdown maximal attendu, la probabilité d'être en perte après N paris malgré un edge positif, et un p-value du ROI observé. Effet direct : éviter l'abandon après une mauvaise série ou la sur-confiance après une bonne, les deux erreurs qui tuent les value bettors amateurs.

Sources : https://punter2pro.com/sample-size-betting-results-analysis/ · https://www.rebelbetting.com/faq/expected-value-and-variance · https://winnerodds.com/valuebettingblog/drawdown-monte-carlo-simulation-calculator-for-sports-betting/


### Quart de Kelly : bon choix, mais plafond à baisser et mise à réduire quand modèle et marché divergent fort
**Difficulté : facile**

Le Kelly fractionnaire (1/4 ou 1/2) est le standard professionnel précisément parce que l'edge estimé est incertain : surestimer son edge de 10% conduit à sur-miser dangereusement, et le full Kelly implique 50% de chances de perdre 50% de la bankroll. Le quart de Kelly de l'app est conforme, mais le plafond de 5% de bankroll par pari reste élevé pour un modèle amateur non validé ; les pros plafonnent plutôt à 1-2% tant que le CLV n'a pas confirmé l'edge.

**Pour l'app :** Baisser le plafond de 5% à 2% de la bankroll tant que le CLV moyen n'est pas positif sur 100+ paris (déblocage progressif ensuite). Ajouter un multiplicateur de confiance qui réduit la mise quand l'écart modèle-marché est extrême (value > 15-20% = probable erreur de modèle, cohérent avec le garde-fou existant) et afficher à côté de chaque mise suggérée le drawdown attendu pour la fraction de Kelly choisie.

Sources : https://matthewdowney.github.io/uncertainty-kelly-criterion-optimal-bet-size.html · https://www.degruyterbrill.com/document/doi/10.1515/jqas-2020-0122/html · https://en.wikipedia.org/wiki/Kelly_criterion


### Risque de limitation de compte chez un book soft : à anticiper dans l'outil
**Difficulté : facile**

Les bookmakers softs comme Betclic limitent ou ferment systématiquement les comptes gagnants réguliers (« high margin, lower volume ») — c'est le risque opérationnel n°1 du value bettor, avant même la variance. Les pros gèrent leur longévité : mises arrondies à des montants « naturels », éviter de ne parier que des marchés obscurs à forte value, éviter de frapper instantanément les lignes périmées, et répartir sur plusieurs books quand c'est possible.

**Pour l'app :** Ajouter un mode « discrétion » : arrondir les mises Kelly suggérées à des montants ronds en FCFA (500/1000/2000), alerter quand le profil de paris devient trop typé sharp (ex. 100% de paris sur totaux équipe de petites ligues), et ajouter un champ dans le suivi pour noter les baisses de limite constatées chez Betclic CI afin de mesurer la durée de vie du compte. Sensibiliser l'utilisateur dans l'UI : le but est de durer, pas de maximiser chaque pari.

Sources : https://www.rebelbetting.com/blog/difference-soft-sharp-bookmakers · https://punter2pro.com/common-mistakes-betting-strategy-system/ · https://www.rebelbetting.com/bookmakers


### Capturer le mouvement des cotes : l'edge durable d'un book soft vient des lignes lentes
**Difficulté : difficile**

La recherche académique montre que les inefficiences par ligue ne sont ni persistantes ni systématiques — parier « les petites ligues » en soi n'est pas un edge durable. En revanche, les books softs réagissent délibérément plus lentement aux mouvements du marché (blessures, steam, argent sharp), surtout sur petites ligues et marchés secondaires : la fenêtre de value se situe entre le mouvement du marché de référence et la mise à jour de la ligne soft, typiquement proche du coup d'envoi.

**Pour l'app :** Stocker des snapshots horodatés des cotes Betclic dans SQLite (ouverture → J-1 → H-1) pour chaque match suivi. Signaler les lignes « gelées » : quand la référence sharp ou le modèle a bougé mais que Betclic n'a pas ajusté, c'est le signal de value le plus fiable pour un book soft. Bonus : ces snapshots fournissent gratuitement la cote de clôture nécessaire au calcul du CLV (finding n°1) — les deux fonctionnalités partagent la même infrastructure de scraping périodique.

Sources : https://journals.sagepub.com/doi/10.1177/15270025231204997 · https://idenfy.com/blog/arbitrage-sports-betting/ · https://arxiv.org/pdf/1910.08858


## Axe 3 — Sources de données

*Trois gains immédiats à 0$ : (1) football-data.co.uk fournit des CSV gratuits avec cotes de clôture, buts par mi-temps et tirs cadrés sur 25+ saisons et 27 pays — de quoi backtester le modèle Poisson et recalibrer la répartition 45/55 et le ratio tirs cadrés 3.3× ; (2) les 500 crédits/mois dormants de The Odds API suffisent largement à capturer des cotes quasi-clôture (Pinnacle inclus, 1-2 crédits par ligue) pour mesurer le CLV, les snapshots historiques étant payants ; (3) la lib Python soccerdata (maintenue, avril 2026) débloque Understat/Sofascore/ClubElo sans scraping maison, et ClubElo offre un vrai Elo gratuit en CSV pour dé-circulariser le blend 1X2. Côté payant, le premier plan utile est API-Football Pro à 19$/mois (saison courante + blessés + 1200 compétitions dont l'Afrique), légèrement au-dessus du budget ; football-data.org gratuit est à écarter (12 compétitions, sans cotes ni compos). Les ligues africaines de Betclic CI restent le point dur : aucun CSV/API gratuit propre, seulement un patchwork Sofascore/FootyStats/TheSportsDB/BetExplorer — ou API-Football payant.*


### football-data.co.uk : backtesting massif gratuit avec cotes de clôture, buts mi-temps et tirs cadrés
**Difficulté : facile**

CSV gratuits sans clé API : 11 pays « main » (Angleterre, Écosse, Allemagne, Italie, Espagne, France, Pays-Bas, Belgique, Portugal, Turquie, Grèce, jusqu'à 22 divisions) avec résultats depuis 1993/94 et cotes depuis 2000/01, plus 16 pays « extra » (Argentine, Brésil, Japon, Mexique, etc.) depuis 2012/13 avec cotes de clôture. Champs clés confirmés dans notes.txt : FTHG/FTAG et HTHG/HTAG (buts par mi-temps), HS/AS et HST/AST (tirs et tirs cadrés), corners, cartons, et cotes 1X2 + O/U 2.5 + Asian Handicap de ~10 bookmakers avec colonnes de clôture suffixées C (B365CH, PSCH, MaxCH, AvgCH). Attention : les cotes Pinnacle sont signalées obsolètes depuis juillet 2025, utiliser Max/Avg closing à la place.

**Pour l'app :** C'est LA source pour backtester le modèle Poisson ancré no-vig sur des milliers de matchs : (1) valider le seuil de value 5% et le poids marché 60% contre les cotes de clôture réelles ; (2) calibrer la répartition mi-temps 45/55 fixe avec les vrais HTHG/HTAG ; (3) remplacer le ratio arbitraire « tirs cadrés ≈ 3.3 × buts attendus » par une régression sur HST/AST réels. Un simple téléchargement pandas.read_csv suffit, zéro scraping.

Sources : https://www.football-data.co.uk/data.php · https://www.football-data.co.uk/notes.txt · https://www.football-data.co.uk/downloadm.php


### The Odds API (clé déjà en poche, 500 crédits/mois inutilisés) : capturer soi-même les cotes quasi-clôture pour le CLV
**Difficulté : facile**

Coût confirmé : GET /odds = [marchés] × [régions] crédits et renvoie TOUS les matchs à venir d'une ligue en un appel ; /sports et /events sont gratuits (0 crédit). Donc 1 appel région=eu, markets=h2h,totals = 2 crédits pour toute une journée de Premier League. Région EU inclut Pinnacle, 1xBet, Betfair, Unibet. Marchés soccer : h2h, totals, spreads, plus btts, double_chance, draw_no_bet et alternate_totals via /events/{id}/odds (par match). Les snapshots historiques (closing odds rétroactives) sont réservés aux plans payants (coût ×10), mais rien n'empêche de poller juste avant le coup d'envoi et de stocker en SQLite. Couverture : grandes ligues EU + secondes divisions + Coupe d'Afrique des Nations ; PAS de championnats domestiques africains.

**Pour l'app :** Avec 500 crédits/mois : poller les ligues du jour 15-30 min avant kickoff (planifier via /events gratuit) ≈ 2-4 crédits par ligue et par capture, largement suffisant pour 5-6 ligues suivies. Permet enfin de mesurer le CLV (cote Betclic prise vs cote Pinnacle de quasi-clôture) — le meilleur indicateur avancé que le modèle détecte de la vraie value, bien avant que le ROI SQLite soit significatif.

Sources : https://the-odds-api.com/liveapi/guides/v4/ · https://the-odds-api.com/ · https://the-odds-api.com/sports-odds-data/betting-markets.html


### xG gratuit en 2026 : passer par la librairie soccerdata plutôt que du scraping maison
**Difficulté : moyen**

La lib Python soccerdata (v1.9.0, release avril 2026, activement maintenue) embarque des scrapers à jour pour Understat, Sofascore, FBref, Football-Data.co.uk, ClubElo, ESPN et WhoScored, avec cache local et rate-limiting. Understat (top 5 ligues + RFPL) embarque toujours ses données xG/npxG/xGChain en JSON dans des balises script — c'est ce « changement de format » qui casse les parsers HTML maison, et soccerdata le gère. FotMob exige depuis oct. 2024 un header signé x-fm-req (issue #742, corrigé par PR #745 mais fragile). FBref renvoie 403 aux clients non-navigateur mais le module soccerdata FBref avec headers/cache passe encore en usage léger. Complément : StatsBomb Open Data (JSON gratuit, événements + xG) pour calibrer historiquement.

**Pour l'app :** Remplacer le scraping xG bloqué (403) par `pip install soccerdata` puis `sd.Understat(leagues=..., seasons=...)` : la maintenance des changements de format est externalisée à une lib communautaire active. Le xG réel par équipe (att/déf, domicile/extérieur) remplacerait avantageusement les stats API-Football de saisons anciennes dans le Poisson, au moins pour les 5 grands championnats — là où l'utilisateur parie le plus gros volume probablement.

Sources : https://github.com/probberechts/soccerdata · https://soccerdata.readthedocs.io/ · https://github.com/probberechts/soccerdata/issues/742 · https://github.com/statsbomb/open-data


### API-Football : le premier plan payant est à 19$/mois (pas 10-15$) et débloque la saison courante
**Difficulté : facile**

Plan gratuit : 100 req/jour, TOUS les endpoints (injuries, sidelined, prédictions, cotes pre-match et live) mais saisons limitées aux anciennes (2021-2023) — exactement la limite actuelle de l'app. Premier plan payant « Pro » : 19$/mois, 7 500 req/jour, saison courante débloquée sur les 1 200+ compétitions (dont championnats africains), prépayé sans auto-renouvellement ni frais cachés. Point important : les cotes API-Football ne sont conservées que 7 jours — inutilisables pour l'historique/CLV, The Odds API et football-data.co.uk restent nécessaires pour ça.

**Pour l'app :** C'est le seul upgrade payant qui débloque d'un coup stats saison courante + blessés + compos pour les ligues africaines et mineures de Betclic CI. Il dépasse de 4$ le budget annoncé (19$ vs 15$ max) : à arbitrer. Alternative à 0$ : rester en free (une fois le compte réactivé) pour le H2H/forme des saisons passées et combler la saison courante avec Sofascore/FootyStats (finding ligues africaines).

Sources : https://www.api-football.com/pricing · https://www.api-football.com/news/post/how-to-get-started-with-api-football-the-complete-beginners-guide · https://sportsapi.com/api-directory/api-football/


### football-data.org gratuit : couverture réelle trop étroite pour cette app, à écarter comme remplaçant
**Difficulté : facile**

Le tier gratuit couvre seulement 12 compétitions majeures (Big 5, CL, etc.) à 10 appels/min, avec scores différés, SANS cotes ni compositions (payant). Les cotes sont un add-on à 15€/mois par-dessus un plan de base, les compos exigent le plan « Deep Data » à 29€/mois. Aucune ligue africaine dans le tier gratuit. Le premier vrai upgrade utile (livescores 12€/mois) n'apporte rien que l'app n'ait déjà.

**Pour l'app :** Constat négatif mais utile : ne pas investir de temps d'intégration ici. Sa seule valeur pour l'app est un filet de secours gratuit pour calendriers/classements des grandes ligues quand API-Football est suspendu — mais soccerdata + football-data.co.uk font déjà mieux gratuitement.

Sources : https://www.football-data.org/pricing · https://www.football-data.org/coverage


### ClubElo : un vrai Elo gratuit en CSV pour remplacer l'« Elo estimé des cotes »
**Difficulté : facile**

API publique gratuite en CSV, sans clé : api.clubelo.com/YYYY-MM-DD (classement Elo complet du jour), api.clubelo.com/{Club} (historique complet d'un club), api.clubelo.com/Fixtures (matchs à venir AVEC probabilités par différence de buts et par score exact). Ratings quotidiens des clubs européens depuis 1939, mis à jour après chaque journée. Accessible en HTTP simple (pas HTTPS) ou via le module soccerdata.ClubElo. Limite : clubs européens uniquement.

**Pour l'app :** Le blend 1X2 actuel utilise un Elo « estimé des cotes » — donc circulaire avec l'ancrage no-vig (la même info marché comptée deux fois). Brancher le vrai Elo ClubElo pour les ligues européennes apporte un signal réellement indépendant du marché dans le blend 60/40, et les probabilités de score exact du endpoint /Fixtures peuvent servir de contrôle croisé du Poisson. Intégration : une requête requests.get + pandas par jour de match.

Sources : http://clubelo.com/API · https://soccerdata.readthedocs.io/en/latest/datasources/ClubElo.html · https://fcpython.com/blog/calling-api-python-requests-visualising-clubelo-data


### Ligues africaines de Betclic CI : aucun CSV/API gratuit propre, patchwork Sofascore + FootyStats + BetExplorer + TheSportsDB
**Difficulté : difficile**

Constat de couverture pour la Ligue 1 ivoirienne et les championnats africains : football-data.co.uk = non ; The Odds API = non (seulement la CAN, clé soccer_africa_cup_of_nations) ; football-data.org gratuit = non. Ce qui couvre réellement : (1) Sofascore a la Ivory Coast Ligue 1 (tournament id 1211) avec classements, stats et compos — atteignable via le scraper Sofascore de soccerdata ; (2) FootyStats publie des stats agrégées gratuites (O/U%, BTTS%, buts/match) pour la Ligue 1 ivoirienne ; (3) BetExplorer (groupe Livesport) archive résultats ET cotes historiques de la ligue — scraping léger toléré mais non permis, à doser ; (4) TheSportsDB (API gratuite, clé de test « 3 ») a la ligue (id 5241) pour calendriers/résultats bruts ; (5) API-Football payant reste le seul agrégat API propre incluant les championnats africains en saison courante.

**Pour l'app :** Pour les matchs africains, le modèle tourne aujourd'hui quasi à l'aveugle. Pipeline réaliste à 0$ : résultats TheSportsDB + moyennes de buts FootyStats pour alimenter un Poisson simplifié, et relever manuellement que la marge Betclic est structurellement plus élevée sur ces marchés (le garde-fou « marge >15% suspecte » va s'y déclencher souvent — c'est normal, pas un bug). Si le budget passe à 19$, API-Football Pro résout ce finding d'un coup.

Sources : https://www.sofascore.com/football/tournament/ivory-coast/ligue-1/1211 · https://footystats.org/ivory-coast/ivory-coast-ligue-1 · https://www.betexplorer.com/football/ivory-coast/ligue-1/ · https://www.thesportsdb.com/league/5241-ivory-coast-ligue-1 · https://the-odds-api.com/sports-odds-data/sports-apis.html


## Priorisation (synthèse)

### Phase 1 — Précision du modèle (gratuit, pur code, ~1 session)
1. Correction **Dixon-Coles** (rho ≈ −0.10) sur les scores faibles → corrige nul, Under, BTTS
2. **Méthode de Shin** pour retirer la marge (pip install shin) → tue les fausses values sur les grosses cotes
3. **Seuil de value progressif** : 5% (cote < 2.5), 8% (2.5-4), 12% (> 4), refus > 6
4. **Time decay** exponentiel (ξ ≈ 0.002/jour) sur les stats d'équipe
5. Mi-temps **par ligue** + ratio tirs cadrés **par équipe** (shrinkage 50/50 vers 3.1)

### Phase 2 — Mesurer la vérité (le plus important à moyen terme)
6. **Backtest massif** sur les CSV gratuits de football-data.co.uk (cotes de clôture,
   buts mi-temps, tirs cadrés réels) → valider seuils, MARKET_WEIGHT, ratios
7. Logger **toutes** les probabilités du modèle → courbes de calibration + Brier vs marché
8. **CLV** via The Odds API (clé déjà disponible, 500 crédits/mois inutilisés) :
   cotes Pinnacle quasi-clôture vs cote Betclic prise
9. Simulateur **Monte Carlo** de variance + plafond Kelly à 2% tant que le CLV
   n'est pas prouvé positif sur 100+ paris

### Phase 3 — Meilleures données
10. **ClubElo** (CSV gratuit) : vrai Elo indépendant à la place de l'Elo estimé des cotes (circulaire)
11. **soccerdata** (pip) : xG Understat maintenu par la communauté, top 5 ligues
12. API-Football : réactivation (support contacté) ; plan 19$/mois = saison courante + blessés (à arbitrer)
13. Ligues africaines : patchwork TheSportsDB + FootyStats (difficile, dernier)

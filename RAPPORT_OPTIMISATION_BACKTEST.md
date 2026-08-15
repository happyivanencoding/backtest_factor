# Rapport d’optimisation du moteur de backtest

Date de mesure : 22 juillet 2026

Version chinoise : [RAPPORT_OPTIMISATION_BACKTEST_ZH.md](RAPPORT_OPTIMISATION_BACKTEST_ZH.md)

## 1. Objectif

Cette phase réduit le temps d’exécution et le pic de mémoire d’un backtest complet, sans modifier les positions ni les résultats financiers. Elle complète la première phase, qui avait déjà rendu les résultats légers, libéré les builders après chaque test et supprimé les scores `Unitary_*` temporaires.

Une seconde passe, mesurée le même jour, optimise spécifiquement les campagnes comportant plusieurs variables et plusieurs horizons. Elle regroupe la création des dérivées, conserve une base mensuelle commune entre les signaux et compacte les identifiants répétés.

Les optimisations portent sur six points :

1. charger uniquement la période utile avec `load_backtest_data(..., start_date, lookback_periods=12)` ;
2. partager `screen` et `returns` entre les builders en lecture seule ;
3. préparer une seule fois chaque mois pour Top et Worst ;
4. vectoriser la neutralisation sectorielle des rangs ;
5. calculer les rendements quotidiens sans convertir les matrices complètes en tables longues ;
6. vérifier strictement l’équivalence des performances, positions et métriques.

## 2. Chargement limité à la période utile

L’appel recommandé est désormais :

```python
screen, returns = load_backtest_data(
    SCREEN_PATH,
    RETURNS_PATH,
    variables=RAW_VARIABLES,
    signal_config=WIDE_CONFIG,
    bench=BENCHMARK,
    start_date=START_DATE,
    lookback_periods=12,
)
```

`start_date` filtre les rendements dès la lecture logique. Pour le screen, la fonction conserve aussi les périodes antérieures nécessaires aux transformations `pct_N`, `diff_N` et `rank_diff_N`. La valeur par défaut `lookback_periods=12` couvre donc les comparaisons à 1, 3, 6 et 12 périodes.

Dans le scénario mesuré, la projection et le filtre produisent :

| Donnée | Avant | Après | Réduction |
|---|---:|---:|---:|
| Lignes du screen | 3 454 342 | 2 255 832 | 34,7 % |
| Lignes de rendements | 5 511 | 4 232 | 23,2 % |
| Colonnes de rendements | 1 288 | 1 164 | 9,6 % |

Seules les colonnes techniques, les variables demandées, leurs éventuels dénominateurs et les membres du benchmark sont chargés.

## 3. Partage des données et du prétraitement Top/Worst

Les builders ne réalisent plus de copie profonde de `screen` et `returns` lors de leur construction. Les deux objets sont partagés en lecture seule. Une copie locale et limitée aux colonnes nécessaires est créée uniquement avant une transformation effective.

La préparation mensuelle commune comprend notamment :

- la fusion des tickers secondaires ;
- le filtre des membres du benchmark ;
- le traitement de la capitalisation ;
- le lissage des pondérations ;
- le calcul des poids sectoriels du benchmark.

Cette préparation est exécutée une fois, puis consommée séparément par Top et Worst. Les règles propres à Top, notamment l’ESG et la liste noire, restent appliquées uniquement à Top.

## 4. Neutralisation sectorielle vectorisée

Le calcul des rangs globaux et sectoriels utilise désormais les opérations vectorisées de pandas. La succession des normalisations reste identique : rang global, normalisation entre 0 et 1, rang au sein du secteur, puis nouvelle normalisation entre 0 et 1.

Les cas de valeurs manquantes et d’égalités conservent le comportement de référence. L’équivalence a été vérifiée sur un marché synthétique déterministe et sur l’historique réel complet.

## 5. Calcul quotidien des performances

L’ancien moteur transformait deux matrices complètes, `returns` et `returns_drift`, en tables longues avec `stack()`, puis les fusionnait avec chaque ligne du portefeuille. Cette étape créait plusieurs tables intermédiaires très volumineuses.

Le nouveau moteur :

1. associe chaque date de rendement à la dernière date de rebalancement avec une recherche vectorisée ;
2. rebase la matrice cumulée sur cette date ;
3. récupère directement le multiplicateur de dérive et le rendement par leurs positions ligne/colonne ;
4. évite les deux `stack()` et les deux fusions correspondantes.

Le calcul reste effectué avant toute fonction de graphique ou d’export.

## 6. Protocole de mesure

Les mesures avant et après utilisent exactement le même scénario :

- processeur Intel Core i9-13900HX et 64 Go de mémoire ;
- données réelles `screen_aggregateCIQ.parquet` et `returns.parquet` ;
- benchmark `STOXX EUROPE 600`, calculé une fois puis injecté ;
- période du 1er janvier 2010 à juillet 2026 ;
- test unitaire de `Revenue 5Y CAGR | level` ;
- portefeuilles Top et Worst à 13 % ;
- points de rupture 2020, 2022 et 2024 ;
- `fill_method="copy"`, graphiques désactivés et builders non conservés.

Le RSS du processus est échantillonné toutes les 10 ms. Le « pic additionnel » correspond au pic RSS de l’étape moins son RSS initial. Les mesures « avant » proviennent d’un snapshot créé avant les modifications de cette phase.

## 7. Résultats mesurés

| Étape | Temps avant | Temps après | Gain temps | Pic additionnel avant | Pic additionnel après | Gain pic |
|---|---:|---:|---:|---:|---:|---:|
| Chargement | 0,888 s | 0,791 s | 10,8 % | 690,1 Mio | 616,8 Mio | 10,6 % |
| Benchmark | 10,187 s | 5,530 s | 45,7 % | 1 544,6 Mio | 1 011,6 Mio | 34,5 % |
| Test unitaire complet | 62,021 s | 36,647 s | **40,9 %** | 1 691,2 Mio | 616,0 Mio | **63,6 %** |

Le temps cumulé des trois étapes passe de **73,10 s à 42,97 s**, soit un gain de **41,2 %**. Le pic RSS absolu maximal observé passe de **2 458,7 Mio à 1 655,5 Mio**, soit une baisse de **32,7 %**.

La mémoire encore retenue après le test unitaire passe de 60,1 Mio à 42,4 Mio, soit une baisse supplémentaire de 29,5 %. Cette valeur s’ajoute au gain de la première phase, qui avait déjà supprimé la rétention des builders complets dans chaque résultat.

Une mesure isolée dépend de la charge de la machine et du cache disque. Les gains les plus robustes sont la disparition des tables longues intermédiaires et la réduction structurelle du nombre de lignes/colonnes chargées.

## 8. Équivalence stricte des résultats

Le contrôle réel complet compare les objets avant et après avec `pandas.testing.assert_frame_equal(..., check_exact=True)`. Tous les contrôles sont exacts :

| Objet contrôlé | Résultat |
|---|---|
| Performance Top, Worst et Benchmark | Exact |
| Positions historiques Top | Exact |
| Positions historiques Worst | Exact |
| Ratios Top/Benchmark, Worst/Benchmark et Top/Worst | Exact |
| Métriques classiques | Exact |
| Métriques par période | Exact |
| Score robuste et métriques robustes | Exact |

Les valeurs principales restent donc :

| Indicateur | Avant | Après |
|---|---:|---:|
| Score de robustesse | -0,7371 | -0,7371 |
| Top/Benchmark | -0,0665 | -0,0665 |
| Top/Worst | 0,0845 | 0,0845 |

La suite automatisée fige en plus des empreintes SHA-256 déterministes pour les performances, les positions Top/Worst, les métriques classiques et les périodes. Elle couvre également le benchmark fourni, le partage des entrées, la préparation mensuelle commune, le cas où `ISIN` est un index parquet, les directions, les compositions et la reconstruction par période.

## 9. Audit de la neutralisation sectorielle historique

L’algorithme historique n’est pas erroné dans son périmètre, mais il faut préciser sa définition. Il transforme d’abord chaque score en rang sectoriel comparable, sélectionne ensuite globalement les meilleurs ou les moins bons rangs, puis ajuste les poids du portefeuille aux poids sectoriels du benchmark. Il garantit donc surtout une neutralité des **poids** après sélection. Il ne garantit pas qu’exactement 13 % des titres de chaque secteur soient sélectionnés.

Sur `Revenue 5Y CAGR` dans le `STOXX EUROPE 600`, l’audit de 209 mois et 3 971 groupes secteur-mois montre :

- aucune ligne du benchmark sans secteur ICB 19 ;
- 572 groupes contenant au moins une égalité de valeur ;
- 19 groupes sans aucune valeur brute valide, tous en avril 2026 ;
- au dernier mois, un taux de sélection sectoriel compris entre 11,1 % et 16,7 % pour une cible globale de 13 % ;
- sur l’historique brut, un mois peut sélectionner des scores manquants lorsque toute la coupe transversale est indisponible.

Les principaux risques sont les suivants :

1. une normalisation min-max renvoie `NaN` lorsqu’un secteur contient moins de deux scores distincts ;
2. `nlargest()` ou `nsmallest()` peut alors compléter la liste avec des lignes manquantes selon leur ordre d’origine ;
3. les égalités au seuil ne disposent pas d’une règle secondaire métier explicite ;
4. le nombre cible est calculé avant certains filtres d’éligibilité, ce qui peut modifier le percentile effectif ;
5. la neutralisation utilise le secteur, mais pas la combinaison région × secteur nécessaire à un benchmark réellement multirégional ;
6. le premier rang global et sa normalisation sont redondants lorsque le rang sectoriel est ensuite recalculé.

La vectorisation livrée conserve volontairement cette définition historique afin de garantir l’équivalence des backtests. Le cas d’un secteur manquant conserve également le rang global, comme dans l’ancienne boucle.

Une future version, activée par une option distincte afin de ne pas casser l’historique, devrait :

1. appliquer les filtres d’éligibilité avant de calculer les quotas ;
2. contrôler la couverture, le nombre de valeurs valides et le nombre de valeurs distinctes par date et groupe ;
3. définir une politique explicite pour les groupes invalides : erreur, conservation du portefeuille précédent ou score neutre ;
4. classer directement dans chaque groupe région × secteur, sans préclassement global ;
5. attribuer un quota par groupe avec la méthode des plus forts restes afin de respecter à la fois le percentile sectoriel et le nombre total ;
6. départager les égalités de manière stable avec une clé secondaire documentée, par exemple capitalisation puis ISIN ;
7. exporter les diagnostics de couverture, quotas, égalités et solutions de repli avec les résultats.

## 10. Profilage des variables dérivées

Le profilage séparé utilise dix variables réelles, les treize dimensions `level`, `pct_N`, `diff_N` et `rank_diff_N`, ainsi que les quatre horizons 1, 3, 6 et 12. La fenêtre contrôlée commence le 1er juillet 2025 et contient 130 072 lignes. Elle produit 120 colonnes dérivées conservées dans `screen`.

Avant cette passe, chaque dimension dérivée reconstruisait et triait sa propre table. Une variable demandant douze dérivées déclenchait donc douze tris. La nouvelle fonction prépare la valeur brute et le rang une fois, trie une seule fois par ISIN et date, puis génère tous les horizons à partir de cette table ordonnée.

| Mesure profilée | Avant | Après | Gain |
|---|---:|---:|---:|
| Temps total | 41,400 s | 16,445 s | **60,3 %** |
| Nombre de tris | 120 | 10 | **91,7 %** |
| Pic additionnel | 383,1 Mio | 383,1 Mio | stable |
| Mémoire retenue | 139,1 Mio | 134,5 Mio | 3,3 % |

Le pic reste stable parce que les 120 colonnes finales doivent volontairement rester disponibles dans `screen`. Le gain porte donc surtout sur les calculs et les tables temporaires. Les 130 colonnes de sortie, comprenant les dix niveaux bruts et les 120 dérivées, ont été comparées avec `check_exact=True` : elles sont strictement identiques à la référence.

## 11. Base mensuelle commune entre les signaux

`monthly_base_cache` conserve, pour chaque date et pour un univers de backtest donné, uniquement les colonnes techniques déjà préparées : date, secteur, poids du benchmark et capitalisation. Le score courant est rattaché à cette base au moment du test. Top et Worst, puis les signaux suivants, réutilisent ainsi la même préparation mensuelle.

Le cache est lié à l’identité de `screen` et se vide automatiquement si un autre objet est fourni. Une valeur `None` le désactive explicitement. Dans le notebook, un dictionnaire partagé dans `RUN_OPTIONS` permet de le conserver entre les appels unitaires, incrémentaux et composites.

Une première version conservait accidentellement les catégories ISIN globales dans chaque table mensuelle et occupait 361,1 Mio. Le cache final utilise un index ISIN local et ne conserve pas SEDOL, qui n’intervient pas dans la sélection mensuelle. Pour 198 mois, il n’occupe plus que **9,5 Mio**, soit une réduction de **97,4 %** par rapport à cette version intermédiaire.

Sur le test réel complet, le premier signal construit le cache en 11,52 s. Le second signal, exécuté avec le même univers et les mêmes options, le réutilise en 10,92 s, soit un gain de 5,2 %. Le gain global est volontairement limité : après les optimisations précédentes, le calcul quotidien des performances et des métriques constitue désormais la majeure partie du temps d’un signal.

## 12. Types compacts pour les identifiants et secteurs

`load_backtest_data(..., compact_dtypes=True)` est activé par défaut. Il utilise :

- un `CategoricalIndex` pour ISIN ;
- le type `category` pour SEDOL et la région ;
- `Int8` nullable pour les codes de supersecteur et d’industrie lorsqu’ils sont entiers et compatibles ;
- les types numériques d’origine pour les variables financières et les rendements, afin de ne perdre aucune précision.

Sur le screen projeté de 2 255 832 lignes et sept colonnes, la mémoire pandas mesurée passe de 485,7 Mio à **72,8 Mio**, soit une baisse de **85,0 %**. Le RSS réellement retenu par l’étape de chargement passe de 555,4 Mio lors de la mesure précédente à 356,8 Mio dans la nouvelle mesure. Le temps de lecture peut légèrement augmenter à cause de la conversion des catégories ; ce coût est payé une seule fois et bénéficie ensuite à toute la campagne.

## 13. Mesure complète et équivalence de la nouvelle passe

La mesure complète conserve le scénario de la section 6. Les résultats suivants comparent la passe précédente, déjà optimisée, à la présente passe :

| Étape | Passe précédente | Nouvelle passe | Évolution |
|---|---:|---:|---:|
| Chargement | 0,791 s | 1,338 s | conversion compacte incluse |
| Pic additionnel du chargement | 616,8 Mio | 535,4 Mio | **-13,2 %** |
| Benchmark | 5,530 s | 3,140 s | **-43,2 %** |
| Pic additionnel du benchmark | 1 011,6 Mio | 811,8 Mio | **-19,8 %** |
| Premier signal complet | 36,647 s | 11,520 s | **-68,6 %** |
| Pic additionnel du premier signal | 616,0 Mio | 487,9 Mio | **-20,8 %** |
| Signal suivant avec cache | — | 10,918 s | base mensuelle réutilisée |

Les temps isolés restent sensibles au cache disque et à la charge de la machine. Les mesures structurelles les plus fiables sont la réduction du nombre de tris, la taille pandas du screen et la taille du cache mensuel.

Après ces modifications, le contrôle sur l’historique réel complet reste exact pour les performances, les positions Top et Worst, les ratios, les métriques classiques, les métriques par période et tous les scalaires du score robuste. La suite automatisée contient désormais 27 tests, complétés par deux sous-tests de protection, et couvre explicitement le tri unique, les types compacts, la réutilisation de la base mensuelle et la parallélisation.

## 14. Comparaison du code avant et après

Les extraits ci-dessous sont volontairement réduits aux instructions qui portent le coût principal. `...` indique uniquement du code inchangé ou non pertinent pour la comparaison. La version « avant » correspond au snapshot antérieur aux optimisations du moteur ; la version « après » correspond au commit `f3447f0`.

### 14.1 Chargement projeté, filtré et compacté

Avant, les colonnes utiles étaient déjà projetées, mais tout l’historique du screen était chargé et les identifiants répétés restaient de type `object` :

```python
screen = pd.read_parquet(
    screen_path,
    columns=requested_columns,
)
returns = pd.read_parquet(
    returns_path,
    columns=return_columns,
)
```

Après, le filtre parquet conserve uniquement la période utile avec son lookback, puis les identifiants et secteurs sont compactés :

```python
screen_start_date = resolved_start_date - pd.DateOffset(
    months=int(lookback_periods),
)
screen_filters = [('Date', '>=', screen_start_date.to_pydatetime())]

screen = pd.read_parquet(
    screen_path,
    columns=requested_columns,
    filters=screen_filters,
)
if compact_dtypes:
    screen = compact_screen_dtypes(screen)

returns = pd.read_parquet(
    returns_path,
    columns=return_columns,
)
returns = returns.loc[returns.index >= resolved_start_date]
```

Effet mesuré : 34,7 % de lignes screen en moins et une taille pandas du screen projeté réduite de 485,7 Mio à 72,8 Mio.

### 14.2 Partage des entrées et préparation mensuelle réutilisable

Avant, chaque builder copiait intégralement les deux grandes tables, puis chaque appel mensuel recopiait encore son screen :

```python
self.screen = copy.deepcopy(screen)
self.returns = copy.deepcopy(returns)

screen = copy.deepcopy(screen_agg_monthly)
```

Après, les builders partagent les entrées en lecture seule. La base mensuelle technique est calculée une fois et le score du signal courant est rattaché lors de la lecture du cache :

```python
self.screen = screen
self.returns = returns
self.monthly_base_cache = monthly_base_cache

cache_key = self._monthly_base_cache_key(raw_date)
if cache_key is not None and cache_key in self.monthly_base_cache:
    preparation = self._preparation_from_monthly_base(
        self.monthly_base_cache[cache_key],
        source_screen,
        list_score_col,
    )
    return self._finalize_sec_list_spot(preparation)
```

Effet mesuré : le cache de 198 mois occupe 9,5 Mio. Un signal suivant sur le même univers passe de 11,52 s, coût du premier signal incluant la construction du cache, à 10,92 s.

### 14.3 Un seul tri par variable pour tous les horizons

Avant, le tri par ISIN et date se trouvait dans la boucle des dimensions. Douze dérivées d’une même variable entraînaient donc douze tris :

```python
for component, column in components:
    if component != 'level':
        ordered = pd.DataFrame({
            '_position': np.arange(len(screen)),
            '_isin': isin_values,
            '_date': pd.to_datetime(screen['Date']).to_numpy(),
            '_value': source_values.to_numpy(),
        }).sort_values(['_isin', '_date'])

        if base_dimension == 'pct':
            filled_values = ordered.groupby('_isin')['_value'].ffill()
            derived = filled_values.groupby(ordered['_isin']).pct_change(
                periods=period,
                fill_method=None,
            )
        else:
            derived = ordered.groupby('_isin')['_value'].diff(period)
        screen[column] = pd.Series(
            derived.to_numpy(),
            index=ordered['_position'],
        ).sort_index().to_numpy()
```

Après, la table est ordonnée avant la boucle ; les groupes, le forward fill et le rang sont préparés une seule fois, puis tous les horizons sont produits ensemble :

```python
ordered = pd.DataFrame(ordered_data).sort_values(['_isin', '_date'])
ordered_groups = ordered.groupby('_isin')
filled_values = (
    ordered_groups['_value'].ffill()
    if any(dimension.startswith('pct_') for dimension in dimensions)
    else None
)

derivatives = {}
for dimension in dimensions:
    base_dimension, period_text = dimension.rsplit('_', 1)
    period = int(period_text)
    if base_dimension == 'pct':
        derived = filled_values.groupby(ordered['_isin']).pct_change(
            periods=period,
            fill_method=None,
        )
    elif base_dimension == 'rank_diff':
        derived = ordered_groups['_rank_value'].diff(periods=period)
    else:
        derived = ordered_groups['_value'].diff(periods=period)
    derivatives[f'{variable}__{dimension}'] = pd.Series(
        derived.to_numpy(),
        index=ordered['_position'],
    ).sort_index().to_numpy()

screen[list(derivatives)] = pd.DataFrame(derivatives, index=screen.index)
```

Effet mesuré : le nombre de tris passe de 120 à 10 dans la campagne profilée et le temps de création des dérivées passe de 41,40 s à 16,45 s.

### 14.4 Calcul quotidien sans tables longues

Avant, deux matrices complètes étaient transformées en tables longues, puis fusionnées avec le portefeuille :

```python
returns_drift_flat = (
    returns_drift.stack().to_frame().reset_index()
)
returns_flat = (
    df_returns.stack().to_frame().reset_index()
)
df_merge = df_merge.merge(
    returns_drift_flat,
    how='left',
    on=[col_date, col_id],
)
df_merge = df_merge.merge(
    returns_flat,
    how='left',
    on=[col_date, col_id],
)
```

Après, les dates et les titres sont convertis en positions, puis les deux valeurs sont lues directement dans les matrices NumPy :

```python
date_positions = returns_drift.index.get_indexer(
    pd.DatetimeIndex(df_merge[col_date]),
)
security_positions = returns_drift.columns.get_indexer(df_merge[col_id])
valid_positions = (date_positions >= 0) & (security_positions >= 0)

drift_values = np.full(len(df_merge), np.nan)
return_values = np.full(len(df_merge), np.nan)
drift_matrix = returns_drift.to_numpy(copy=False)
return_matrix = df_returns.to_numpy(copy=False)
drift_values[valid_positions] = drift_matrix[
    date_positions[valid_positions],
    security_positions[valid_positions],
]
return_values[valid_positions] = return_matrix[
    date_positions[valid_positions],
    security_positions[valid_positions],
]
```

Effet mesuré conjointement avec les autres optimisations de la première passe : le test unitaire complet est passé de 62,02 s à 36,65 s, tandis que son pic additionnel est passé de 1 691,2 Mio à 616,0 Mio. La seconde passe l’a ensuite réduit à 11,52 s et 487,9 Mio.

## 15. Synthèse consolidée des temps avant et après

Le tableau suivant rassemble les trois générations mesurées avec le même historique réel et le même test `Revenue 5Y CAGR | level`. La colonne « initiale » précède la vectorisation du moteur ; la « première passe » correspond aux optimisations décrites dans les sections 2 à 5 ; la version « actuelle » ajoute le tri unique, les types compacts et le cache mensuel.

| Étape | Version initiale | Première passe | Version actuelle | Gain initial → actuel |
|---|---:|---:|---:|---:|
| Chargement | 0,888 s | 0,791 s | 1,338 s | -50,7 % |
| Benchmark | 10,187 s | 5,530 s | 3,140 s | **69,2 %** |
| Premier signal complet | 62,021 s | 36,647 s | 11,520 s | **81,4 %** |
| **Total mesuré** | **73,096 s** | **42,968 s** | **15,998 s** | **78,1 %** |

Le chargement actuel est plus lent de 0,45 s que le chargement initial parce qu’il inclut la conversion vers les types compacts. Ce coût unique évite ensuite de conserver plus de 400 Mio supplémentaires pendant toute la campagne. Il ne faut donc pas l’interpréter isolément comme une régression globale.

La lecture chronologique est la suivante :

1. la première passe réduit le total de 73,10 s à 42,97 s, soit 41,2 % ;
2. la seconde passe réduit encore ce total à 16,00 s, soit 62,8 % par rapport à la première passe ;
3. le gain cumulé entre la version initiale et la version actuelle atteint 78,1 %, soit un temps divisé par environ 4,6.

La mémoire suit la même tendance :

| Mesure mémoire | Version initiale | Première passe | Version actuelle | Gain initial → actuel |
|---|---:|---:|---:|---:|
| Pic additionnel du chargement | 690,1 Mio | 616,8 Mio | 535,4 Mio | 22,4 % |
| Pic additionnel du benchmark | 1 544,6 Mio | 1 011,6 Mio | 811,8 Mio | 47,4 % |
| Pic additionnel du premier signal | 1 691,2 Mio | 616,0 Mio | 487,9 Mio | **71,2 %** |
| Pic RSS absolu maximal | 2 458,7 Mio | 1 655,5 Mio | 1 261,8 Mio | **48,7 %** |

Le profiling des dix variables et treize dimensions, présenté en section 10, utilise une fenêtre distincte de 130 072 lignes. Ses 41,40 s avant et 16,45 s après ne doivent donc pas être additionnées au total de 15,998 s ci-dessus : il s’agit d’une mesure ciblée destinée à isoler le coût de fabrication des colonnes dérivées.

## 16. Parallélisation des signaux indépendants

Les fonctions `test_unitary_signals()`, `test_incremental_signals()` et `test_composite_signals()` acceptent désormais `n_jobs`. La valeur par défaut `n_jobs=1` conserve strictement l’exécution séquentielle historique. Une valeur supérieure à 1 distribue les backtests de signaux indépendants entre plusieurs processus.

La parallélisation reste placée au niveau où les tâches sont réellement indépendantes :

- les dérivées d’une variable sont toujours générées ensemble après un seul tri ;
- Top et Worst restent dans la même tâche et partagent la préparation mensuelle ;
- le parent calcule le benchmark une seule fois ;
- le premier signal construit le cache mensuel lorsqu’il est vide ;
- les workers reçoivent ensuite le screen compact, les rendements, le benchmark et une copie prête à lire du cache ;
- les résultats sont réunis dans l’ordre d’origine, indépendamment de l’ordre réel de fin des processus ;
- les figures Plotly sont renvoyées au processus principal et affichées dans cet ordre.

Le changement principal peut être résumé ainsi :

Avant :

```python
results = {}
for variable in signal_config:
    results[variable] = run_top_worst_backtest(...)
```

Après :

```python
with ProcessPoolExecutor(
    max_workers=worker_count,
    initializer=_init_worker_context,
    initargs=(screen, returns, list_noire_path, execution_options),
) as executor:
    ordered_results = list(
        executor.map(_run_worker_signal, signal_tasks)
    )
```

Le scénario réel de validation utilise quatre signaux `level` sur tout l’historique 2010-2026 : `Revenue 5Y CAGR`, `FCF Conversion`, `Gross Margin` et `Ebitda Margin`. Le benchmark, les points de rupture et toutes les options sont identiques entre les deux exécutions.

| Lot de quatre signaux | Temps | Accélération | Gain temps |
|---|---:|---:|---:|
| `n_jobs=1` | 39,039 s | 1,00× | — |
| `n_jobs=4` | 22,332 s | **1,75×** | **42,8 %** |

Le premier signal reste séquentiel afin de préchauffer le cache mensuel ; les trois signaux suivants s’exécutent simultanément. La mesure parallèle inclut la création des processus Windows et la sérialisation des données compactes vers chaque worker.

Un second benchmark, plus proche d’une campagne unitaire courante, utilise dix variables et les quatre dimensions par défaut `level`, `pct_1`, `diff_1` et `rank_diff_1`, soit quarante backtests complets sans figure :

| Lot unitaire de quarante tests | Temps | Accélération de 8 vs 6 | Gain temps de 8 vs 6 |
|---|---:|---:|---:|
| `n_jobs=6` | 215,509 s | 1,00× | — |
| `n_jobs=8` | 132,912 s | **1,62×** | **38,3 %** |

Les quarante résultats restent strictement identiques. Pour ce type de campagne unitaire large, `n_jobs=8` est donc le réglage recommandé sur la machine de mesure. Pour un lot plus court, deux à quatre processus évitent de créer plus de workers que de tâches réellement disponibles.

L’équivalence stricte est confirmée pour les performances, les positions Top et Worst, les ratios, les métriques classiques, les métriques par période et la Composition. La suite automatisée vérifie séparément les lots unitaires, incrémentaux et composites, ainsi que le maintien de l’ordre et le retour des figures Plotly. La validation a également été exécutée depuis un véritable kernel Jupyter sous Windows.

Chaque processus possède ses propres tables de travail, même si les entrées ont été compactées. `retain_builders=True` et un `save_path` unique sont donc refusés lorsque `n_jobs > 1` ; les résultats doivent être exportés après la réunion du lot.

## 17. Limites et prochaines optimisations possibles

Le coût restant est principalement concentré dans le calcul quotidien répété pour chaque signal et dans certaines opérations pandas historiques encore basées sur `groupby.apply()` ou sur des boucles de dates.

Les prochaines pistes, à traiter avec le même protocole d’équivalence, sont :

1. remplacer les derniers `groupby.apply()` de sélection et de pondération par des opérations groupées plus directes ;
2. mutualiser, lorsque cela est mathématiquement possible, les étapes quotidiennes communes à plusieurs portefeuilles ;
3. ajouter un benchmark automatisé non bloquant afin de suivre les régressions de temps et de mémoire sans rendre les tests fonctionnels instables ;
4. étudier un cache persistant optionnel uniquement si les campagnes traversant plusieurs redémarrages de kernel le justifient.

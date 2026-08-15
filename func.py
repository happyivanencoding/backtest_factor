"""Outils génériques pour construire et tester des signaux factoriels.

Le dictionnaire SIGNAL_CONFIG ci-dessous est uniquement un exemple de format.
Dans un notebook, définissez votre propre configuration dans une cellule puis
passez-la explicitement aux fonctions de ce module.
"""

import copy
import json
import re
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from plotly.colors import qualitative
from plotly.subplots import make_subplots

from factor_config import (
    COMPARISON_DIMENSIONS,
    DEFAULT_SIGNAL_DIMENSIONS,
    LEGACY_DIMENSION_ALIASES,
    LOWER_IS_BETTER,
    SIGNAL_DIMENSIONS,
    factor_columns,
    signal_options,
)

try:
    from BacktestEngine import PtfBuilder, build_periods_from_breakpoints
except ImportError:
    from Codes.BacktestEngine import PtfBuilder, build_periods_from_breakpoints


GROUP_COLS = [' Benchmark ICB Supersector ', 'Date', 'Exchange Country Region']
DEFAULT_BENCHMARK = 'STOXX EUROPE 600'
DEFAULT_PERCENTILE = 0.13
DEFAULT_START_DATE = '2007-12-01'
CLASSIC_METRIC_NAMES = (
    'total_return', 'annualized_return', 'annualized_volatility',
    'sharpe_ratio', 'max_drawdown', 'sortino_ratio',
    'beta', 'tracking_error', 'information_ratio',
)
CLASSIC_METRIC_COLUMNS = tuple(
    f'{portfolio}_{metric}'
    for portfolio in ('top', 'worst', 'bench')
    for metric in CLASSIC_METRIC_NAMES
)
# 2009 pour la crise financière, 2020 pour la pandémie, 2022 pour le régime inflationniste et 2024 pour la normalisation.
RECOMMENDED_PERIOD_BREAKPOINTS = [2009, 2022]


# Exemple de configuration : un poids strictement positif active la dimension ; zéro la désactive.
SIGNAL_CONFIG = {
    'Quality Avg Percentile': signal_options(level=1.0, pct_1=1.0),
    'Revenue 5Y CAGR': signal_options(level=1.0),
    'Sales Growth FY1 CIQ': signal_options(level=1.0),
    'Ebitda 5Y CAGR': signal_options(level=1.0),
    'EBITDA Growth FY1 CIQ': signal_options(level=1.0),
    'Ebit 5Y CAGR': signal_options(level=1.0),
    'EPS Growth FY1 CIQ': signal_options(level=1.0),
    'SP Est 5Y EPS Gr CIQ': signal_options(level=1.0),
    'CFO 5Y CAGR': signal_options(level=1.0),
    'FCF Conversion': signal_options(level=1.0, diff_1=1.0),
    'Gross Profit 5Y CAGR': signal_options(level=1.0),
    'Const Earning 5Y CAGR': signal_options(diff_1=1.0),
    'Gross Margin': signal_options(level=1.0),
    'Ebitda Margin': signal_options(level=1.0),
    'Cont Op Earning Margin': signal_options(diff_1=1.0),
    'R&D Expense CIQ': signal_options(level=1.0, denominator='Sales'),
    'Capex CIQ': signal_options(level=1.0, denominator='Sales'),
    'Sales FY1': signal_options(level=1.0, denominator='Sales'),
    'Net Debt to Ebit': signal_options(
        higher_is_better=False, diff_1=1.0,
    ),
    'Net Debt to Tot Equity': signal_options(
        higher_is_better=False, level=1.0, diff_1=1.0,
    ),
    'Interest expense CIQ': signal_options(
        higher_is_better=False, level=1.0, denominator='Ebitda',
    ),
}


def _config_variables(signal_config):
    """Retourne les variables et dénominateurs demandés par une configuration."""
    if signal_config is None:
        return []
    if isinstance(signal_config, dict):
        columns = list(signal_config)
        columns.extend(
            options.get('denominator')
            for options in signal_config.values()
            if isinstance(options, dict) and options.get('denominator')
        )
        return list(dict.fromkeys(columns))
    if isinstance(signal_config, (list, tuple, set, pd.Index)):
        return list(dict.fromkeys(signal_config))
    raise TypeError('La configuration doit être un dictionnaire ou une liste de variables.')


def required_screen_columns(variables=None, signal_config=None,
                            bench=DEFAULT_BENCHMARK):
    """Liste les seules colonnes de screen nécessaires à la préparation et au backtest."""
    columns = [
        'Date', 'ISIN', 'Company SEDOL',
        ' Benchmark ICB Supersector ', 'Exchange Country Region',
        f'Weight in {bench}', 'Benchmark Market Value Millions in EUR ',
    ]
    columns.extend([] if variables is None else list(variables))
    columns.extend(_config_variables(signal_config))
    return list(dict.fromkeys(columns))


def _parquet_columns(path):
    """Lit uniquement le schéma d'un parquet, sans charger ses données."""
    try:
        import pyarrow.parquet as parquet
    except ImportError as error:
        raise ImportError(
            'pyarrow est requis pour sélectionner les colonnes parquet avant lecture.'
        ) from error
    return parquet.ParquetFile(path).schema_arrow.names


def compact_screen_dtypes(screen):
    """Compacte les identifiants répétés sans réduire la précision numérique."""
    if screen.index.name == 'ISIN' and not isinstance(
        screen.index, pd.CategoricalIndex,
    ):
        screen.index = pd.CategoricalIndex(screen.index, name='ISIN')
    for column in ('ISIN', 'Company SEDOL', 'Exchange Country Region'):
        if column in screen.columns and not isinstance(
            screen[column].dtype, pd.CategoricalDtype,
        ):
            screen[column] = screen[column].astype('category')
    for column in (
        ' Benchmark ICB Supersector ', ' Benchmark ICB Industry ',
    ):
        if column not in screen.columns:
            continue
        numeric = pd.to_numeric(screen[column], errors='coerce')
        finite = numeric.dropna()
        if finite.empty or not np.equal(finite, np.floor(finite)).all():
            continue
        if finite.min() >= -128 and finite.max() <= 127:
            screen[column] = numeric.astype('Int8')
    return screen


def load_backtest_data(screen_path, returns_path, variables=None, signal_config=None,
                       bench=DEFAULT_BENCHMARK, start_date=None,
                       lookback_periods=12, compact_dtypes=True):
    """Charge les données utiles avec l'historique requis avant le backtest."""
    if lookback_periods < 0:
        raise ValueError('lookback_periods doit être positif ou nul.')
    screen_path = Path(screen_path)
    returns_path = Path(returns_path)
    available_screen_columns = set(_parquet_columns(screen_path))
    requested_columns = required_screen_columns(
        variables=variables, signal_config=signal_config, bench=bench,
    )
    market_cap_column = 'Benchmark Market Value Millions in EUR '
    if market_cap_column not in available_screen_columns:
        requested_columns.remove(market_cap_column)
        requested_columns.append(market_cap_column.rstrip())
    missing_columns = [
        column for column in requested_columns if column not in available_screen_columns
    ]
    if missing_columns:
        raise KeyError(f'Colonnes absentes du screen : {missing_columns}')

    screen_filters = None
    resolved_start_date = None
    if start_date is not None:
        resolved_start_date = pd.Timestamp(start_date)
        screen_start_date = resolved_start_date - pd.DateOffset(
            months=int(lookback_periods),
        )
        screen_filters = [('Date', '>=', screen_start_date.to_pydatetime())]

    screen = pd.read_parquet(
        screen_path, columns=requested_columns, filters=screen_filters,
    )
    if compact_dtypes:
        screen = compact_screen_dtypes(screen)
    weight_column = f'Weight in {bench}'
    benchmark_sedols = (
        screen.loc[screen[weight_column].fillna(0).gt(0), 'Company SEDOL']
        .dropna()
        .astype(str)
        .drop_duplicates()
        .tolist()
    )
    available_return_columns = set(_parquet_columns(returns_path))
    return_columns = [
        sedol for sedol in benchmark_sedols if sedol in available_return_columns
    ]
    if not return_columns:
        raise ValueError('Aucun SEDOL du benchmark n’est disponible dans les rendements.')
    returns = pd.read_parquet(returns_path, columns=return_columns)
    if resolved_start_date is not None:
        returns.index = pd.to_datetime(returns.index)
        returns = returns.loc[returns.index >= resolved_start_date]

    print(
        f'Données chargées : {len(screen.columns)} colonnes screen et '
        f'{len(returns.columns)} colonnes de rendements du benchmark.'
    )
    return screen, returns


def assess_variable_missingness(screen, families=None, variables=None,
                                threshold=None, date_col='Date',
                                bench=DEFAULT_BENCHMARK):
    """Calcule les données manquantes uniquement dans l'univers du benchmark."""
    if families is not None:
        if variables is not None:
            raise ValueError('Indiquez soit families, soit variables, mais pas les deux.')
        families = (families,) if isinstance(families, str) else families
        variables = factor_columns(*families)
    if variables is None:
        raise ValueError('Indiquez au moins une famille ou une liste de variables.')
    variables = (variables,) if isinstance(variables, str) else variables
    variables = list(dict.fromkeys(map(str, variables)))
    if not variables:
        raise ValueError('La liste de variables est vide.')

    threshold_pct = None if threshold is None else float(threshold)
    if threshold_pct is not None:
        threshold_pct *= 100 if threshold_pct <= 1 else 1
        if not 0 <= threshold_pct <= 100:
            raise ValueError('Le seuil doit être compris entre 0 et 1 ou 0 et 100.')

    weight_column = f'Weight in {bench}'
    required_columns = [date_col, weight_column]
    missing_columns = [column for column in required_columns if column not in screen]
    if missing_columns:
        raise KeyError(f'Colonnes requises absentes : {missing_columns}')

    universe_screen = screen.loc[
        pd.to_numeric(screen[weight_column], errors='coerce').fillna(0).gt(0)
    ]
    if universe_screen.empty:
        raise ValueError(f'Aucune observation n’appartient à l’univers {bench}.')
    dates = pd.to_datetime(universe_screen[date_col], errors='coerce')
    valid_dates = dates.notna()
    if not valid_dates.any():
        raise ValueError('Aucune date valide n’est disponible pour mesurer les données manquantes.')

    available_variables = [variable for variable in variables if variable in screen]
    missing_by_date = pd.DataFrame(
        index=pd.Index(sorted(dates.loc[valid_dates].unique()), name=date_col)
    )
    overall_missing = pd.Series(100.0, index=variables)
    if available_variables:
        missing_indicators = universe_screen.loc[valid_dates, available_variables].isna()
        missing_indicators.index = dates.loc[valid_dates].to_numpy()
        missing_indicators.index.name = date_col
        missing_by_date = missing_indicators.groupby(level=0, sort=True).mean()
        overall_missing.loc[available_variables] = missing_indicators.mean().mul(100)
    missing_by_date = missing_by_date.reindex(columns=variables, fill_value=1.0).mul(100)

    summary = pd.DataFrame({
        'variable': variables,
        'missing_pct': overall_missing.to_numpy(),
        'available': [variable in available_variables for variable in variables],
    })
    summary['selected'] = summary['available']
    if threshold_pct is not None:
        summary['selected'] &= summary['missing_pct'].le(threshold_pct)
    summary['selection_reason'] = 'retenue'
    summary.loc[~summary['available'], 'selection_reason'] = 'colonne absente'
    summary.loc[summary['available'] & ~summary['selected'], 'selection_reason'] = (
        'seuil de données manquantes'
    )
    selected = summary.loc[summary['selected'], 'variable'].tolist()
    excluded = summary.loc[~summary['selected'], 'variable'].tolist()
    return {
        'missing_by_date': missing_by_date,
        'summary': summary.sort_values('missing_pct').reset_index(drop=True),
        'selected_variables': selected,
        'excluded_variables': excluded,
        'threshold_pct': threshold_pct,
    }


def plot_variable_missingness(screen, families=None, variables=None,
                              threshold=None, date_col='Date', title=None,
                              show_plot=True, bench=DEFAULT_BENCHMARK):
    """Trace les données manquantes par date et retourne la liste filtrée."""
    assessment = assess_variable_missingness(
        screen=screen,
        families=families,
        variables=variables,
        threshold=threshold,
        date_col=date_col,
        bench=bench,
    )
    missing_by_date = assessment['missing_by_date']
    selected_variables = set(assessment['selected_variables'])
    figure = go.Figure()
    for variable in missing_by_date:
        selected = variable in selected_variables
        figure.add_trace(go.Scatter(
            x=missing_by_date.index,
            y=missing_by_date[variable],
            mode='lines',
            name=f"{variable} | {'retenue' if selected else 'exclue'}",
            line=dict(dash='solid' if selected else 'dot'),
            opacity=1.0 if selected else 0.45,
            hovertemplate=(
                f'{variable}<br>Date=%{{x|%Y-%m-%d}}<br>Données manquantes=%{{y:.1f}}%<extra></extra>'
            ),
        ))
    threshold_pct = assessment['threshold_pct']
    if threshold_pct is not None:
        figure.add_hline(
            y=threshold_pct,
            line_dash='dash',
            line_color='black',
            annotation_text=f'Seuil : {threshold_pct:.1f}%',
            annotation_position='top left',
        )
    if title is None:
        title = 'Données manquantes par variable'
        if families is not None:
            family_label = ', '.join(
                (families,) if isinstance(families, str) else families,
            )
            title = f'{title} | {family_label}'
        title = f'{title} | Univers : {bench}'
    figure.update_layout(
        title=title,
        width=1450,
        height=900,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            x=1.01,
            xanchor='left',
            y=1.0,
            yanchor='top',
            font=dict(size=9),
            title=dict(text='Variables', font=dict(size=10)),
        ),
        margin=dict(r=420, t=90),
        xaxis_title='Date',
        yaxis=dict(title='Données manquantes', ticksuffix='%'),
    )
    if show_plot:
        figure.show()
    return {**assessment, 'figure': figure}


def _backtest_inputs(screen, returns, metric, bench):
    """Réduit les deux tables aux colonnes réellement consommées par le moteur."""
    if not isinstance(screen, pd.DataFrame) or not isinstance(returns, pd.DataFrame):
        raise TypeError('screen et returns doivent être des DataFrames pandas.')
    market_cap_column = 'Benchmark Market Value Millions in EUR '
    source_market_cap = market_cap_column.rstrip()
    if market_cap_column in screen.columns:
        selected_market_cap = market_cap_column
    elif source_market_cap in screen.columns:
        selected_market_cap = source_market_cap
    else:
        raise KeyError(f'Colonne requise absente : {market_cap_column}')

    weight_column = f'Weight in {bench}'
    screen_columns = [
        'Date', 'Company SEDOL', ' Benchmark ICB Supersector ',
        weight_column, selected_market_cap,
    ]
    if metric is not None:
        screen_columns.append(metric)
    if 'ISIN' in screen.columns:
        screen_columns.insert(1, 'ISIN')
    elif screen.index.name != 'ISIN':
        raise KeyError('Colonne ou index ISIN absent du screen.')
    screen_columns = list(dict.fromkeys(screen_columns))
    missing_columns = [column for column in screen_columns if column not in screen.columns]
    if missing_columns:
        raise KeyError(f'Colonnes requises absentes pour le backtest : {missing_columns}')

    benchmark_sedols = set(
        screen.loc[screen[weight_column].fillna(0).gt(0), 'Company SEDOL']
        .dropna()
        .astype(str)
    )
    return_columns = [column for column in returns.columns if str(column) in benchmark_sedols]
    if not return_columns:
        raise ValueError('Aucun rendement ne correspond aux membres du benchmark.')

    slim_screen = screen.loc[:, screen_columns].copy()
    if selected_market_cap != market_cap_column:
        slim_screen.rename(
            columns={selected_market_cap: market_cap_column}, inplace=True,
        )
    slim_returns = returns.loc[:, return_columns].copy()
    return slim_screen, slim_returns


def _screen_for_backtest_metrics(screen, metric_columns, bench):
    """Conserve la structure du screen et les scores nécessaires aux backtests."""
    market_cap_column = 'Benchmark Market Value Millions in EUR '
    source_market_cap = market_cap_column.rstrip()
    if market_cap_column in screen.columns:
        selected_market_cap = market_cap_column
    elif source_market_cap in screen.columns:
        selected_market_cap = source_market_cap
    else:
        raise KeyError(f'Colonne requise absente : {market_cap_column}')

    weight_column = f'Weight in {bench}'
    columns = [
        'Date',
        'Company SEDOL',
        ' Benchmark ICB Supersector ',
        weight_column,
        selected_market_cap,
    ]
    if 'ISIN' in screen.columns:
        columns.insert(1, 'ISIN')
    elif screen.index.name != 'ISIN':
        raise KeyError('Colonne ou index ISIN absent du screen.')
    columns.extend(metric_columns)
    columns = list(dict.fromkeys(columns))
    missing_columns = [column for column in columns if column not in screen.columns]
    if missing_columns:
        raise KeyError(
            f'Colonnes requises absentes pour les scores : {missing_columns}'
        )
    return screen.loc[:, columns].copy()


def calculate_benchmark_performance(screen, returns, bench=DEFAULT_BENCHMARK,
                                    start_date=DEFAULT_START_DATE):
    """Calcule une fois la performance du benchmark destinée aux autres tests."""
    benchmark_screen, benchmark_returns = _backtest_inputs(
        screen, returns, metric=None, bench=bench,
    )
    builder = PtfBuilder(
        benchmark_screen, benchmark_returns,
        ptf_name=f'{bench}_benchmark', bench=bench,
        percentile=DEFAULT_PERCENTILE, esg_exclusion=0,
        liste_noire=None, metrics=None, Top=True,
    )
    builder.start_date = pd.Timestamp(start_date)
    builder.backtest_get_bench_perf(builder.screen, builder.start_date, bench)
    return builder.perf_bench.copy()


def _ensure_monthly_base_cache(backtest_options):
    """Ajoute un cache mensuel local lorsqu'aucun choix explicite n'est fourni."""
    if 'monthly_base_cache' in backtest_options:
        return backtest_options
    return {**backtest_options, 'monthly_base_cache': {}}


def _bind_monthly_base_cache(screen, monthly_base_cache):
    """Lie le cache à un screen précis et écarte toute base devenue obsolète."""
    if monthly_base_cache is None:
        return
    if not isinstance(monthly_base_cache, dict):
        raise TypeError('monthly_base_cache doit être un dictionnaire ou None.')
    source_id = id(screen)
    if monthly_base_cache.get('_source_id') != source_id:
        monthly_base_cache.clear()
        monthly_base_cache['_source_id'] = source_id


def handle_missing_values(df, columns, group_cols=None):
    """Remplace les valeurs manquantes par la médiane du groupe disponible."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    for col in columns:
        if col in df.columns:
            df[col] = df[col].fillna(
                df.groupby(group_cols, observed=False)[col].transform('median')
            )
    return df


def neutralize_score(df, score_col, higher_is_better, group_cols=None):
    """Convertit une variable en rang centile de 0 à 10 dans chaque groupe."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    df[score_col] = (
        df.groupby(group_cols, observed=False)[score_col]
        .rank(pct=True, ascending=higher_is_better) * 10
    )
    return df


def _canonical_dimension(dimension):
    """Convertit les anciens noms sans horizon vers la comparaison à une période."""
    return LEGACY_DIMENSION_ALIASES.get(dimension, dimension)


def _derived_dimension(variable):
    """Identifie la dimension dérivée présente à la fin d'un nom de colonne."""
    dimensions = tuple(COMPARISON_DIMENSIONS) + tuple(LEGACY_DIMENSION_ALIASES)
    for dimension in sorted(dimensions, key=len, reverse=True):
        if str(variable).endswith(f'__{dimension}'):
            return _canonical_dimension(dimension)
    return None


def _raw_variable_name(variable):
    """Retrouve la variable brute derrière une colonne dérivée connue."""
    raw_variable = str(variable)
    dimension = _derived_dimension(raw_variable)
    if dimension:
        suffixes = (f'__{dimension}',) + tuple(
            f'__{legacy}'
            for legacy, canonical in LEGACY_DIMENSION_ALIASES.items()
            if canonical == dimension
        )
        for suffix in suffixes:
            if raw_variable.endswith(suffix):
                raw_variable = raw_variable[:-len(suffix)]
                break
    return raw_variable.split('__over__', 1)[0]


def _default_higher_is_better(variable):
    """Déduit la direction d'une variable à partir du catalogue central."""
    dimension = _derived_dimension(variable)
    if dimension and dimension.startswith('rank_diff_'):
        return True
    return _raw_variable_name(variable) not in LOWER_IS_BETTER


def _resolve_signal_config(screen, signal_config):
    """Conserve les signaux disponibles et complète leur direction par défaut."""
    resolved = {}
    for variable, source_options in signal_config.items():
        if not isinstance(source_options, dict):
            raise TypeError(f'Les options de {variable} doivent former un dictionnaire.')
        options = copy.deepcopy(source_options)
        options.setdefault('higher_is_better', _default_higher_is_better(variable))
        denominator = options.get('denominator')
        if variable not in screen.columns:
            print(f'Avertissement : colonne absente {variable}. Signal ignoré.')
            continue
        if denominator and denominator not in screen.columns:
            print(
                f'Avertissement : dénominateur {denominator} absent pour '
                f'{variable}. Signal ignoré.'
            )
            continue
        resolved[variable] = options
    return resolved


def prepare_signals(screen, signal_config, group_cols=None, copy_data=False):
    """Prépare les variables brutes dans screen, ou dans une copie si demandé."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    prepared = screen.copy() if copy_data else screen
    resolved_config = {}
    prepared_cols = []

    for variable, options in signal_config.items():
        if variable not in prepared.columns:
            print(f'Avertissement : {variable} absent des données. Signal ignoré.')
            continue

        denominator = options.get('denominator')
        prepared_variable = variable
        if denominator:
            if denominator not in prepared.columns:
                print(f'Avertissement : dénominateur {denominator} absent pour {variable}. Signal ignoré.')
                continue
            prepared_variable = f'{variable}__over__{denominator}'
            prepared[prepared_variable] = prepared[variable] / prepared[denominator]

        resolved_config[prepared_variable] = copy.deepcopy(options)
        prepared_cols.append(prepared_variable)

    for column in prepared_cols:
        prepared[column] = prepared[column].replace([np.inf, -np.inf], np.nan)
    return handle_missing_values(prepared, prepared_cols, group_cols), resolved_config


def _component_weight(options, dimension):
    """Retourne un poids positif ; zéro, une valeur absente ou négative désactive la dimension."""
    dimension = _canonical_dimension(dimension)
    weight_key = f'weight_{dimension}'
    if weight_key in options:
        weight = options[weight_key]
    else:
        base_dimension, period = (
            dimension.rsplit('_', 1)
            if dimension != 'level' else (None, None)
        )
        legacy_key = f'weight_{base_dimension}' if period == '1' else None
        weight = options.get(legacy_key, 0.0) if legacy_key else 0.0
    try:
        weight = float(weight)
    except (TypeError, ValueError) as error:
        raise ValueError(
            f'Le poids {dimension} doit être numérique : {weight}'
        ) from error
    return weight if pd.notna(weight) and weight > 0 else 0.0


def prepare_signal_derivatives(screen, variable, options, dimensions,
                               group_cols=None):
    """Génère tous les horizons demandés après un tri unique par titre et date."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    dimensions = tuple(dict.fromkeys(
        _canonical_dimension(dimension) for dimension in dimensions
        if _canonical_dimension(dimension) != 'level'
    ))
    unknown_dimensions = set(dimensions) - set(COMPARISON_DIMENSIONS)
    if unknown_dimensions:
        raise ValueError(f'Dimensions dérivées inconnues : {sorted(unknown_dimensions)}')
    if not dimensions:
        return screen

    isin_values = (
        screen['ISIN'].to_numpy()
        if 'ISIN' in screen.columns else screen.index.to_numpy()
    )
    ordered_data = {
        '_position': np.arange(len(screen)),
        '_isin': isin_values,
        '_date': pd.to_datetime(screen['Date']).to_numpy(),
        '_value': screen[variable].to_numpy(),
    }
    if any(dimension.startswith('rank_diff_') for dimension in dimensions):
        ordered_data['_rank_value'] = (
            screen.groupby(group_cols, observed=False)[variable]
            .rank(pct=True, ascending=options['higher_is_better']) * 10
        ).to_numpy()
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
                periods=period, fill_method=None,
            )
        elif base_dimension == 'rank_diff':
            derived = ordered_groups['_rank_value'].diff(periods=period)
        else:
            derived = ordered_groups['_value'].diff(periods=period)
        derivatives[f'{variable}__{dimension}'] = pd.Series(
            derived.to_numpy(), index=ordered['_position'],
        ).sort_index().to_numpy()

    screen[list(derivatives)] = pd.DataFrame(
        derivatives, index=screen.index,
    )
    # Regroupe les blocs après l'ajout massif pour éviter la fragmentation pandas.
    screen._consolidate_inplace()
    return screen


def build_signal_component(screen, variable, options, group_cols=None,
                           keep_derived_columns=True,
                           precomputed_dimensions=None):
    """Construit le niveau et les variations explicites à 1, 3, 6 ou 12 périodes."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    contribution = pd.Series(0.0, index=screen.index)
    precomputed_dimensions = set(precomputed_dimensions or ())

    components = [('level', variable)] + [
        (dimension, f'{variable}__{dimension}')
        for dimension in COMPARISON_DIMENSIONS
    ]
    active_dimensions = [
        component for component, _ in components
        if component != 'level'
        and _component_weight(options, component) > 0
        and component not in precomputed_dimensions
    ]
    screen = prepare_signal_derivatives(
        screen, variable, options, active_dimensions, group_cols,
    )

    for component, column in components:
        weight = _component_weight(options, component)
        if weight == 0:
            continue

        screen[column] = screen[column].replace([np.inf, -np.inf], np.nan)
        screen = handle_missing_values(screen, [column], group_cols)
        score_col = f'{column}__score'
        screen[score_col] = screen[column]
        component_direction = (
            True if component.startswith('rank_diff_')
            else options['higher_is_better']
        )
        screen = neutralize_score(screen, score_col, component_direction, group_cols)
        contribution = contribution.add(screen[score_col] * weight, fill_value=0.0)

        if component != 'level' and not keep_derived_columns:
            screen.drop(columns=[column], inplace=True)
        screen.drop(columns=[score_col], inplace=True)

    return screen, contribution


def calculate_composite_score(screen, score_col, signal_config, group_cols=None,
                              copy_data=False, keep_derived_columns=True,
                              signals_prepared=False,
                              precomputed_derivatives=None):
    """Agrège les signaux et conserve par défaut les variables dérivées dans screen."""
    group_cols = GROUP_COLS if group_cols is None else group_cols
    signal_config = _resolve_signal_config(screen, signal_config)
    if signals_prepared:
        prepared = screen.copy() if copy_data else screen
        resolved_config = copy.deepcopy(signal_config)
    else:
        prepared, resolved_config = prepare_signals(
            screen, signal_config, group_cols, copy_data=copy_data,
        )
    precomputed_derivatives = precomputed_derivatives or {}
    total_score = pd.Series(0.0, index=prepared.index)
    active_signals = 0

    for variable, options in resolved_config.items():
        if not any(
            _component_weight(options, component) > 0
            for component in SIGNAL_DIMENSIONS
        ):
            continue
        prepared, contribution = build_signal_component(
            prepared, variable, options, group_cols,
            keep_derived_columns=keep_derived_columns,
            precomputed_dimensions=precomputed_derivatives.get(variable),
        )
        total_score = total_score.add(contribution, fill_value=0.0)
        active_signals += 1

    if not active_signals:
        raise ValueError('Aucun signal actif et disponible n’a été fourni.')

    prepared[score_col] = total_score
    return neutralize_score(prepared, score_col, higher_is_better=True, group_cols=group_cols)


def describe_signal_config(signal_config, role='signal'):
    """Convertit une configuration de signaux en composition longue et explicite."""
    components = []
    for raw_variable, options in signal_config.items():
        denominator = options.get('denominator')
        prepared_variable = (
            f'{raw_variable}__over__{denominator}' if denominator else raw_variable
        )
        for dimension in SIGNAL_DIMENSIONS:
            weight = _component_weight(options, dimension)
            if weight == 0:
                continue
            derived_variable = (
                prepared_variable if dimension == 'level'
                else f'{prepared_variable}__{dimension}'
            )
            components.append({
                'role': role,
                'raw_variable': raw_variable,
                'prepared_variable': prepared_variable,
                'derived_variable': derived_variable,
                'denominator': denominator,
                'dimension': dimension,
                'higher_is_better': (
                    True if dimension.startswith('rank_diff_')
                    else options.get('higher_is_better')
                ),
                'source_higher_is_better': options.get('higher_is_better'),
                'weight': weight,
            })
    return components


def summarize_component_weights(components):
    """Regroupe les poids actifs par variable brute et par dimension."""
    summary = {}
    total_absolute_weight = sum(abs(float(item.get('weight', 0.0))) for item in components)
    for component in components:
        raw_variable = component.get('raw_variable')
        weight = float(component.get('weight', 0.0))
        variable = summary.setdefault(raw_variable, {
            **{f'weight_{dimension}': 0.0 for dimension in SIGNAL_DIMENSIONS},
            'total_weight': 0.0,
            'absolute_weight': 0.0,
            'absolute_weight_share': 0.0,
        })
        variable[f"weight_{component.get('dimension')}"] = weight
        variable['total_weight'] += weight
        variable['absolute_weight'] += abs(weight)
    if total_absolute_weight:
        for variable in summary.values():
            variable['absolute_weight_share'] = (
                variable['absolute_weight'] / total_absolute_weight
            )
    return summary


def run_top_worst_backtest(screen, returns, metric, list_noire_path, bench=DEFAULT_BENCHMARK,
                           percentile=DEFAULT_PERCENTILE, show_plot=True,
                           save_path=None, metadata=None, period_breakpoints=None,
                           build_figure=True, bench_perf=None,
                           start_date=DEFAULT_START_DATE, freq_rebal=1,
                           fill_method='copy', retain_builders=False,
                           monthly_base_cache=None):
    """Exécute Top/Worst et ne conserve les builders que sur demande."""
    _bind_monthly_base_cache(screen, monthly_base_cache)
    backtest_screen, backtest_returns = _backtest_inputs(
        screen, returns, metric=metric, bench=bench,
    )
    resolved_breakpoints = list(
        RECOMMENDED_PERIOD_BREAKPOINTS
        if period_breakpoints is None else period_breakpoints
    )
    builder_top = PtfBuilder(
        backtest_screen, backtest_returns, ptf_name=f'{metric}_top', bench=bench,
        percentile=percentile, esg_exclusion=0, liste_noire=list_noire_path,
        metrics=metric, Top=True, bench_perf=bench_perf,
        monthly_base_cache=monthly_base_cache,
    )
    builder_worst = PtfBuilder(
        backtest_screen, backtest_returns, ptf_name=f'{metric}_worst', bench=bench,
        percentile=percentile, esg_exclusion=0, liste_noire=list_noire_path,
        metrics=metric, Top=False, bench_perf=bench_perf,
        monthly_base_cache=monthly_base_cache,
    )

    for builder in (builder_top, builder_worst):
        builder.start_date = pd.Timestamp(start_date)
        builder.freq_rebal = freq_rebal
        builder.fill_method = fill_method

    builder_top.generic_histo_seclists_pair(
        builder_bottom=builder_worst,
        start_date=pd.Timestamp(start_date),
        freq_rebal=freq_rebal,
        fill_method=fill_method,
    )
    comparison = builder_top.calculate_top_vs_bottom_results(
        builder_bottom=builder_worst,
        period_breakpoints=resolved_breakpoints,
    )
    print(
        f"Résultats pour {metric} : "
        f"score de robustesse {comparison['robust_score']:.4f}, "
        f"Top/Bench {comparison['top_bench_ratio']:.4f}, "
        f"Top/Worst {comparison['top_worst_ratio']:.4f}"
    )
    should_build_figure = build_figure or show_plot or save_path is not None
    comparison['figure'] = (
        builder_top.plot_top_vs_bottom_results(
            result=comparison,
            title=f'Analyse factorielle : {metric}',
            save_path=save_path,
            show_plot=show_plot,
        )
        if should_build_figure else None
    )
    result = {
        'top_holdings': builder_top.sec_list_historical.copy(deep=True),
        'worst_holdings': builder_worst.sec_list_historical.copy(deep=True),
        'metadata': {
            'metric': metric,
            'benchmark': bench,
            'percentile': percentile,
            'start_date': str(pd.Timestamp(start_date).date()),
            'frequency_rebalancing': freq_rebal,
            'fill_method': fill_method,
            'period_breakpoints': resolved_breakpoints,
            'benchmark_performance_provided': bench_perf is not None,
            'builders_retained': bool(retain_builders),
            'components': [],
        },
    }
    if retain_builders:
        result.update({
            'top_builder': builder_top,
            'worst_builder': builder_worst,
        })
    if metadata:
        result['metadata'].update(copy.deepcopy(metadata))
    if isinstance(comparison, dict):
        result.update(comparison)
    components = copy.deepcopy(result['metadata'].get('components', []))
    result['composition'] = pd.DataFrame(components)
    result['raw_variables'] = list(dict.fromkeys(
        component.get('raw_variable') for component in components
    ))
    result['raw_variable_weights'] = summarize_component_weights(components)
    return result


_WORKER_CONTEXT = None


def _run_one_signal(screen, returns, task, list_noire_path, backtest_options):
    """Ajoute éventuellement un score pré-calculé puis exécute son backtest."""
    metric = task['metric']
    score_specification = task.get('score_specification')
    signal_screen = screen
    metric_values = task.get('metric_values')
    if metric_values is not None:
        if len(metric_values) != len(signal_screen):
            raise ValueError(
                f'La longueur du score {metric} ne correspond pas au screen.'
            )
        signal_screen[metric] = metric_values
    if score_specification is not None:
        signal_screen = calculate_composite_score(
            screen,
            metric,
            score_specification['signal_config'],
            keep_derived_columns=score_specification.get(
                'keep_derived_columns', True,
            ),
            signals_prepared=score_specification.get(
                'signals_prepared', False,
            ),
            precomputed_derivatives=score_specification.get(
                'precomputed_derivatives',
            ),
        )
    try:
        return run_top_worst_backtest(
            signal_screen,
            returns,
            metric,
            list_noire_path,
            metadata=task.get('metadata'),
            **backtest_options,
        )
    finally:
        if (
            (task.get('drop_metric') or metric_values is not None)
            and metric in signal_screen.columns
        ):
            signal_screen.drop(columns=[metric], inplace=True)


def _disable_parallel_progress(iterable, **kwargs):
    """Supprime les barres mensuelles concurrentes dans les workers."""
    return iterable


def _init_worker_context(screen, returns, list_noire_path, backtest_options):
    """Installe une seule copie des données dans chaque processus de travail."""
    global _WORKER_CONTEXT
    import tqdm as tqdm_module
    tqdm_module.tqdm = _disable_parallel_progress
    worker_options = dict(backtest_options)
    monthly_base_cache = worker_options.get('monthly_base_cache')
    if isinstance(monthly_base_cache, dict):
        monthly_base_cache['_source_id'] = id(screen)
    _WORKER_CONTEXT = {
        'screen': screen,
        'returns': returns,
        'list_noire_path': list_noire_path,
        'backtest_options': worker_options,
    }


def _run_worker_signal(task):
    """Exécute une tâche avec les données initialisées dans le processus."""
    context = _WORKER_CONTEXT
    if context is None:
        raise RuntimeError('Le contexte parallèle du backtest n’est pas initialisé.')
    return _run_one_signal(
        context['screen'],
        context['returns'],
        task,
        context['list_noire_path'],
        context['backtest_options'],
    )


def _run_sequential_signals(screen, returns, tasks, list_noire_path,
                            backtest_options):
    """Exécute les signaux dans le processus appelant."""
    return [
        _run_one_signal(
            screen, returns, task, list_noire_path, backtest_options,
        )
        for task in tasks
    ]


def _show_signal_figures(results):
    """Affiche les figures après la fin des processus de travail."""
    for result in results:
        figure = result.get('figure')
        if figure is not None:
            figure.show()


def _run_parallel_signals(screen, returns, tasks, list_noire_path,
                          backtest_options, worker_count):
    """Exécute les signaux dans plusieurs processus et garde leur ordre."""

    execution_options = dict(backtest_options)
    show_plot = bool(execution_options.get('show_plot', True))
    execution_options['show_plot'] = False
    execution_options['build_figure'] = bool(
        execution_options.get('build_figure', True) or show_plot
    )
    monthly_base_cache = execution_options.get('monthly_base_cache')
    _bind_monthly_base_cache(screen, monthly_base_cache)
    cache_is_ready = (
        monthly_base_cache is None
        or any(key != '_source_id' for key in monthly_base_cache)
    )

    completed_results = []
    remaining_tasks = list(tasks)
    if not cache_is_ready:
        completed_results.append(
            _run_one_signal(
                screen,
                returns,
                remaining_tasks.pop(0),
                list_noire_path,
                execution_options,
            )
        )

    if remaining_tasks:
        actual_worker_count = min(worker_count, len(remaining_tasks))
        print(
            f'Exécution parallèle de {len(remaining_tasks)} signaux '
            f'avec {actual_worker_count} processus.'
        )
        with ProcessPoolExecutor(
            max_workers=actual_worker_count,
            initializer=_init_worker_context,
            initargs=(
                screen,
                returns,
                list_noire_path,
                execution_options,
            ),
        ) as executor:
            completed_results.extend(
                executor.map(_run_worker_signal, remaining_tasks)
            )
        print('Exécution parallèle terminée.')

    if show_plot:
        _show_signal_figures(completed_results)
    return completed_results


def _run_signal_tasks(screen, returns, tasks, list_noire_path,
                      backtest_options, n_jobs):
    """Exécute les signaux séquentiellement ou par processus."""
    if not tasks:
        return []
    worker_count = min(n_jobs, len(tasks))
    if worker_count == 1:
        return _run_sequential_signals(
            screen, returns, tasks, list_noire_path, backtest_options,
        )
    return _run_parallel_signals(
        screen,
        returns,
        tasks,
        list_noire_path,
        backtest_options,
        worker_count,
    )


def test_unitary_signals(screen, returns, signal_config, list_noire_path,
                         dimensions=DEFAULT_SIGNAL_DIMENSIONS,
                         n_jobs=1, **backtest_options):
    """Teste séparément les dimensions et retourne un lot standardisé."""
    backtest_options = _ensure_monthly_base_cache(backtest_options)
    if not isinstance(signal_config, dict):
        signal_config = {
            variable: {'higher_is_better': _default_higher_is_better(variable)}
            for variable in _config_variables(signal_config)
        }
    signal_config = _resolve_signal_config(screen, signal_config)
    results = {}
    working_screen = screen
    requested_dimensions = tuple(dict.fromkeys(
        _canonical_dimension(dimension) for dimension in dimensions
    ))
    unknown_dimensions = set(requested_dimensions) - set(SIGNAL_DIMENSIONS)
    if unknown_dimensions:
        raise ValueError(f'Dimensions inconnues : {sorted(unknown_dimensions)}')
    dimension_options = {
        dimension: f'weight_{dimension}' for dimension in SIGNAL_DIMENSIONS
    }
    tasks = []

    for variable, options in signal_config.items():
        working_screen, prepared_config = prepare_signals(
            working_screen, {variable: options}, copy_data=False,
        )
        if not prepared_config:
            continue
        prepared_variable, prepared_options = next(iter(prepared_config.items()))
        scoring_options = copy.deepcopy(prepared_options)
        scoring_options.pop('denominator', None)
        derived_dimensions = tuple(
            dimension for dimension in requested_dimensions
            if dimension != 'level'
        )
        working_screen = prepare_signal_derivatives(
            working_screen,
            prepared_variable,
            scoring_options,
            derived_dimensions,
        )
        for label, weight_key in dimension_options.items():
            if label not in requested_dimensions:
                continue

            unitary_options = copy.deepcopy(options)
            for dimension in SIGNAL_DIMENSIONS:
                unitary_options[f'weight_{dimension}'] = 0.0
            unitary_options[weight_key] = 1.0

            metric = f'Unitary_{label}_{variable}'
            print(f'Test de signal unitaire : {variable} | {label}')
            internal_options = copy.deepcopy(scoring_options)
            for dimension in SIGNAL_DIMENSIONS:
                internal_options[f'weight_{dimension}'] = 0.0
            internal_options[weight_key] = 1.0
            result_name = f'{variable} | {label}'
            tasks.append({
                'name': result_name,
                'metric': metric,
                'score_specification': {
                    'signal_config': {prepared_variable: internal_options},
                    'signals_prepared': True,
                    'precomputed_derivatives': {
                        prepared_variable: derived_dimensions,
                    },
                },
                'drop_metric': True,
                'metadata': {
                    'test_type': 'unitary',
                    'test_name': f'{variable} | {label}',
                    'components': describe_signal_config(
                        {variable: unitary_options}, role='unitary',
                    ),
                },
            })

    task_results = _run_signal_tasks(
        working_screen,
        returns,
        tasks,
        list_noire_path,
        backtest_options,
        n_jobs,
    )
    results.update({
        task['name']: result
        for task, result in zip(tasks, task_results)
    })
    return {'screen': working_screen, 'results': results}


def test_incremental_signals(screen, returns, baseline_config, candidate_config,
                             list_noire_path, n_jobs=1, **backtest_options):
    """Compare une base et ses candidats après un calcul groupé des scores."""
    backtest_options = _ensure_monthly_base_cache(backtest_options)
    baseline_config = _resolve_signal_config(screen, baseline_config)
    results = {}
    tasks = []
    baseline_metric = 'Score_Baseline'
    scored_screen = screen.copy()
    scored_screen = calculate_composite_score(
        scored_screen,
        baseline_metric,
        baseline_config,
        keep_derived_columns=False,
    )
    tasks.append({
        'name': 'Baseline',
        'metric': baseline_metric,
        'metric_values': scored_screen[baseline_metric].to_numpy(copy=True),
        'metadata': {
            'test_type': 'incremental_baseline',
            'test_name': 'Baseline',
            'components': describe_signal_config(baseline_config, role='baseline'),
        },
    })
    scored_screen.drop(columns=[baseline_metric], inplace=True)

    for variable, options in candidate_config.items():
        if variable in baseline_config:
            print(f'Avertissement : {variable} appartient déjà à la base. Signal ignoré.')
            continue

        resolved_candidate = _resolve_signal_config(screen, {variable: options})
        if not resolved_candidate:
            continue
        incremental_config = copy.deepcopy(baseline_config)
        incremental_config.update(resolved_candidate)
        metric = f'Score_Incremental_{variable}'
        print(f'Test incrémental : {variable}')
        scored_screen = calculate_composite_score(
            scored_screen,
            metric,
            incremental_config,
            keep_derived_columns=False,
        )
        tasks.append({
            'name': variable,
            'metric': metric,
            'metric_values': scored_screen[metric].to_numpy(copy=True),
            'metadata': {
                'test_type': 'incremental_candidate',
                'test_name': variable,
                'components': (
                    describe_signal_config(baseline_config, role='baseline')
                    + describe_signal_config(resolved_candidate, role='candidate')
                ),
            },
        })
        scored_screen.drop(columns=[metric], inplace=True)

    backtest_screen = _screen_for_backtest_metrics(
        scored_screen,
        [],
        backtest_options.get('bench', DEFAULT_BENCHMARK),
    )
    del scored_screen
    task_results = _run_signal_tasks(
        backtest_screen,
        returns,
        tasks,
        list_noire_path,
        backtest_options,
        n_jobs,
    )
    results.update({
        task['name']: result
        for task, result in zip(tasks, task_results)
    })
    return {'screen': screen, 'results': results}


def test_composite_signal(screen, returns, score_col, signal_config,
                          list_noire_path, test_name=None, n_jobs=1,
                          **backtest_options):
    """Construit un seul composite et retourne un lot standardisé."""
    backtest_options = _ensure_monthly_base_cache(backtest_options)
    signal_config = _resolve_signal_config(screen, signal_config)
    scored_screen = calculate_composite_score(screen, score_col, signal_config)
    result_name = test_name or score_col
    task = {
        'name': result_name,
        'metric': score_col,
        'metadata': {
            'test_type': 'composite',
            'test_name': result_name,
            'components': describe_signal_config(signal_config, role='composite'),
        },
    }
    backtest_result = _run_signal_tasks(
        scored_screen,
        returns,
        [task],
        list_noire_path,
        backtest_options,
        n_jobs,
    )[0]
    return {'screen': scored_screen, 'results': {result_name: backtest_result}}


def test_composite_signals(screen, returns, composite_configs, list_noire_path,
                           score_prefix='Score_Composite', n_jobs=1,
                           **backtest_options):
    """Construit et teste plusieurs recettes composites indépendantes."""
    backtest_options = _ensure_monthly_base_cache(backtest_options)
    if not isinstance(composite_configs, dict) or not composite_configs:
        raise ValueError('Ajoutez au moins une configuration composite nommée.')

    working_screen = screen
    results = {}
    tasks = []
    for index, (composite_name, signal_config) in enumerate(composite_configs.items(), start=1):
        if not isinstance(signal_config, dict) or not signal_config:
            raise ValueError(
                f'La configuration composite « {composite_name} » doit contenir des signaux.'
            )
        score_col = f'{score_prefix}_{index}_{_safe_filename(composite_name)}'
        print(f'Test de score composite : {composite_name}')
        resolved_config = _resolve_signal_config(working_screen, signal_config)
        working_screen = calculate_composite_score(
            working_screen, score_col, resolved_config,
        )
        tasks.append({
            'name': composite_name,
            'metric': score_col,
            'metadata': {
                'test_type': 'composite',
                'test_name': str(composite_name),
                'components': describe_signal_config(
                    resolved_config, role='composite',
                ),
            },
        })

    task_results = _run_signal_tasks(
        working_screen,
        returns,
        tasks,
        list_noire_path,
        backtest_options,
        n_jobs,
    )
    results.update({
        task['name']: result
        for task, result in zip(tasks, task_results)
    })

    return {'screen': working_screen, 'results': results}


def _iter_backtest_results(results, path=()):
    """Parcourt récursivement les différentes structures de résultats du module."""
    if not isinstance(results, dict):
        return
    if 'screen' in results and isinstance(results.get('results'), dict):
        yield from _iter_backtest_results(results['results'], path)
        return
    if 'backtest' in results and isinstance(results['backtest'], dict):
        yield ' / '.join(path), results['backtest']
        return
    if 'figure' in results and ('top_builder' in results or 'metadata' in results):
        yield ' / '.join(path), results
        return
    for name, value in results.items():
        if isinstance(value, dict):
            yield from _iter_backtest_results(value, path + (str(name),))


def _component_recipe(components):
    """Produit une description compacte et lisible de la recette d'un test."""
    recipe = []
    for component in components:
        direction = component.get('higher_is_better')
        direction_label = 'higher' if direction is True else 'lower' if direction is False else 'raw'
        recipe.append(
            f"{component.get('role')}:{component.get('raw_variable')}"
            f"[{component.get('dimension')}]x{component.get('weight')}"
            f"({direction_label})"
        )
    return ' + '.join(recipe)


def _composite_display_name(metadata, raw_variables):
    """Décrit un composite avec ses variables, dimensions et poids."""
    if metadata.get('test_type') != 'composite' or not raw_variables:
        return None
    components = metadata.get('components', [])
    variable_labels = []
    for raw_variable in raw_variables:
        dimensions = [
            f"{component.get('dimension')}×{component.get('weight')}"
            for component in components
            if component.get('raw_variable') == raw_variable
        ]
        variable_label = str(raw_variable)
        if dimensions:
            variable_label += f'[{", ".join(dimensions)}]'
        variable_labels.append(variable_label)
    return f'composite / {" | ".join(variable_labels)}'


def _period_metric_records(period_metrics):
    """Normalise les métriques par période sous forme de liste de dictionnaires."""
    if isinstance(period_metrics, pd.DataFrame):
        return period_metrics.to_dict(orient='records')
    if isinstance(period_metrics, list):
        return period_metrics
    if isinstance(period_metrics, dict):
        return list(period_metrics.values())
    return []


def _components_with_weight_summary(components):
    """Ajoute le résumé de poids à chaque composant sérialisé."""
    weight_summary = summarize_component_weights(components)
    records = []
    for component in components:
        raw_variable = component.get('raw_variable')
        records.append({
            **component,
            'raw_variable_total_weight': weight_summary.get(
                raw_variable, {},
            ).get('total_weight'),
            'raw_variable_absolute_weight': weight_summary.get(
                raw_variable, {},
            ).get('absolute_weight'),
            'raw_variable_absolute_weight_share': weight_summary.get(
                raw_variable, {},
            ).get('absolute_weight_share'),
        })
    return records


def _components_json(components):
    """Sérialise sans perte la composition d'un test dans une cellule CSV."""
    return json.dumps(
        _components_with_weight_summary(components),
        ensure_ascii=False,
        default=str,
    )


def compare_backtest_results(results):
    """Crée une table comparable des scores, pénalités et compositions enregistrées."""
    rows = []
    for test_path, result in _iter_backtest_results(results):
        metadata = result.get('metadata', {})
        components = metadata.get('components', [])
        raw_variables = result.get('raw_variables') or list(dict.fromkeys(
            component.get('raw_variable') for component in components
        ))
        raw_variable_weights = (
            result.get('raw_variable_weights') or summarize_component_weights(components)
        )
        source_test_name = metadata.get('test_name', metadata.get('metric'))
        test_name = (
            _composite_display_name(metadata, raw_variables) or source_test_name
        )
        row = {
            'test_path': test_path,
            'test_group': test_path.rsplit(' / ', 1)[0] if ' / ' in test_path else test_path,
            'test_type': metadata.get('test_type'),
            'test_name': test_name,
            'metric': metadata.get('metric'),
            'benchmark': metadata.get('benchmark'),
            'percentile': metadata.get('percentile'),
            'start_date': metadata.get('start_date'),
            'frequency_rebalancing': metadata.get('frequency_rebalancing'),
            'fill_method': metadata.get('fill_method'),
            'robust_score': result.get('robust_score'),
            'top_bench_ratio': result.get('top_bench_ratio'),
            'top_worst_ratio': result.get('top_worst_ratio'),
            'active_max_drawdown': result.get('active_max_drawdown'),
            'tracking_error_annualized': result.get('tracking_error_annualized'),
            'min_rolling_3y_cagr': result.get('min_rolling_3y_cagr'),
            'observation_count': result.get('observation_count'),
            'component_count': len(components),
            'raw_variable_count': len(raw_variables),
            'raw_variables': ' | '.join(str(variable) for variable in raw_variables),
            'raw_variable_weights': json.dumps(
                raw_variable_weights, ensure_ascii=False, sort_keys=False,
            ),
            'composition_recipe': _component_recipe(components),
            'components_json': _components_json(components),
            'classic_metrics_json': json.dumps(
                result.get('classic_metrics', {}),
                ensure_ascii=False,
                default=str,
            ),
        }
        row.update({column: result.get(column) for column in CLASSIC_METRIC_COLUMNS})
        rows.append(row)
    return _rank_backtest_summary(pd.DataFrame(rows))


def _rank_backtest_summary(summary):
    """Recalcule les classements d'une synthèse éventuellement fusionnée."""
    if summary.empty:
        return summary
    summary = summary.drop(
        columns=[
            column for column in summary.columns
            if column in ('robust_rank_global', 'robust_rank_within_type')
            or column.endswith('_delta_vs_baseline')
        ],
        errors='ignore',
    ).copy()
    summary['robust_rank_global'] = summary['robust_score'].rank(
        ascending=False, method='min', na_option='bottom',
    )
    summary['robust_rank_within_type'] = summary.groupby('test_type')['robust_score'].rank(
        ascending=False, method='min', na_option='bottom',
    )
    for test_group, group_rows in summary.groupby('test_group'):
        baseline_rows = group_rows[group_rows['test_type'] == 'incremental_baseline']
        if baseline_rows.empty:
            continue
        baseline = baseline_rows.iloc[0]
        candidate_mask = (
            (summary['test_group'] == test_group)
            & (summary['test_type'] == 'incremental_candidate')
        )
        for metric in ('robust_score', 'top_bench_ratio', 'top_worst_ratio'):
            summary.loc[candidate_mask, f'{metric}_delta_vs_baseline'] = (
                summary.loc[candidate_mask, metric] - baseline[metric]
            )
    return summary.sort_values(['test_type', 'robust_rank_within_type', 'test_name'])


def _merge_export_table(path, current, replaced_paths):
    """Fusionne une table exportée en remplaçant intégralement les tests relancés."""
    path = Path(path)
    if not path.exists():
        return current.reset_index(drop=True)
    previous = pd.read_csv(path)
    if 'test_path' not in previous.columns:
        return current.reset_index(drop=True)
    previous = previous.loc[
        ~previous['test_path'].astype(str).isin(replaced_paths)
    ]
    return pd.concat([previous, current], ignore_index=True, sort=False)


def _safe_filename(value):
    """Transforme un nom de test en nom de fichier portable."""
    cleaned = re.sub(r'[^A-Za-z0-9._-]+', '_', str(value)).strip('._')
    return cleaned or 'backtest'


def _performance_file_stem(test_path):
    """Construit un nom de fichier court à partir de l'identifiant unique du test."""
    # L'identifiant du chemin suffit à distinguer chaque résultat et évite de répéter
    # le nom du test dans les exports incrémentaux très longs.
    return _safe_filename(test_path)


def _write_tabular(value, path):
    """Écrit une série ou une table sans imposer de format aux autres objets."""
    if isinstance(value, pd.Series):
        value.to_frame().to_csv(path, index=True)
    elif isinstance(value, pd.DataFrame):
        value.to_csv(path, index=True)


def _read_performance_csv(path):
    """Recharge une courbe de performance exportée et restaure son index temporel."""
    performance = pd.read_csv(path)
    if performance.empty:
        raise ValueError(f'Fichier de performance vide : {path}')
    date_column = 'Date' if 'Date' in performance.columns else performance.columns[0]
    performance[date_column] = pd.to_datetime(performance[date_column], errors='coerce')
    performance = performance.loc[performance[date_column].notna()].set_index(date_column)
    performance.index.name = 'Date'
    for column in performance.columns:
        performance[column] = pd.to_numeric(performance[column], errors='coerce')
    return performance.sort_index()


def _load_saved_performances(export_dir):
    """Recharge les performances locales en utilisant le registre quand il existe."""
    export_dir = Path(export_dir)
    data_dir = export_dir / 'data'
    registry_path = export_dir / 'backtest_registry.json'
    sources = {}
    registered_files = set()

    if registry_path.exists():
        with registry_path.open('r', encoding='utf-8') as registry_file:
            registry = json.load(registry_file)
        for entry in registry:
            test_path = entry.get('test_path')
            metadata = entry.get('metadata', {})
            components = metadata.get('components', [])
            test_name = metadata.get('test_name') or metadata.get('metric') or test_path
            relative_path = entry.get('files', {}).get('performance')
            if relative_path:
                performance_path = export_dir / relative_path
            else:
                file_stem = _performance_file_stem(test_path)
                performance_path = data_dir / f'{file_stem}_performance.csv'
            if not test_path or not performance_path.exists():
                continue
            resolved_path = performance_path.resolve()
            registered_files.add(resolved_path)
            sources[test_path] = {
                'test_name': test_name,
                'metadata': metadata,
                'raw_variables': entry.get('raw_variables') or list(dict.fromkeys(
                    component.get('raw_variable') for component in components
                )),
                'raw_variable_weights': (
                    entry.get('raw_variable_weights')
                    or summarize_component_weights(components)
                ),
                'robust_score': entry.get('metrics', {}).get('robust_score'),
                'performance': _read_performance_csv(performance_path),
                'origin': str(performance_path),
            }

    if data_dir.exists() and not registry_path.exists():
        for performance_path in sorted(data_dir.glob('*_performance.csv')):
            if performance_path.resolve() in registered_files:
                continue
            test_path = performance_path.stem.removesuffix('_performance')
            sources[test_path] = {
                'test_name': test_path,
                'metadata': {},
                'raw_variables': [],
                'raw_variable_weights': {},
                'performance': _read_performance_csv(performance_path),
                'origin': str(performance_path),
            }
    return sources


def _collect_performance_sources(results=None, export_dir=None):
    """Réunit les performances du disque et de la mémoire dans un registre unique."""
    sources = _load_saved_performances(export_dir) if export_dir is not None else {}
    if results is not None:
        for test_path, result in _iter_backtest_results(results):
            performance = result.get('performance')
            if not isinstance(performance, pd.DataFrame) or performance.empty:
                continue
            metadata = result.get('metadata', {})
            components = metadata.get('components', [])
            sources[test_path] = {
                'test_name': metadata.get('test_name') or metadata.get('metric') or test_path,
                'metadata': metadata,
                'raw_variables': result.get('raw_variables') or list(dict.fromkeys(
                    component.get('raw_variable') for component in components
                )),
                'raw_variable_weights': (
                    result.get('raw_variable_weights')
                    or summarize_component_weights(components)
                ),
                'robust_score': result.get('robust_score'),
                'performance': performance.copy(),
                'origin': 'mémoire',
            }
    if not sources:
        raise ValueError('Aucune performance disponible en mémoire ou dans le dossier exporté.')
    return sources


def _resolve_performance_source(identifier, sources):
    """Résout un chemin de test exact ou un nom de test non ambigu."""
    if identifier in sources:
        return identifier, sources[identifier]
    matches = [
        (test_path, source)
        for test_path, source in sources.items()
        if source.get('test_name') == identifier
    ]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        paths = ', '.join(test_path for test_path, _ in matches)
        raise ValueError(
            f'Nom de test ambigu « {identifier} ». Utilisez un chemin parmi : {paths}'
        )
    available = ', '.join(sorted(sources))
    raise KeyError(f'Test introuvable « {identifier} ». Tests disponibles : {available}')


def _performance_display_path(test_path, source):
    """Décrit un composite avec ses variables, dimensions et poids."""
    metadata = source.get('metadata', {})
    return (
        _composite_display_name(metadata, source.get('raw_variables', []))
        or test_path
    )


def _performance_composition_table(selected_sources):
    """Construit la composition détaillée des tests présents dans la comparaison."""
    rows = []
    seen_paths = set()
    for test_path, source in selected_sources:
        if test_path in seen_paths:
            continue
        seen_paths.add(test_path)
        metadata = source.get('metadata', {})
        components = metadata.get('components', [])
        weight_summary = (
            source.get('raw_variable_weights')
            or summarize_component_weights(components)
        )
        for component in components:
            raw_variable = component.get('raw_variable')
            variable_summary = weight_summary.get(raw_variable, {})
            rows.append({
                'display_path': _performance_display_path(test_path, source),
                'test_path': test_path,
                'test_type': metadata.get('test_type'),
                'test_name': source.get('test_name'),
                **component,
                'raw_variable_total_weight': variable_summary.get('total_weight'),
                'raw_variable_absolute_weight': variable_summary.get('absolute_weight'),
                'raw_variable_absolute_weight_share': variable_summary.get(
                    'absolute_weight_share'
                ),
            })
    return pd.DataFrame(rows)


def combine_backtest_performances(results=None, export_dir=None, selections=None,
                                  portfolios=('Top',), save_path=None,
                                  return_composition=False):
    """Combine des performances en mémoire et complète les absences depuis le disque.

    ``selections`` associe le nom final d'une colonne à un couple
    ``(chemin_ou_nom_du_test, portefeuille)``. Sans sélection, tous les tests
    disponibles sont combinés pour les portefeuilles demandés. Pour un composite,
    le libellé automatique contient directement toutes ses variables brutes.
    Avec ``return_composition=True``, la fonction renvoie aussi la table détaillée
    des dimensions, directions, dénominateurs et poids.
    """
    sources = _collect_performance_sources(results=results, export_dir=export_dir)

    series = {}
    selected_sources = []
    if selections is None:
        for test_path, source in sources.items():
            display_path = _performance_display_path(test_path, source)
            for portfolio in portfolios:
                if portfolio in source['performance'].columns:
                    label = f'{display_path} | {portfolio}'
                    if label in series:
                        label = f'{label} [{source.get("test_name")}]'
                    series[label] = source['performance'][portfolio]
                    selected_sources.append((test_path, source))
    else:
        for label, selection in selections.items():
            if isinstance(selection, str):
                identifier, portfolio = selection, 'Top'
            else:
                try:
                    identifier, portfolio = selection
                except (TypeError, ValueError) as error:
                    raise ValueError(
                        f'Sélection invalide pour « {label} » : utilisez (test, portefeuille).'
                    ) from error
            test_path, source = _resolve_performance_source(identifier, sources)
            if portfolio not in source['performance'].columns:
                raise KeyError(
                    f'Portefeuille « {portfolio} » absent pour le test « {identifier} ». '
                    f'Colonnes disponibles : {", ".join(source["performance"].columns)}'
                )
            series[label] = source['performance'][portfolio]
            selected_sources.append((test_path, source))

    if not series:
        raise ValueError('Aucune série ne correspond aux portefeuilles demandés.')
    combined = pd.concat(series, axis=1).sort_index()
    combined.index.name = 'Date'
    composition = _performance_composition_table(selected_sources)
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(save_path, index=True)
        print(f'Comparaison des performances exportée : {save_path}')
    if return_composition:
        return combined, composition
    return combined


def _comparison_test_paths(sources, max_tests, scores_by_path=None):
    """Retient les meilleurs tests pour garder la comparaison graphique lisible."""
    if max_tests is None:
        return list(sources)
    if not isinstance(max_tests, int) or max_tests < 1:
        raise ValueError('max_tests doit être un entier positif ou None.')
    scores_by_path = scores_by_path or {}

    def sort_key(item):
        test_path, source = item
        robust_score = pd.to_numeric(
            scores_by_path.get(test_path, source.get('robust_score')),
            errors='coerce',
        )
        return (
            pd.isna(robust_score),
            -float(robust_score) if pd.notna(robust_score) else 0.0,
            test_path,
        )

    return [
        test_path for test_path, _ in sorted(sources.items(), key=sort_key)[:max_tests]
    ]


def _comparison_scores_by_period(metrics, period_id):
    """Associe chaque test à son Robust Score pour une période donnée."""
    if not isinstance(metrics, pd.DataFrame) or metrics.empty:
        return {}
    required_columns = {'test_path', 'period_id', 'robust_score'}
    if not required_columns.issubset(metrics.columns):
        return {}
    period_rows = metrics.loc[
        metrics['period_id'].astype(str).eq(str(period_id)),
        ['test_path', 'robust_score'],
    ].drop_duplicates('test_path', keep='last')
    return period_rows.set_index('test_path')['robust_score'].to_dict()


def _comparison_metrics(results=None, export_dir=None):
    """Recharge ou reconstruit la table unique des métriques de comparaison."""
    if export_dir is not None:
        metrics_path = Path(export_dir) / 'backtest_metrics.csv'
        if metrics_path.exists():
            return pd.read_csv(metrics_path)
    if results is not None:
        return _build_analysis_tables_from_results(results)['metrics_by_period']
    return pd.DataFrame()


def _comparison_period_definitions(metrics, period_breakpoints=None):
    """Décrit la période totale et les sous-périodes disponibles dans les métriques."""
    total_period = {
        'id': 'total',
        'label': 'Période totale',
        'start': None,
        'end': None,
    }
    required_columns = {
        'period_id', 'period_label', 'actual_start_date', 'actual_end_date',
    }
    if isinstance(metrics, pd.DataFrame) and required_columns.issubset(metrics.columns):
        period_rows = metrics.loc[
            metrics['period_id'].notna(),
            ['period_id', 'period_label', 'actual_start_date', 'actual_end_date'],
        ].drop_duplicates('period_id', keep='last').copy()
        if not period_rows.empty:
            total_rows = period_rows.loc[
                period_rows['period_id'].astype(str).eq('total')
            ]
            if not total_rows.empty:
                total_label = total_rows.iloc[0]['period_label']
                if pd.notna(total_label):
                    total_period['label'] = str(total_label)
            subperiod_rows = period_rows.loc[
                ~period_rows['period_id'].astype(str).eq('total')
            ].copy()
            if not subperiod_rows.empty:
                subperiod_rows['_start_sort'] = pd.to_datetime(
                    subperiod_rows['actual_start_date'], errors='coerce',
                )
                subperiod_rows = subperiod_rows.sort_values(
                    ['_start_sort', 'period_id'], na_position='last',
                )
                subperiods = []
                for _, row in subperiod_rows.iterrows():
                    subperiods.append({
                        'id': str(row['period_id']),
                        'label': str(row['period_label']),
                        'start': row['actual_start_date'],
                        'end': row['actual_end_date'],
                    })
                return [total_period, *subperiods]
            return [total_period]
    if period_breakpoints:
        return [total_period, *build_periods_from_breakpoints(period_breakpoints)]
    return [total_period]


def build_performance_comparison_definitions(results=None, export_dir=None,
                                             max_tests=8,
                                             include_worst_benchmark_ratio=False,
                                             period_id='total', metrics=None):
    """Crée une comparaison lisible des meilleurs tests disponibles.

    Le score de robustesse de ``period_id`` sélectionne au plus ``max_tests``
    tests. Passez ``None`` pour reproduire une comparaison exhaustive.
    """
    sources = _collect_performance_sources(results=results, export_dir=export_dir)
    if metrics is None:
        metrics = _comparison_metrics(results=results, export_dir=export_dir)
    scores_by_path = _comparison_scores_by_period(metrics, period_id)
    selections = {}
    labels_by_test = {}
    benchmark_selection = None

    for test_path in _comparison_test_paths(sources, max_tests, scores_by_path):
        source = sources[test_path]
        performance = source['performance']
        display_path = _performance_display_path(test_path, source)
        test_labels = {}
        for portfolio in ('Top', 'Worst'):
            if portfolio not in performance.columns:
                continue
            label = f'{display_path} | {portfolio}'
            if label in selections:
                label = f'{label} [{source.get("test_name")}]'
            selections[label] = (test_path, portfolio)
            test_labels[portfolio] = label
        labels_by_test[test_path] = test_labels
        if benchmark_selection is None and 'Bench' in performance.columns:
            benchmark_selection = (test_path, 'Bench')

    if benchmark_selection is None:
        raise KeyError('Aucune performance Benchmark n’est disponible dans les tests.')
    selections['Benchmark'] = benchmark_selection

    ratio_definitions = {}
    for test_path, labels in labels_by_test.items():
        top_label = labels.get('Top')
        worst_label = labels.get('Worst')
        if top_label:
            ratio_definitions[f'{top_label} / Benchmark'] = (top_label, 'Benchmark')
        if include_worst_benchmark_ratio and worst_label:
            ratio_definitions[f'{worst_label} / Benchmark'] = (
                worst_label, 'Benchmark',
            )
        if {'Top', 'Worst'}.issubset(labels):
            ratio_definitions[f'{top_label} / Worst'] = (top_label, worst_label)
    return selections, ratio_definitions


def prepare_performance_comparison(results=None, export_dir=None, save_path=None,
                                   max_tests=8,
                                   include_worst_benchmark_ratio=False,
                                   period_id='total', metrics=None):
    """Prépare une comparaison graphique limitée aux tests les plus pertinents."""
    selections, ratio_definitions = build_performance_comparison_definitions(
        results=results,
        export_dir=export_dir,
        max_tests=max_tests,
        include_worst_benchmark_ratio=include_worst_benchmark_ratio,
        period_id=period_id,
        metrics=metrics,
    )
    performance, composition = combine_backtest_performances(
        results=results,
        export_dir=export_dir,
        selections=selections,
        save_path=save_path,
        return_composition=True,
    )
    ratios = calculate_performance_ratios(
        performance,
        benchmark_column='Benchmark',
        ratio_definitions=ratio_definitions,
    )
    return {
        'performance': performance,
        'ratios': ratios,
        'composition': composition,
        'performance_selection': selections,
        'ratio_definitions': ratio_definitions,
    }


def prepare_performance_comparisons_by_period(results=None, export_dir=None,
                                               max_tests=8,
                                               include_worst_benchmark_ratio=False,
                                               period_breakpoints=None):
    """Prépare une comparaison indépendante pour la période totale et chaque segment."""
    metrics = _comparison_metrics(results=results, export_dir=export_dir)
    periods = _comparison_period_definitions(metrics, period_breakpoints)
    comparisons = {}
    for period in periods:
        comparison = prepare_performance_comparison(
            results=results,
            export_dir=export_dir,
            max_tests=max_tests,
            include_worst_benchmark_ratio=include_worst_benchmark_ratio,
            period_id=period['id'],
            metrics=metrics,
        )
        comparison['period'] = period
        comparison['period_definitions'] = periods
        comparisons[period['id']] = comparison
    return comparisons


def calculate_performance_ratios(performance, benchmark_column='Benchmark',
                                 ratio_definitions=None):
    """Calcule les ratios demandés sans construire de figure."""
    if not isinstance(performance, pd.DataFrame) or performance.empty:
        raise ValueError('La table de performances doit être un DataFrame non vide.')
    if ratio_definitions is None:
        if benchmark_column not in performance.columns:
            raise KeyError(
                f'Benchmark « {benchmark_column} » absent. '
                f'Colonnes disponibles : {", ".join(performance.columns)}'
            )
        comparison_columns = [
            column for column in performance.columns if column != benchmark_column
        ]
        if not comparison_columns:
            raise ValueError('Ajoutez au moins une performance à comparer au benchmark.')
        benchmark = performance[benchmark_column].replace(0, np.nan)
        ratios = performance[comparison_columns].div(benchmark, axis=0)
    else:
        if not ratio_definitions:
            raise ValueError('Ajoutez au moins une définition de ratio.')
        ratio_series = {}
        for label, definition in ratio_definitions.items():
            try:
                numerator, denominator = definition
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f'Définition invalide pour « {label} » : utilisez '
                    '(numérateur, dénominateur).'
                ) from error
            missing_columns = [
                column for column in (numerator, denominator)
                if column not in performance.columns
            ]
            if missing_columns:
                raise KeyError(
                    f'Colonnes absentes pour le ratio « {label} » : '
                    f'{missing_columns}'
                )
            ratio_series[label] = (
                performance[numerator]
                / performance[denominator].replace(0, np.nan)
            )
        ratios = pd.DataFrame(ratio_series, index=performance.index)
    ratios.index.name = performance.index.name
    return ratios


def _build_analysis_tables_from_results(results):
    """Construit en mémoire les vues dérivées de la table unique de métriques."""
    summary = compare_backtest_results(results)
    period_metrics = _period_metrics_from_results(results)
    metrics = _finalize_backtest_metrics(
        _combine_total_and_period_metrics(summary, period_metrics),
    )
    return _analysis_views_from_metrics(metrics)


def _period_metrics_from_results(results):
    """Extrait les lignes de sous-périodes des résultats mémoire."""
    rows = []
    for test_path, result in _iter_backtest_results(results):
        metadata = result.get('metadata', {})
        raw_variables = result.get('raw_variables') or list(dict.fromkeys(
            component.get('raw_variable') for component in metadata.get('components', [])
        ))
        source_test_name = metadata.get('test_name') or metadata.get('metric') or test_path
        test_name = _composite_display_name(metadata, raw_variables) or source_test_name
        for period_row in _period_metric_records(result.get('period_metrics')):
            rows.append({
                'test_path': test_path,
                'test_name': test_name,
                'test_type': metadata.get('test_type'),
                'metric': metadata.get('metric'),
                **period_row,
            })
    return pd.DataFrame(rows)


def _summary_context_columns(summary):
    """Liste les colonnes de configuration répétées pour chaque sous-période."""
    candidates = (
        'test_path', 'test_group', 'test_type', 'test_name', 'metric',
        'benchmark', 'percentile', 'start_date', 'frequency_rebalancing',
        'fill_method', 'component_count', 'raw_variable_count',
        'raw_variables', 'raw_variable_weights', 'composition_recipe',
        'components_json',
    )
    return [column for column in candidates if column in summary.columns]


def _add_summary_context_to_periods(summary, period_metrics):
    """Ajoute la configuration du test à ses lignes de sous-périodes."""
    if period_metrics.empty or summary.empty or 'test_path' not in period_metrics:
        return period_metrics.copy()
    context_columns = _summary_context_columns(summary)
    if 'test_path' not in context_columns:
        return period_metrics.copy()
    context = summary.loc[:, context_columns].drop_duplicates('test_path').set_index(
        'test_path', drop=False,
    )
    enriched = period_metrics.copy()
    for column in context_columns:
        if column == 'test_path':
            continue
        values = enriched['test_path'].map(context[column])
        if column in enriched.columns:
            enriched[column] = enriched[column].where(enriched[column].notna(), values)
        else:
            enriched[column] = values
    return enriched


def _combine_total_and_period_metrics(summary, period_metrics):
    """Réunit les métriques, la configuration et la composition dans une table unique."""
    total_rows = []

    def relative_cagr(portfolio_return, reference_return):
        if pd.isna(portfolio_return) or pd.isna(reference_return):
            return float('nan')
        if 1 + reference_return <= 0:
            return float('nan')
        return (1 + portfolio_return) / (1 + reference_return) - 1

    for _, summary_row in summary.iterrows():
        observation_count = summary_row.get('observation_count')
        years = (
            max((observation_count - 1) / 252, 0)
            if pd.notna(observation_count) else float('nan')
        )
        total_row = {
            **summary_row.to_dict(),
            'scope': 'total',
            'period_id': 'total',
            'period_label': 'Période totale',
            'requested_start_date': summary_row.get('start_date'),
            'requested_end_date': None,
            'actual_start_date': None,
            'actual_end_date': None,
            'observation_count': observation_count,
            'years': years,
            'top_cagr': summary_row.get('top_annualized_return'),
            'worst_cagr': summary_row.get('worst_annualized_return'),
            'bench_cagr': summary_row.get('bench_annualized_return'),
            'active_cagr': relative_cagr(
                summary_row.get('top_annualized_return'),
                summary_row.get('bench_annualized_return'),
            ),
            'top_worst_cagr': relative_cagr(
                summary_row.get('top_annualized_return'),
                summary_row.get('worst_annualized_return'),
            ),
            'robust_score': summary_row.get('robust_score'),
            'top_bench_ratio': summary_row.get('top_bench_ratio'),
            'top_worst_ratio': summary_row.get('top_worst_ratio'),
            'active_max_drawdown': summary_row.get('active_max_drawdown'),
            'tracking_error_annualized': summary_row.get('tracking_error_annualized'),
            'min_rolling_3y_cagr': summary_row.get('min_rolling_3y_cagr'),
        }
        total_rows.append(total_row)

    total_metrics = pd.DataFrame(total_rows)
    subperiod_metrics = _add_summary_context_to_periods(summary, period_metrics)
    if not subperiod_metrics.empty:
        subperiod_metrics.insert(0, 'scope', 'subperiod')
    column_order = list(dict.fromkeys([
        *total_metrics.columns, *subperiod_metrics.columns,
    ]))
    available_metrics = [
        frame.dropna(axis=1, how='all')
        for frame in (total_metrics, subperiod_metrics)
        if not frame.empty
    ]
    if not available_metrics:
        return pd.DataFrame(columns=column_order)
    combined = pd.concat(
        available_metrics,
        ignore_index=True,
        sort=False,
    ).reindex(columns=column_order)
    return combined


def _finalize_backtest_metrics(metrics):
    """Recalcule les classements après la fusion de plusieurs exports."""
    if metrics.empty or 'scope' not in metrics.columns:
        return metrics.reset_index(drop=True)
    metrics = metrics.copy()
    total_mask = metrics['scope'].eq('total')
    if total_mask.any():
        ranked_summary = _rank_backtest_summary(metrics.loc[total_mask].copy())
        ranking_columns = [
            column for column in ranked_summary.columns
            if column.endswith('_delta_vs_baseline')
        ]
        for column in ranking_columns:
            ranks = ranked_summary.set_index('test_path')[column]
            metrics.loc[total_mask, column] = metrics.loc[
                total_mask, 'test_path'
            ].map(ranks)

    ranking_metrics = (
        'robust_score', 'active_cagr', 'top_worst_cagr', 'top_sharpe_ratio',
    )
    period_ranking_columns = [
        f'{metric}_{suffix}'
        for metric in ranking_metrics
        for suffix in ('rank_global', 'rank_within_type')
        if f'{metric}_{suffix}' in metrics.columns
    ]
    metrics = metrics.drop(columns=period_ranking_columns, errors='ignore')
    for metric in ranking_metrics:
        if metric not in metrics.columns:
            continue
        metrics[f'{metric}_rank_global'] = metrics.groupby(
            'period_id', dropna=False,
        )[metric].rank(ascending=False, method='min', na_option='bottom')
        metrics[f'{metric}_rank_within_type'] = metrics.groupby(
            ['period_id', 'test_type'], dropna=False,
        )[metric].rank(ascending=False, method='min', na_option='bottom')
    scope_order = metrics['scope'].map({'total': 0, 'subperiod': 1}).fillna(2)
    return metrics.assign(_scope_order=scope_order).sort_values(
        ['_scope_order', 'period_id', 'test_type', 'test_name'],
    ).drop(columns='_scope_order').reset_index(drop=True)


def _components_from_metrics(metrics):
    """Reconstruit une table de composition à partir de la colonne JSON."""
    if metrics.empty or 'components_json' not in metrics.columns:
        return pd.DataFrame()
    total_metrics = metrics.loc[metrics['scope'].eq('total')]
    rows = []
    for _, metric_row in total_metrics.drop_duplicates('test_path').iterrows():
        serialized_components = metric_row.get('components_json')
        if not isinstance(serialized_components, str):
            continue
        try:
            components = json.loads(serialized_components)
        except json.JSONDecodeError:
            continue
        if not isinstance(components, list):
            continue
        for component in components:
            if isinstance(component, dict):
                rows.append({
                    'test_path': metric_row.get('test_path'),
                    'test_name': metric_row.get('test_name'),
                    **component,
                })
    return pd.DataFrame(rows)


def _classic_metrics_from_summary(summary):
    """Reconstruit le format long des métriques classiques depuis la période totale."""
    rows = []
    for _, summary_row in summary.iterrows():
        serialized_metrics = summary_row.get('classic_metrics_json')
        if isinstance(serialized_metrics, str):
            try:
                classic_metrics = json.loads(serialized_metrics)
            except json.JSONDecodeError:
                classic_metrics = {}
            if isinstance(classic_metrics, dict):
                for portfolio, metrics in classic_metrics.items():
                    if not isinstance(metrics, dict):
                        continue
                    for metric, value in metrics.items():
                        rows.append({
                            'test_path': summary_row.get('test_path'),
                            'test_name': summary_row.get('test_name'),
                            'portfolio': portfolio,
                            'metric': metric,
                            'value': value,
                        })
                continue
        for portfolio in ('top', 'worst', 'bench'):
            for metric in CLASSIC_METRIC_NAMES:
                column = f'{portfolio}_{metric}'
                value = summary_row.get(column)
                if pd.notna(value):
                    rows.append({
                        'test_path': summary_row.get('test_path'),
                        'test_name': summary_row.get('test_name'),
                        'portfolio': portfolio.title(),
                        'metric': metric,
                        'value': value,
                    })
    return pd.DataFrame(rows)


def _analysis_views_from_metrics(metrics):
    """Expose les vues historiques sans créer de fichiers CSV supplémentaires."""
    metrics = metrics.copy()
    total_metrics = metrics.loc[metrics['scope'].eq('total')].copy()
    period_metrics = metrics.loc[metrics['scope'].eq('subperiod')].copy()
    return {
        'summary': total_metrics,
        'total_metrics': total_metrics,
        'classic_metrics': _classic_metrics_from_summary(total_metrics),
        'period_metrics': period_metrics,
        'metrics_by_period': metrics,
        'signal_composition': _components_from_metrics(metrics),
    }


def reconstruct_backtest_analysis(results=None, export_dir=None, selections=None,
                                  portfolios=('Top',), performance_save_path=None):
    """Restaure les performances et les vues dérivées de la table unique."""
    performance, performance_composition = combine_backtest_performances(
        results=results,
        export_dir=export_dir,
        selections=selections,
        portfolios=portfolios,
        save_path=performance_save_path,
        return_composition=True,
    )

    if results is not None:
        views = _build_analysis_tables_from_results(results)
        metrics = views['metrics_by_period']
        source = 'mémoire'
    else:
        if export_dir is None:
            raise ValueError('Indiquez export_dir lorsque les résultats mémoire sont absents.')
        export_dir = Path(export_dir)
        metrics_path = export_dir / 'backtest_metrics.csv'
        if metrics_path.exists():
            metrics = pd.read_csv(metrics_path)
        else:
            legacy_summary_path = export_dir / 'backtest_summary.csv'
            legacy_period_path = export_dir / 'period_metrics.csv'
            if not legacy_summary_path.exists() or not legacy_period_path.exists():
                raise FileNotFoundError(
                    f'Table unique absente dans {export_dir} : {metrics_path.name}'
                )
            legacy_summary = pd.read_csv(legacy_summary_path).rename(
                columns={'recipe': 'composition_recipe'},
            )
            legacy_periods = pd.read_csv(legacy_period_path)
            metrics = _finalize_backtest_metrics(
                _combine_total_and_period_metrics(legacy_summary, legacy_periods),
            )
        views = _analysis_views_from_metrics(metrics)
        source = 'disque'

    composite_names = (
        performance_composition.loc[
            performance_composition['test_type'].eq('composite'),
            ['test_path', 'display_path'],
        ]
        .drop_duplicates('test_path')
        .set_index('test_path')['display_path']
    )
    for view_name, table in views.items():
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        if not {'test_path', 'test_name'}.issubset(table.columns):
            continue
        display_names = table['test_path'].map(composite_names)
        if display_names.notna().any():
            table = table.copy()
            table.loc[display_names.notna(), 'test_name'] = display_names.dropna()
            views[view_name] = table
    views['metrics_by_period'] = views['metrics_by_period'].copy()
    display_names = views['metrics_by_period']['test_path'].map(composite_names)
    if display_names.notna().any():
        views['metrics_by_period'].loc[display_names.notna(), 'test_name'] = (
            display_names.dropna()
        )
    return {
        'source': source,
        'performance': performance,
        'performance_composition': performance_composition,
        **views,
    }


def _rebase_frame(frame, base_value):
    """Rebase chaque série sur sa première observation valide et non nulle."""
    rebased = frame.copy()
    for column in rebased.columns:
        valid = rebased[column].dropna()
        valid = valid[valid.ne(0)]
        if not valid.empty:
            rebased[column] = rebased[column] / valid.iloc[0] * base_value
    return rebased


def plot_performance_comparison(performance, ratios, benchmark_column='Benchmark',
                                title='Comparaison des performances',
                                save_path=None, show_plot=True, rebase=True,
                                period_breakpoints=None,
                                show_worst_performance=False,
                                period_definitions=None,
                                default_period_id='total'):
    """Trace une comparaison compacte avec un sélecteur de périodes."""
    if benchmark_column not in performance.columns:
        raise KeyError(f'Benchmark « {benchmark_column} » absent des performances.')
    performance = performance.sort_index()
    ratios = ratios.sort_index()
    displayed_performance_columns = [
        column for column in performance.columns
        if (
            column == benchmark_column
            or show_worst_performance
            or not (
                str(column) == 'Worst' or str(column).endswith(' | Worst')
            )
        )
    ]

    def displayed_frames(start=None, end=None):
        """Prépare une fenêtre en ramenant chaque série à sa base locale."""
        window_performance = performance.loc[start:end]
        window_ratios = ratios.loc[start:end]
        if rebase:
            window_performance = _rebase_frame(window_performance, base_value=100.0)
            window_ratios = _rebase_frame(window_ratios, base_value=1.0)
        return window_performance.loc[:, displayed_performance_columns], window_ratios

    if period_definitions is None:
        periods = _comparison_period_definitions(
            pd.DataFrame(), period_breakpoints,
        )
    else:
        periods = []
        known_period_ids = set()
        for period in period_definitions:
            period_id = str(period.get('id', 'total'))
            if period_id in known_period_ids:
                continue
            known_period_ids.add(period_id)
            start = period.get('start')
            end = period.get('end')
            periods.append({
                'id': period_id,
                'label': str(period.get('label', period_id)),
                'start': None if pd.isna(start) else start,
                'end': None if pd.isna(end) else end,
            })
        if 'total' not in known_period_ids:
            periods.insert(0, {
                'id': 'total',
                'label': 'Période totale',
                'start': None,
                'end': None,
            })

    periods_by_id = {period['id']: period for period in periods}
    if default_period_id not in periods_by_id:
        raise KeyError(
            f'Période par défaut inconnue : {default_period_id}. '
            f"Périodes disponibles : {', '.join(periods_by_id)}"
        )
    default_period = periods_by_id[default_period_id]
    displayed_performance, displayed_ratios = displayed_frames(
        default_period['start'], default_period['end'],
    )

    def period_title(period):
        """Construit le titre cohérent avec la période effectivement affichée."""
        if period['id'] == 'total':
            return title
        return f"{title} | {period['label']}"

    fig = make_subplots(
        rows=2,
        cols=1,
        shared_xaxes=True,
        vertical_spacing=0.1,
        subplot_titles=(
            'Performance cumulée',
            'Ratios relatifs',
        ),
    )
    non_benchmark_columns = [
        column for column in performance.columns if column != benchmark_column
    ]
    color_map = {
        column: qualitative.Plotly[index % len(qualitative.Plotly)]
        for index, column in enumerate(non_benchmark_columns)
    }
    color_map[benchmark_column] = 'black'

    for column in displayed_performance.columns:
        fig.add_trace(
            go.Scatter(
                x=displayed_performance.index,
                y=displayed_performance[column],
                mode='lines',
                name=column,
                legendgroup=column,
                legend='legend',
                line=dict(
                    color=color_map[column],
                    width=3 if column == benchmark_column else 2,
                ),
            ),
            row=1,
            col=1,
        )

    for index, column in enumerate(ratios.columns):
        is_legacy_benchmark_ratio = column in performance.columns
        ratio_name = (
            f'{column} / {benchmark_column}'
            if is_legacy_benchmark_ratio else column
        )
        numerator = ratio_name.split(' / ', 1)[0]
        fig.add_trace(
            go.Scatter(
                x=displayed_ratios.index,
                y=displayed_ratios[column],
                mode='lines',
                name=ratio_name,
                legendgroup=ratio_name,
                legend='legend2',
                line=dict(
                    color=color_map.get(
                        numerator,
                        qualitative.Plotly[index % len(qualitative.Plotly)],
                    ),
                    width=2,
                    dash=(
                        'solid'
                        if ratio_name.endswith(f' / {benchmark_column}')
                        else 'dash'
                    ),
                ),
            ),
            row=2,
            col=1,
        )
    fig.add_hline(
        y=1.0,
        line_dash='dash',
        line_color='grey',
        row=2,
        col=1,
    )
    fig.update_layout(
        title=period_title(default_period),
        width=1400,
        height=1050,
        template='plotly_white',
        hovermode='x unified',
        legend=dict(
            orientation='v',
            x=1.01,
            xanchor='left',
            y=1,
            yanchor='top',
            title=dict(text='Performances', font=dict(size=10)),
            font=dict(size=9),
        ),
        legend2=dict(
            orientation='v',
            x=1.01,
            xanchor='left',
            y=0.34,
            yanchor='top',
            title=dict(text='Ratios', font=dict(size=10)),
            font=dict(size=9),
        ),
        margin=dict(r=460, t=100),
    )
    if len(periods) > 1:
        period_buttons = []
        active_period_index = 0
        for period in periods:
            period_performance, period_ratios = displayed_frames(
                period.get('start'), period.get('end'),
            )
            if period_performance.dropna(how='all').empty:
                continue
            period_x = [
                period_performance.index.tolist()
                for _ in period_performance.columns
            ] + [
                period_ratios.index.tolist()
                for _ in period_ratios.columns
            ]
            period_y = [
                period_performance[column].tolist()
                for column in period_performance.columns
            ] + [
                period_ratios[column].tolist()
                for column in period_ratios.columns
            ]
            period_buttons.append({
                'label': period['label'],
                'method': 'update',
                'args': [
                    {'x': period_x, 'y': period_y},
                    {
                        'xaxis.autorange': True,
                        'xaxis2.autorange': True,
                        'title.text': period_title(period),
                    },
                ],
            })
            if period['id'] == default_period_id:
                active_period_index = len(period_buttons) - 1
        if len(period_buttons) > 1:
            fig.update_layout(
                updatemenus=[{
                    'buttons': period_buttons,
                    'active': active_period_index,
                    'direction': 'down',
                    'showactive': True,
                    'x': 0,
                    'xanchor': 'left',
                    'y': 1.16,
                    'yanchor': 'top',
                }],
                annotations=[
                    *list(fig.layout.annotations),
                    {
                        'text': 'Période :',
                        'showarrow': False,
                        'x': 0,
                        'xref': 'paper',
                        'y': 1.18,
                        'yref': 'paper',
                        'xanchor': 'left',
                    },
                ],
                margin=dict(r=460, t=145),
            )
    if save_path is not None:
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.write_html(save_path)
    if show_plot:
        fig.show()
    return fig


def export_backtest_results(results, output_dir, export_name=None, export_png=True,
                            export_html=True, export_holdings=False):
    """Exporte une table unique de métriques et les performances nécessaires."""
    export_name = export_name or datetime.now().strftime('backtest_export_%Y%m%d_%H%M%S')
    export_dir = Path(output_dir) / _safe_filename(export_name)
    figures_dir = export_dir / 'figures'
    data_dir = export_dir / 'data'
    holdings_dir = export_dir / 'holdings'
    directories = [data_dir]
    if export_html or export_png:
        directories.append(figures_dir)
    if export_holdings:
        directories.append(holdings_dir)
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)

    flattened = list(_iter_backtest_results(results))
    summary = compare_backtest_results(results)
    replaced_paths = {test_path for test_path, _ in flattened}
    period_metrics = _period_metrics_from_results(results)
    registry = []
    figure_jobs = []
    png_enabled = export_png

    for test_path, result in flattened:
        metadata = copy.deepcopy(result.get('metadata', {}))
        file_stem = _performance_file_stem(test_path)
        performance_path = data_dir / f'{file_stem}_performance.csv'
        top_holdings_path = holdings_dir / f'{file_stem}_top.csv'
        worst_holdings_path = holdings_dir / f'{file_stem}_worst.csv'
        html_path = figures_dir / f'{file_stem}.html'
        png_path = figures_dir / f'{file_stem}.png'

        _write_tabular(result.get('performance'), performance_path)
        if export_holdings:
            _write_tabular(result.get('top_holdings'), top_holdings_path)
            _write_tabular(result.get('worst_holdings'), worst_holdings_path)

        figure = result.get('figure')
        if figure is not None:
            figure_jobs.append((figure, html_path, png_path))

        registry.append({
            'test_path': test_path,
            'metadata': metadata,
            'metrics': {
                key: result.get(key) for key in (
                    'robust_score', 'top_bench_ratio', 'top_worst_ratio',
                    'active_max_drawdown', 'tracking_error_annualized',
                    'min_rolling_3y_cagr', 'observation_count',
                )
            },
            'raw_variables': result.get('raw_variables', []),
            'raw_variable_weights': result.get('raw_variable_weights', {}),
            'classic_metrics': result.get('classic_metrics', {}),
            'period_metrics': _period_metric_records(result.get('period_metrics')),
            'files': {
                'performance': performance_path.relative_to(export_dir).as_posix(),
                'top_holdings': (
                    top_holdings_path.relative_to(export_dir).as_posix()
                    if export_holdings else None
                ),
                'worst_holdings': (
                    worst_holdings_path.relative_to(export_dir).as_posix()
                    if export_holdings else None
                ),
                'html': html_path.relative_to(export_dir).as_posix() if export_html and figure is not None else None,
                'png': png_path.relative_to(export_dir).as_posix() if export_png and figure is not None else None,
            },
        })

    backtest_metrics = _combine_total_and_period_metrics(summary, period_metrics)
    backtest_metrics = _merge_export_table(
        export_dir / 'backtest_metrics.csv', backtest_metrics, replaced_paths,
    )
    backtest_metrics = _finalize_backtest_metrics(backtest_metrics)
    backtest_metrics.to_csv(export_dir / 'backtest_metrics.csv', index=False)
    views = _analysis_views_from_metrics(backtest_metrics)
    registry_path = export_dir / 'backtest_registry.json'
    registry_by_path = {}
    if registry_path.exists():
        with registry_path.open('r', encoding='utf-8') as registry_file:
            previous_registry = json.load(registry_file)
        registry_by_path.update({
            entry.get('test_path'): entry
            for entry in previous_registry
            if entry.get('test_path')
        })
    registry_by_path.update({entry['test_path']: entry for entry in registry})
    registry = list(registry_by_path.values())
    with registry_path.open('w', encoding='utf-8') as registry_file:
        json.dump(registry, registry_file, ensure_ascii=False, indent=2, default=str)

    print(f'Données exportées avant les figures : {export_dir}')
    for figure, html_path, png_path in figure_jobs:
        if export_html:
            figure.write_html(html_path)
        if png_enabled:
            try:
                figure.write_image(png_path)
            except Exception as error:
                png_enabled = False
                print(f'Avertissement : export PNG indisponible ({error}).')

    print(f'Export terminé : {export_dir}')
    return {
        'export_dir': export_dir,
        'metrics': backtest_metrics,
        'summary': views['summary'],
        'composition': views['signal_composition'],
        'classic_metrics': views['classic_metrics'],
        'period_metrics': views['period_metrics'],
        'metrics_by_period': backtest_metrics,
        'registry': registry,
    }

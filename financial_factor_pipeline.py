"""Pipeline de recherche pour les facteurs du secteur financier européen.

Ce module reste dans ``exports`` afin de séparer la recherche sectorielle du
code de production. Il construit des univers propres aux banques, aux
assurances et aux services financiers, exécute des tests Top/Worst relatifs au
segment, puis produit une table compacte qui privilégie la performance et la
stabilité entre régimes.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


PLUGIN_DIR = Path(__file__).resolve().parents[1]
if str(PLUGIN_DIR) not in sys.path:
    sys.path.insert(0, str(PLUGIN_DIR))

from factor_config import signal_options  # noqa: E402
from func import (  # noqa: E402
    calculate_benchmark_performance,
    export_backtest_results,
    load_backtest_data,
    test_composite_signals,
    test_unitary_signals,
)


DATA_DIR = PLUGIN_DIR / "data"
SCREEN_PATH = DATA_DIR / "screen_aggregate.parquet"
RETURNS_PATH = DATA_DIR / "returns.parquet"
DEFAULT_OUTPUT_ROOT = PLUGIN_DIR / "exports" / "financial_sector_research"

MARKETS = {
    "stoxx600": {
        "benchmark": "STOXX EUROPE 600",
        "label": "STOXX Europe 600",
    },
    "europe_small": {
        "benchmark": "MSCI EUR SMALL",
        "label": "Europe Small Cap",
    },
}

# Les codes suivent l'ordre ICB19 utilisé par BacktestEngine.py.
SEGMENTS = {
    "Banks": {"code": 2, "label": "Banques"},
    "Insurance": {"code": 10, "label": "Assurances"},
    "Financial Services": {"code": 6, "label": "Services financiers"},
    "Insurance P&C Overlay": {
        "code": 10,
        "label": "Assurances dommages - module complémentaire",
        "factset_industries": (
            "Property/Casualty Insurance",
            "Specialty Insurance",
            "Multi-Line Insurance",
        ),
    },
}

PERIOD_BREAKPOINTS = [2009, 2013, 2017, 2020, 2022, 2024, 2026]
COMPLETED_PERIOD_IDS = (
    "2009_2012",
    "2013_2016",
    "2017_2019",
    "2020_2021",
    "2022_2023",
    "2024_2025",
)
CURRENT_PERIOD_ID = "since_2026"
LEVEL_AND_TRENDS = ("level", "rank_diff_3", "rank_diff_12")
LEVEL_ONLY = ("level",)


def candidate(
    variable,
    family,
    higher_is_better,
    dimensions=LEVEL_AND_TRENDS,
    numerator=None,
    denominator=None,
    rationale="",
):
    """Décrit un candidat économique et ses transformations autorisées."""
    return {
        "variable": variable,
        "family": family,
        "higher_is_better": bool(higher_is_better),
        "dimensions": tuple(dimensions),
        "numerator": numerator,
        "denominator": denominator,
        "rationale": rationale,
    }


CANDIDATE_REGISTRY = {
    "Banks": [
        candidate(
            "ROE avg FY0", "quality", True,
            rationale="Rentabilité des fonds propres, à confirmer par le capital et le risque de crédit.",
        ),
        candidate(
            "TIER1 Ratio FY0", "quality", True,
            rationale="Coussin de capital bancaire disponible sur les deux marchés.",
        ),
        candidate(
            "Non perf Loans to Total Loans CIQ", "quality", False,
            rationale="Qualité du portefeuille de prêts; une valeur plus faible est préférable.",
        ),
        candidate(
            "NPL / Gross Loans", "quality", False,
            numerator="Non Perf loan CIQ", denominator="Gross Loans CIQ",
            rationale="Ratio de prêts non performants construit sans utiliser la taille absolue du bilan.",
        ),
        candidate(
            "Deposits / Total Assets", "quality", True,
            numerator="Total Deposit CIQ", denominator="Total Asset CIQ",
            rationale="Approximation d'une base de financement par dépôts; elle reste soumise à un contrôle de couverture.",
        ),
        candidate(
            "PB / PTangibleBook FY1", "value", False,
            rationale="Valorisation sur fonds propres tangibles, plus pertinente que EV/EBITDA pour une banque.",
        ),
        candidate(
            "Earns Yield FY1", "value", True,
            rationale="Valorisation bénéficiaire prospective, utilisée avec la qualité pour limiter les value traps.",
        ),
        candidate(
            "Price to Book FY1", "value", False,
            rationale="Valorisation sur fonds propres; candidat redondant à comparer au tangible book.",
        ),
        candidate(
            "EPS Revision Ratio", "momentum", True, LEVEL_ONLY,
            rationale="Révisions bénéficiaires déjà dynamiques; aucune seconde différence n'est testée.",
        ),
        candidate(
            "PMOM 12M1M", "momentum", True, LEVEL_ONLY,
            rationale="Momentum de prix à moyen terme, conservé comme satellite cyclique.",
        ),
        candidate(
            "EPS Growth FY1", "growth", True, LEVEL_ONLY,
            rationale="Croissance bénéficiaire prospective; aucun delta de second ordre n'est testé.",
        ),
        candidate(
            "Gross Income Growth FY1", "growth", True, LEVEL_ONLY,
            rationale="Croissance du produit bancaire, à combiner avec le capital et la qualité du crédit.",
        ),
        candidate(
            "DVD Yield FY1", "dividend", True,
            rationale="Rendement distribué prospectif, sans utiliser un payout monotone non justifié.",
        ),
        candidate(
            "Daily Vol 260J", "lowvol", False,
            rationale="Contrôle de risque transversal et protection contre les bilans fragiles.",
        ),
    ],
    "Insurance": [
        candidate(
            "ROE avg FY0", "quality", True,
            rationale="Rentabilité globale applicable aux assureurs vie, dommages et diversifiés.",
        ),
        candidate(
            "PB / PTangibleBook FY1", "value", False,
            rationale="Valorisation du capital tangible; plus adaptée que les multiples d'entreprise.",
        ),
        candidate(
            "Earns Yield FY1", "value", True,
            rationale="Valorisation prospective des bénéfices, contrôlée par la qualité et la volatilité.",
        ),
        candidate(
            "Price to Book FY1", "value", False,
            rationale="Alternative au tangible book, gardée uniquement si elle apporte une preuve distincte.",
        ),
        candidate(
            "EPS Revision Ratio", "momentum", True, LEVEL_ONLY,
            rationale="Révisions de résultats, sans seconde différence.",
        ),
        candidate(
            "PMOM 12M1M", "momentum", True, LEVEL_ONLY,
            rationale="Momentum de prix, utilisé comme satellite plutôt que comme noyau prudentiel.",
        ),
        candidate(
            "EPS Growth FY1", "growth", True, LEVEL_ONLY,
            rationale="Croissance bénéficiaire prospective, à interpréter avec le cycle de sinistralité.",
        ),
        candidate(
            "Gross Income Growth FY1", "growth", True, LEVEL_ONLY,
            rationale="Croissance du revenu global; le ratio combiné reste réservé au module dommages.",
        ),
        candidate(
            "DVD Yield FY1", "dividend", True,
            rationale="Rendement distribué prospectif, utile pour les assureurs matures.",
        ),
        candidate(
            "Daily Vol 260J", "lowvol", False,
            rationale="Contrôle de risque de marché et de sensibilité au bilan d'investissement.",
        ),
    ],
    "Financial Services": [
        candidate(
            "ROE avg FY0", "quality", True,
            rationale="Rentabilité des fonds propres des gérants, courtiers et groupes financiers.",
        ),
        candidate(
            "PCT OM FY0", "quality", True,
            rationale="Efficacité opérationnelle déjà normalisée dans la source.",
        ),
        candidate(
            "Earns Yield FY1", "value", True,
            rationale="Valorisation bénéficiaire adaptée aux activités de services.",
        ),
        candidate(
            "PB / PTangibleBook FY1", "value", False,
            rationale="Test de valorisation du capital, avec contrôle de redondance face à l'earnings yield.",
        ),
        candidate(
            "Price to Book FY1", "value", False,
            rationale="Alternative simple au tangible book pour les bilans financiers.",
        ),
        candidate(
            "EPS Revision Ratio", "momentum", True, LEVEL_ONLY,
            rationale="Révisions bénéficiaires, sans delta de second ordre.",
        ),
        candidate(
            "PMOM 12M1M", "momentum", True, LEVEL_ONLY,
            rationale="Momentum de prix à moyen terme.",
        ),
        candidate(
            "EPS Growth FY1", "growth", True, LEVEL_ONLY,
            rationale="Croissance bénéficiaire prospective.",
        ),
        candidate(
            "Gross Income Growth FY1", "growth", True, LEVEL_ONLY,
            rationale="Croissance du revenu des activités financières.",
        ),
        candidate(
            "5Y_Hist EPS TrendStab", "growth", True,
            rationale="Persistance historique de la croissance bénéficiaire.",
        ),
        candidate(
            "DVD Yield FY1", "dividend", True,
            rationale="Rendement distribué prospectif.",
        ),
        candidate(
            "Daily Vol 260J", "lowvol", False,
            rationale="Contrôle de risque transversal.",
        ),
    ],
    "Insurance P&C Overlay": [
        candidate(
            "Combined Ratio FY1", "quality", False,
            rationale="Efficacité de souscription prospective, réservée aux assureurs dommages et multi-branches.",
        ),
        candidate(
            "Combined Ratio NTM", "quality", False,
            rationale="Efficacité de souscription sur douze mois, réservée au même sous-univers.",
        ),
        candidate(
            "ROE avg FY0", "quality", True,
            rationale="Ancre de rentabilité qui complète le ratio combiné.",
        ),
        candidate(
            "EPS Revision Ratio", "momentum", True, LEVEL_ONLY,
            rationale="Confirmation bénéficiaire sans seconde différence.",
        ),
        candidate(
            "Daily Vol 260J", "lowvol", False,
            rationale="Contrôle du risque de marché du module dommages.",
        ),
    ],
}


def _signal(higher_is_better=True, level=0.0, rank_diff_3=0.0, rank_diff_12=0.0):
    """Construit une configuration compacte utilisée par les composites."""
    return signal_options(
        higher_is_better=higher_is_better,
        level=level,
        rank_diff_3=rank_diff_3,
        rank_diff_12=rank_diff_12,
    )


COMPOSITE_RECIPES = {
    "Banks": {
        "Banques | coeur qualité-valeur": {
            "ROE avg FY0": _signal(True, level=1.0),
            "TIER1 Ratio FY0": _signal(True, level=0.8),
            "Non perf Loans to Total Loans CIQ": _signal(False, level=0.8),
            "PB / PTangibleBook FY1": _signal(False, level=1.0),
            "Earns Yield FY1": _signal(True, level=1.0),
            "EPS Revision Ratio": _signal(True, level=1.0),
            "DVD Yield FY1": _signal(True, level=0.5),
            "Daily Vol 260J": _signal(False, level=0.5),
        },
        "Banques | coeur avec amélioration": {
            "ROE avg FY0": _signal(True, level=0.8, rank_diff_3=0.2),
            "TIER1 Ratio FY0": _signal(True, level=0.6, rank_diff_12=0.4),
            "Non perf Loans to Total Loans CIQ": _signal(
                False, level=0.6, rank_diff_3=0.4,
            ),
            "PB / PTangibleBook FY1": _signal(False, level=0.8, rank_diff_3=0.2),
            "Earns Yield FY1": _signal(True, level=0.8, rank_diff_3=0.2),
            "EPS Revision Ratio": _signal(True, level=1.0),
            "PMOM 12M1M": _signal(True, level=0.5),
        },
        "Banques | cycle bénéficiaire": {
            "ROE avg FY0": _signal(True, level=1.0),
            "Earns Yield FY1": _signal(True, level=1.0),
            "EPS Revision Ratio": _signal(True, level=1.2),
            "PMOM 12M1M": _signal(True, level=0.7),
            "EPS Growth FY1": _signal(True, level=0.6),
            "Gross Income Growth FY1": _signal(True, level=0.6),
        },
    },
    "Insurance": {
        "Assurances | noyau persistant (hypothèse dérivée)": {
            "ROE avg FY0": _signal(True, rank_diff_3=1.0),
            "Price to Book FY1": _signal(False, rank_diff_3=1.0),
            "DVD Yield FY1": _signal(True, rank_diff_3=0.75),
            "Earns Yield FY1": _signal(True, rank_diff_12=0.75),
        },
        "Assurances | coeur qualité-valeur": {
            "ROE avg FY0": _signal(True, level=1.0),
            "PB / PTangibleBook FY1": _signal(False, level=1.0),
            "Earns Yield FY1": _signal(True, level=1.0),
            "EPS Revision Ratio": _signal(True, level=1.0),
            "DVD Yield FY1": _signal(True, level=0.6),
            "Daily Vol 260J": _signal(False, level=0.6),
        },
        "Assurances | coeur avec amélioration": {
            "ROE avg FY0": _signal(True, level=0.8, rank_diff_3=0.2),
            "PB / PTangibleBook FY1": _signal(False, level=0.8, rank_diff_3=0.2),
            "Earns Yield FY1": _signal(True, level=0.8, rank_diff_3=0.2),
            "EPS Revision Ratio": _signal(True, level=1.0),
            "PMOM 12M1M": _signal(True, level=0.5),
            "Daily Vol 260J": _signal(False, level=0.5),
        },
        "Assurances | cycle bénéficiaire": {
            "ROE avg FY0": _signal(True, level=1.0),
            "Earns Yield FY1": _signal(True, level=1.0),
            "EPS Revision Ratio": _signal(True, level=1.2),
            "PMOM 12M1M": _signal(True, level=0.7),
            "EPS Growth FY1": _signal(True, level=0.6),
            "Gross Income Growth FY1": _signal(True, level=0.6),
        },
    },
    "Financial Services": {
        "Services financiers | coeur diversifié": {
            "ROE avg FY0": _signal(True, level=1.0),
            "PCT OM FY0": _signal(True, level=0.8),
            "Earns Yield FY1": _signal(True, level=1.0),
            "PB / PTangibleBook FY1": _signal(False, level=0.7),
            "EPS Revision Ratio": _signal(True, level=1.0),
            "DVD Yield FY1": _signal(True, level=0.5),
            "Daily Vol 260J": _signal(False, level=0.5),
        },
        "Services financiers | coeur avec amélioration": {
            "ROE avg FY0": _signal(True, level=0.8, rank_diff_3=0.2),
            "PCT OM FY0": _signal(True, level=0.8, rank_diff_3=0.2),
            "Earns Yield FY1": _signal(True, level=0.8, rank_diff_3=0.2),
            "PB / PTangibleBook FY1": _signal(False, level=0.7, rank_diff_3=0.3),
            "EPS Revision Ratio": _signal(True, level=1.0),
            "PMOM 12M1M": _signal(True, level=0.5),
        },
        "Services financiers | bénéfices persistants": {
            "ROE avg FY0": _signal(True, level=1.0),
            "Earns Yield FY1": _signal(True, level=1.0),
            "EPS Revision Ratio": _signal(True, level=1.2),
            "PMOM 12M1M": _signal(True, level=0.6),
            "EPS Growth FY1": _signal(True, level=0.5),
            "Gross Income Growth FY1": _signal(True, level=0.5),
            "5Y_Hist EPS TrendStab": _signal(True, level=0.7),
        },
    },
    "Insurance P&C Overlay": {
        "Assurances dommages | discipline de souscription": {
            "Combined Ratio FY1": _signal(False, level=1.0),
            "Combined Ratio NTM": _signal(False, level=0.5),
            "ROE avg FY0": _signal(True, level=1.0),
            "Daily Vol 260J": _signal(False, level=0.5),
        },
        "Assurances dommages | souscription en amélioration": {
            "Combined Ratio FY1": _signal(False, level=0.7, rank_diff_3=0.3),
            "Combined Ratio NTM": _signal(False, level=0.5, rank_diff_12=0.5),
            "ROE avg FY0": _signal(True, level=1.0),
            "EPS Revision Ratio": _signal(True, level=0.8),
        },
    },
}


def candidate_table():
    """Retourne le registre économique sous forme tabulaire."""
    rows = []
    for segment, candidates in CANDIDATE_REGISTRY.items():
        for item in candidates:
            rows.append({"segment": segment, **item})
    return pd.DataFrame(rows)


def _source_columns():
    """Liste les colonnes brutes nécessaires aux candidats et aux filtres."""
    columns = ["FactSet Ind"]
    for candidates in CANDIDATE_REGISTRY.values():
        for item in candidates:
            columns.append(item.get("numerator") or item["variable"])
            if item.get("denominator"):
                columns.append(item["denominator"])
    return list(dict.fromkeys(columns))


def _materialize_ratios(screen):
    """Construit les ratios financiers explicites avant toute imputation."""
    ratio_specs = {}
    for candidates in CANDIDATE_REGISTRY.values():
        for item in candidates:
            if item.get("numerator") and item.get("denominator"):
                ratio_specs[item["variable"]] = (
                    item["numerator"], item["denominator"],
                )
    for name, (numerator, denominator) in ratio_specs.items():
        values = pd.to_numeric(screen[numerator], errors="coerce")
        bases = pd.to_numeric(screen[denominator], errors="coerce")
        screen[name] = values.div(bases.where(bases.ne(0)))
        screen[name] = screen[name].replace([np.inf, -np.inf], np.nan)
    return screen


def _segment_screen(screen, segment):
    """Isole un segment ICB et applique le filtre FactSet éventuel."""
    definition = SEGMENTS[segment]
    subset = screen.loc[
        pd.to_numeric(
            screen[" Benchmark ICB Supersector "], errors="coerce",
        ).eq(definition["code"])
    ].copy()
    industries = definition.get("factset_industries")
    if industries:
        subset = subset.loc[subset["FactSet Ind"].isin(industries)].copy()
    return subset


def _coverage_for_candidate(screen, item, minimum_coverage, minimum_median_names):
    """Mesure la couverture sans combler les observations manquantes."""
    values = pd.to_numeric(screen[item["variable"]], errors="coerce")
    finite = values.replace([np.inf, -np.inf], np.nan).notna()
    monthly_valid = finite.groupby(pd.to_datetime(screen["Date"])).sum()
    coverage = float(finite.mean()) if len(finite) else 0.0
    median_names = float(monthly_valid.median()) if not monthly_valid.empty else 0.0
    minimum_names = int(monthly_valid.min()) if not monthly_valid.empty else 0
    selected = coverage >= minimum_coverage and median_names >= minimum_median_names
    reasons = []
    if coverage < minimum_coverage:
        reasons.append("couverture insuffisante")
    if median_names < minimum_median_names:
        reasons.append("trop peu de titres valides par mois")
    return {
        "segment": None,
        "variable": item["variable"],
        "family": item["family"],
        "coverage": coverage,
        "missing_pct": (1.0 - coverage) * 100.0,
        "median_valid_names": median_names,
        "minimum_valid_names": minimum_names,
        "selected": selected,
        "selection_reason": "retenue" if selected else "; ".join(reasons),
        "dimensions": ",".join(item["dimensions"]),
        "higher_is_better": item["higher_is_better"],
        "rationale": item["rationale"],
    }


def audit_segment_coverage(
    screen,
    segment,
    minimum_coverage=0.60,
    minimum_median_names=10,
):
    """Applique le filtre de couverture préenregistré à un segment."""
    rows = []
    for item in CANDIDATE_REGISTRY[segment]:
        row = _coverage_for_candidate(
            screen, item, minimum_coverage, minimum_median_names,
        )
        row["segment"] = segment
        rows.append(row)
    return pd.DataFrame(rows)


def _unitary_groups(segment, coverage):
    """Regroupe les candidats par ensemble de dimensions autorisées."""
    selected = set(coverage.loc[coverage["selected"], "variable"])
    grouped = defaultdict(dict)
    for item in CANDIDATE_REGISTRY[segment]:
        if item["variable"] not in selected:
            continue
        grouped[item["dimensions"]][item["variable"]] = signal_options(
            higher_is_better=item["higher_is_better"],
        )
    return grouped


def _filtered_composites(segment, coverage):
    """Retire d'une recette les signaux qui échouent au filtre de couverture."""
    selected = set(coverage.loc[coverage["selected"], "variable"])
    filtered = {}
    for name, recipe in COMPOSITE_RECIPES[segment].items():
        available = {
            variable: options
            for variable, options in recipe.items()
            if variable in selected
        }
        if len(available) >= 3:
            filtered[name] = available
    return filtered


def _slug(text):
    """Produit un identifiant de dossier stable et lisible."""
    return (
        str(text).lower()
        .replace("&", "and")
        .replace(" ", "_")
        .replace("|", "_")
        .replace("/", "_")
    )


def _run_segment(
    market_key,
    segment,
    screen,
    returns,
    output_dir,
    start_date,
    percentile,
    n_jobs,
    minimum_coverage,
    minimum_median_names,
):
    """Exécute les tests unitaires et composites d'un segment."""
    market = MARKETS[market_key]
    segment_data = _segment_screen(screen, segment)
    if segment_data.empty:
        raise ValueError(f"Le segment {segment} est vide pour {market['label']}.")

    coverage = audit_segment_coverage(
        segment_data,
        segment,
        minimum_coverage=minimum_coverage,
        minimum_median_names=minimum_median_names,
    )
    selected_count = int(coverage["selected"].sum())
    print(
        f"Segment {SEGMENTS[segment]['label']} : {len(segment_data):,} observations, "
        f"{segment_data['Date'].nunique()} dates, {selected_count} candidats retenus."
    )

    benchmark_performance = calculate_benchmark_performance(
        segment_data,
        returns,
        bench=market["benchmark"],
        start_date=start_date,
    )
    run_options = {
        "bench": market["benchmark"],
        "bench_perf": benchmark_performance,
        "percentile": percentile,
        "start_date": start_date,
        "freq_rebal": 1,
        "fill_method": "drift",
        "n_jobs": n_jobs,
        "retain_builders": False,
        "monthly_base_cache": {},
        "period_breakpoints": PERIOD_BREAKPOINTS,
        "show_plot": False,
        "build_figure": False,
    }

    unitary_results = {}
    working_screen = segment_data
    for dimensions, signal_config in _unitary_groups(segment, coverage).items():
        batch = test_unitary_signals(
            screen=working_screen,
            returns=returns,
            signal_config=signal_config,
            list_noire_path=None,
            dimensions=dimensions,
            **run_options,
        )
        working_screen = batch["screen"]
        unitary_results.update(batch["results"])

    composite_results = {}
    composites = _filtered_composites(segment, coverage)
    if composites:
        batch = test_composite_signals(
            screen=working_screen,
            returns=returns,
            composite_configs=composites,
            list_noire_path=None,
            **run_options,
        )
        working_screen = batch["screen"]
        composite_results.update(batch["results"])

    segment_slug = _slug(segment)
    export = export_backtest_results(
        results={
            "unitary": {"screen": working_screen, "results": unitary_results},
            "composite": {"screen": working_screen, "results": composite_results},
        },
        output_dir=output_dir,
        export_name=segment_slug,
        export_html=False,
        export_png=False,
        export_holdings=False,
    )
    segment_dir = Path(export["export_dir"])
    coverage.to_csv(segment_dir / "coverage_audit.csv", index=False)
    return {
        "segment": segment,
        "segment_dir": segment_dir,
        "coverage": coverage,
        "metrics": export["metrics_by_period"],
        "composites": composites,
    }


def _total_gate(frame):
    """Applique la porte stricte de performance totale."""
    return (
        frame["active_cagr"].gt(0)
        & frame["top_worst_cagr"].gt(0)
        & frame["top_information_ratio"].gt(0)
        & frame["robust_score"].gt(0)
    )


def build_ranked_results(metrics):
    """Classe les résultats en séparant rendement total, stabilité et période courante."""
    if metrics.empty:
        return pd.DataFrame()
    total = metrics.loc[metrics["scope"].eq("total")].copy()
    total["total_gate"] = _total_gate(total)

    periods = metrics.loc[
        metrics["period_id"].isin(COMPLETED_PERIOD_IDS)
    ].copy()
    periods["period_pass"] = (
        periods["active_cagr"].gt(0)
        & periods["top_worst_cagr"].gt(0)
        & periods["top_information_ratio"].gt(0)
    )
    stability = periods.groupby(
        ["market", "segment", "test_path"], as_index=False,
    ).agg(
        completed_periods=("period_id", "nunique"),
        positive_periods=("period_pass", "sum"),
        positive_active_periods=("active_cagr", lambda x: int(x.gt(0).sum())),
        median_active_cagr=("active_cagr", "median"),
        worst_active_cagr=("active_cagr", "min"),
        median_information_ratio=("top_information_ratio", "median"),
        worst_top_worst_cagr=("top_worst_cagr", "min"),
    )
    stability["period_pass_rate"] = (
        stability["positive_periods"] / stability["completed_periods"]
    )

    current = metrics.loc[
        metrics["period_id"].eq(CURRENT_PERIOD_ID),
        [
            "market", "segment", "test_path", "years", "active_cagr",
            "top_worst_cagr", "top_information_ratio", "robust_score",
        ],
    ].rename(columns={
        "years": "current_years",
        "active_cagr": "current_active_cagr",
        "top_worst_cagr": "current_top_worst_cagr",
        "top_information_ratio": "current_information_ratio",
        "robust_score": "current_robust_score",
    })

    ranked = total.merge(
        stability, on=["market", "segment", "test_path"], how="left",
    ).merge(
        current, on=["market", "segment", "test_path"], how="left",
    )
    ranked["persistent_gate"] = (
        ranked["total_gate"]
        & ranked["completed_periods"].eq(len(COMPLETED_PERIOD_IDS))
        & ranked["positive_periods"].ge(4)
        & ranked["worst_active_cagr"].gt(-0.05)
    )
    ranked["current_outperformance"] = (
        ranked["current_active_cagr"].gt(0)
        & ranked["current_information_ratio"].gt(0)
    )
    ranked = ranked.sort_values(
        [
            "market", "segment", "persistent_gate", "total_gate",
            "period_pass_rate", "worst_active_cagr", "robust_score",
        ],
        ascending=[True, True, False, False, False, False, False],
    )
    return ranked.reset_index(drop=True)


def _attach_candidate_metadata(metrics):
    """Ajoute la famille économique aux tests unitaires et composites."""
    lookup = {}
    for segment, candidates in CANDIDATE_REGISTRY.items():
        for item in candidates:
            for dimension in item["dimensions"]:
                lookup[(segment, f"{item['variable']} | {dimension}")] = {
                    "family": item["family"],
                    "economic_variable": item["variable"],
                }
    output = metrics.copy()
    metadata = [
        lookup.get(
            (row.segment, row.test_name),
            {
                "family": "composite" if row.test_type == "composite" else "unknown",
                "economic_variable": row.test_name,
            },
        )
        for row in output.itertuples()
    ]
    output["family"] = [item["family"] for item in metadata]
    output["economic_variable"] = [
        item["economic_variable"] for item in metadata
    ]
    return output


def _shareable_table(ranked, segment, limit=12):
    """Prépare une table courte limitée aux résultats qui battent le benchmark."""
    columns = [
        "test_name", "test_type", "family", "economic_variable",
        "persistent_gate", "total_gate", "active_cagr",
        "top_worst_cagr", "top_information_ratio", "robust_score",
        "positive_periods", "completed_periods", "worst_active_cagr",
        "current_active_cagr", "current_information_ratio",
    ]
    subset = ranked.loc[
        ranked["segment"].eq(segment) & ranked["active_cagr"].gt(0),
        columns,
    ].head(limit)
    return subset


def write_shareable_results(ranked, output_dir, market_label):
    """Écrit un bloc texte que l'utilisateur peut copier dans une conversation."""
    lines = [
        f"RESULTATS FINANCIERS | {market_label}",
        "Filtre d'affichage : active CAGR totale > 0.",
        (
            "persistent_gate = porte totale stricte + au moins 4/6 périodes "
            "complètes positives + pire active CAGR > -5 %."
        ),
        "La période 2026 est tactique car elle dure moins d'un an.",
    ]
    for segment in SEGMENTS:
        lines.extend(["", f"### {SEGMENTS[segment]['label']}"])
        table = _shareable_table(ranked, segment)
        if table.empty:
            lines.append("Aucun test avec active CAGR totale positive.")
        else:
            lines.append(
                table.to_csv(
                    index=False,
                    float_format="%.6f",
                    lineterminator="\n",
                ).strip()
            )
    path = Path(output_dir) / "shareable_results.txt"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def print_shareable_results(output_dir):
    """Imprime le fichier compact destiné au partage après un rerun."""
    path = Path(output_dir) / "shareable_results.txt"
    if not path.exists():
        raise FileNotFoundError(f"Résultat partageable absent : {path}")
    print(path.read_text(encoding="utf-8"))


def run_market_experiment(
    market_key,
    output_root=DEFAULT_OUTPUT_ROOT,
    start_date="2009-02-01",
    percentile=0.20,
    n_jobs=1,
    minimum_coverage=0.60,
    minimum_median_names=10,
):
    """Exécute l'expérience complète d'un marché et retourne ses tables finales."""
    if market_key not in MARKETS:
        raise KeyError(f"Marché inconnu : {market_key}. Choix : {sorted(MARKETS)}")
    market = MARKETS[market_key]
    output_dir = Path(output_root) / market_key
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Chargement du marché {market['label']}.")
    screen, returns = load_backtest_data(
        screen_path=SCREEN_PATH,
        returns_path=RETURNS_PATH,
        variables=_source_columns(),
        bench=market["benchmark"],
        start_date=start_date,
        lookback_periods=12,
        compact_dtypes=True,
    )
    screen["Date"] = pd.to_datetime(screen["Date"])
    screen = _materialize_ratios(screen)
    weight_column = f"Weight in {market['benchmark']}"

    # Le rang et l'imputation sont calculés uniquement parmi les constituants du marché.
    screen = screen.loc[
        pd.to_numeric(screen[weight_column], errors="coerce").fillna(0).gt(0)
    ].copy()
    financial_codes = {definition["code"] for definition in SEGMENTS.values()}
    screen = screen.loc[
        pd.to_numeric(
            screen[" Benchmark ICB Supersector "], errors="coerce",
        ).isin(financial_codes)
    ].copy()

    segment_results = []
    coverage_tables = []
    metric_tables = []
    for segment in SEGMENTS:
        result = _run_segment(
            market_key=market_key,
            segment=segment,
            screen=screen,
            returns=returns,
            output_dir=output_dir,
            start_date=start_date,
            percentile=percentile,
            n_jobs=n_jobs,
            minimum_coverage=minimum_coverage,
            minimum_median_names=minimum_median_names,
        )
        segment_results.append(result)
        coverage = result["coverage"].copy()
        coverage.insert(0, "market", market_key)
        coverage_tables.append(coverage)
        metrics = result["metrics"].copy()
        metrics.insert(0, "segment", result["segment"])
        metrics.insert(0, "market", market_key)
        metric_tables.append(metrics)

    coverage = pd.concat(coverage_tables, ignore_index=True)
    metrics = pd.concat(metric_tables, ignore_index=True)
    metrics = _attach_candidate_metadata(metrics)
    ranked = build_ranked_results(metrics)

    coverage.to_csv(output_dir / "coverage_audit.csv", index=False)
    metrics.to_csv(output_dir / "all_backtest_metrics.csv", index=False)
    ranked.to_csv(output_dir / "ranked_results.csv", index=False)
    shareable_path = write_shareable_results(ranked, output_dir, market["label"])

    manifest = {
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "market_key": market_key,
        "market_label": market["label"],
        "benchmark": market["benchmark"],
        "screen_path": str(SCREEN_PATH),
        "returns_path": str(RETURNS_PATH),
        "start_date": start_date,
        "period_breakpoints": PERIOD_BREAKPOINTS,
        "completed_period_ids": COMPLETED_PERIOD_IDS,
        "percentile": percentile,
        "fill_method": "drift",
        "minimum_coverage": minimum_coverage,
        "minimum_median_names": minimum_median_names,
        "n_jobs": n_jobs,
        "selected_constituent_rows": int(len(screen)),
        "selected_dates": int(screen["Date"].nunique()),
        "tests_total": int(metrics["scope"].eq("total").sum()),
        "notes": [
            "Le total chevauche les sous-periodes et ne constitue pas une validation independante.",
            "La periode depuis 2026 est annualisee sur moins d'un an et reste tactique.",
            "Le module Combined Ratio est limite au sous-univers assurance dommages.",
            "Les multiples EV/EBITDA, P/FCF et les ratios de dette industrielle sont exclus.",
            (
                "Le noyau persistant des assurances est une hypothese derivee apres "
                "screening croise; son backtest reste in-sample et demande une validation future."
            ),
        ],
    }
    (output_dir / "experiment_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"Expérience terminée : {output_dir}")
    print(f"Résultat à partager : {shareable_path}")
    return {
        "output_dir": output_dir,
        "manifest": manifest,
        "coverage": coverage,
        "metrics": metrics,
        "ranked": ranked,
        "segments": segment_results,
        "shareable_path": shareable_path,
    }


def _parse_args():
    """Lit les paramètres de la ligne de commande pour une exécution contrôlée."""
    parser = argparse.ArgumentParser(
        description="Recherche de facteurs financiers par marché européen.",
    )
    parser.add_argument("--market", choices=sorted(MARKETS), required=True)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--start-date", default="2009-02-01")
    parser.add_argument("--percentile", type=float, default=0.20)
    parser.add_argument("--n-jobs", type=int, default=1)
    parser.add_argument("--minimum-coverage", type=float, default=0.60)
    parser.add_argument("--minimum-median-names", type=int, default=10)
    return parser.parse_args()


def main():
    """Point d'entrée de l'expérience en ligne de commande."""
    args = _parse_args()
    run_market_experiment(
        market_key=args.market,
        output_root=args.output_root,
        start_date=args.start_date,
        percentile=args.percentile,
        n_jobs=args.n_jobs,
        minimum_coverage=args.minimum_coverage,
        minimum_median_names=args.minimum_median_names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

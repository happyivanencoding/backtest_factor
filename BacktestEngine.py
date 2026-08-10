import numpy as np
import pandas as pd
import scipy
import datetime
import os
import copy
import math
# from scipy import stats
from multiprocessing import Pool
from dateutil.relativedelta import relativedelta
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re
import warnings
from typing import Optional, Tuple
from pathlib import Path


def build_periods_from_breakpoints(period_breakpoints):
    """Transforme des années de rupture en segments chronologiques non chevauchants."""
    try:
        breakpoints = sorted({int(year) for year in period_breakpoints})
    except (TypeError, ValueError) as error:
        raise ValueError(
            'Chaque point de rupture doit être une année entière, par exemple 2020.'
        ) from error
    if not breakpoints:
        return [{
            'id': 'all',
            'label': 'Échantillon complet',
            'start': None,
            'end': None,
        }]
    if any(year < 1000 or year > 9999 for year in breakpoints):
        raise ValueError('Chaque point de rupture doit contenir quatre chiffres.')

    periods = [{
        'id': f'before_{breakpoints[0]}',
        'label': f'Avant {breakpoints[0]}',
        'start': None,
        'end': f'{breakpoints[0] - 1}-12-31',
    }]
    for start_year, next_year in zip(breakpoints, breakpoints[1:]):
        periods.append({
            'id': f'{start_year}_{next_year - 1}',
            'label': f'{start_year}-{next_year - 1}',
            'start': f'{start_year}-01-01',
            'end': f'{next_year - 1}-12-31',
        })
    periods.append({
        'id': f'since_{breakpoints[-1]}',
        'label': f'Depuis {breakpoints[-1]}',
        'start': f'{breakpoints[-1]}-01-01',
        'end': None,
    })
    return periods


def build_rolling_periods(start_date, end_date, window_months, step_months):
    """Construit des fenêtres calendaires complètes alignées sur la fin commune."""
    start = pd.Timestamp(start_date).normalize()
    end = pd.Timestamp(end_date).normalize()
    if pd.isna(start) or pd.isna(end) or start > end:
        raise ValueError('La plage commune des fenêtres glissantes est invalide.')
    if not isinstance(window_months, int) or window_months < 1:
        raise ValueError('window_months doit être un entier strictement positif.')
    if not isinstance(step_months, int) or step_months < 1:
        raise ValueError('step_months doit être un entier strictement positif.')

    periods = []
    window_end = end
    while True:
        window_start = (
            window_end - relativedelta(months=window_months)
            + datetime.timedelta(days=1)
        )
        if window_start < start:
            break
        start_text = window_start.date().isoformat()
        end_text = window_end.date().isoformat()
        periods.append({
            'id': (
                f'rolling_{window_months}m_'
                f'{window_start:%Y%m%d}_{window_end:%Y%m%d}'
            ),
            'label': f'{window_months} mois | {start_text} → {end_text}',
            'start': start_text,
            'end': end_text,
            'period_group': f'rolling_{window_months}m',
            'window_months': window_months,
            'step_months': step_months,
            'robust_score_comparable': window_months >= 36,
        })
        window_end = window_end - relativedelta(months=step_months)
    return list(reversed(periods))


def drop_duplicates_keep_less_missing(screen):
    """
    This function removes duplicate rows based on ISIN and Date, 
    keeping the version of each duplicate that has the fewest missing values (i.e., the most complete data).
    """
    screen = screen.reset_index()

    subset = ['ISIN', 'Date']  # columns that define duplicates

    # Score rows: higher = better (fewer NaNs)
    score = screen.notna().sum(axis=1)

    # Sort so the "best" row in each duplicate group comes first
    df_best = (
        screen.assign(_score=score)
            .sort_values(subset + ['_score'], ascending=[True] * len(subset) + [False])
            .drop_duplicates(subset=subset, keep='first')
            .drop(columns='_score')
    )

    df_best = df_best.set_index("ISIN")

    return df_best


def  read_liste_noire(file_list_noire, override_exclusion=[], override_inclusion=[], key="ISIN", exclu_type=["ex_all"]):
    """
    Cette fonction va sortir les ISINs exclus par le groupe.
    exclu_type = ["ex_all"] or ["ex_all", "Controverse"]
    """
    liste_noire = pd.read_excel(file_list_noire)

    # Filtrer les lignes où au moins une des colonnes de exclu_type vaut 1
    filtre = liste_noire[exclu_type].fillna(0).astype(int).any(axis=1)
    liste_noire = liste_noire[filtre]

    liste_noire = liste_noire.dropna(subset=key)[key].tolist()
    if len(override_exclusion) > 0 :
        liste_noire = np.concatenate([liste_noire,np.array(override_exclusion)])
    liste_noire_unique = np.unique(liste_noire)
    liste_noire_finale = list(set(liste_noire_unique) - set(override_inclusion))
    return liste_noire_finale

def merge_weight_by_pairs(df: pd.DataFrame,
                        pairs,
                        weight_col='Weight in MSCI WORLD',
                        drop_second=True): 
    """
    Combiner le poids des entreprises doublons dans le benchmark choisi (par défaut MSCI WORLD)
    Liste à complérer manuellement une fois constatée
    """
    # Ensure "ISIN" is the index of df
    if df.index.name != "ISIN" and "ISIN" in df.columns:
        df.set_index("ISIN", inplace=True)

    # Ensure the weight column is numeric (coerce errors to NaN)
    if weight_col not in df.columns:
        raise KeyError(f"Column '{weight_col}' not found in DataFrame.")

    for keep, drop in pairs:
        has_keep = keep in df.index
        has_drop = drop in df.index

        if has_keep and has_drop: # Only when entreprisese existent dans le screen
            w_keep = df.at[keep, weight_col]
            w_drop = df.at[drop, weight_col] # same as loc but only for one value

            df.at[keep, weight_col] = w_keep + w_drop

            if drop_second:
                # errors='ignore' avoids exceptions if it was dropped elsewhere
                df.drop(index=drop, inplace=True, errors='ignore')
    return df

def merge_ticker_secondaire(df, bench):
        """
        Préparer les merges des ISINs doublons
        Combiner le poids des entreprises doublons dans le benchmark choisi (par défaut MSCI WORLD)
        Liste à complérer manuellement une fois constatée

        """
        isin_pairs = [
                        "US02079K3059", # Google
                        "US02079K1079",

                        "DK0010244508", # A.P. Moller
                        "DK0010244425", 
                        
                        "SE0017486889", # Atlas Copco
                        "SE0017486897",

                        "DE0005190003", # Bayerische Motoren Werke
                        "DE0005190037",

                        "SE0015658109", # Epiroc
                        "SE0015658117",

                        "CH0012032048",
                        "CH0012032113", # Roche Holding

                        "CH0024638196",
                        "CH0024638212", # Schindler


                        "CH0010570767", # Lindt
                        "CH0010570759",

                        "DE0006048432", # Henkel
                        "DE0006048408",

                        "SE0000107203", # Industrivarden
                        "SE0000190126"
                ]

        # Convert to list of (keep, drop) pairs in order
        if len(isin_pairs) % 2 != 0:
                raise ValueError("The ISIN list length must be even (pairs of 2).")

        pairs = list(zip(isin_pairs[::2], isin_pairs[1::2]))

        df = merge_weight_by_pairs(
                        df=df,
                        pairs=pairs,
                        weight_col=f'Weight in {bench}',
                        drop_second=True    # drop the second ISIN after merging
                        )
        return df

class PtfBuilder:
    def __init__(self,
                screen, 
                returns, 
                bench, 
                percentile, 
                metrics, 
                ptf_name = "PTF TEST", 
                ponderation='Racine cube',
                esg_exclusion=0.2,
                cut_mkt_cap=0,
                liste_noire=r"\\groupe-ufg.com\commun\Prive\GestionAM\Ingenierie_Financiere\PROD\_BASE\_ ESG DATA\Liste_Noire_Exclusion.xlsx",
                reco_secto = [0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0],
                reco_facto = [0,0,0,0,0],
                score_neutral="ICB 19", 
                weight_neutral="ICB 19",
                Top=True,
                top_mandatory = None, 
                multiprocessing=False,
                mode_monthly_prod=False, 
                output_dir=None,
                cap_weight_threshold=None,
                score_pivot_esg=None,    # score_pivot_esg = "INDEX MSCI WORLD_vs_1330696"
                score_pivot_esg_path=r"\\groupe-ufg.com\commun\Public\DIRR\Data\riskindics\notes pivots",
                bench_perf=None,
                monthly_base_cache=None):
        """
        initialisation des paramètres de la classe 

        order for "reco_secto" :
            1: "Auto & Parts",  
            2: "Banks",  
            3: "Basic Resources",  
            4: "Chemicals",  
            5: "Construction",  
            6: "Financial Services",  
            7: "Food, Beverage & Tobacco",  
            8: "Health Care",  
            9: "Industrial Goods & Services",  
            10: "Insurance",  
            11: "Media",  
            12: "Energy",  
            13: "Personal & Household Goods",  
            14: "Real Estate",  
            15: "Retail",  
            16: "Technology",  
            17: "Telecommunications",  
            18: "Travel & Leisure",  
            19: "Utilities"  
        
        order for "reco_facto" :
            1: "Growth",
            2: "Low Vol",
            3: "Momentum",
            4: "Quality",
            5: "Value"

        """
        if ponderation not in ["Racine cube","Racine carrée", "Market cap","Log","Equalweight"]:
            print(" ponderation must be Racine cube, Racine carrée, Market cap, Log or Equalweight")
        else:
            self.ponderation=ponderation

        self.bench=bench
        self.percentile=percentile
        self.cut_mkt_cap=cut_mkt_cap
        self.metrics=metrics
        self.ptf_name=ptf_name
        self.score_neutral=score_neutral
        self.weight_neutral=weight_neutral
        self.esg_exclusion=esg_exclusion
        self._liste_noire=liste_noire
        self.top_mandatory=top_mandatory
        self.multiprocessing=multiprocessing
        self.sec_list_monthly=None
        self.sec_list_historical=None
        self.perf_ptf=None
        self.perf_bench = copy.deepcopy(bench_perf) if bench_perf is not None else None
        self.buy_list=None
        self.Top = Top
        self.mode_monthly_prod = mode_monthly_prod
        self.output_dir = output_dir
        self.cap_weight_threshold = cap_weight_threshold
        self.score_pivot_esg = score_pivot_esg
        self.score_pivot_esg_path = score_pivot_esg_path
        self.monthly_base_cache = monthly_base_cache

        if type(screen) not in [str,type(pd.DataFrame())]:
            print("screen must be string or DataFrame")
        else:
            self.screen=screen

        if type(returns) !=type(pd.DataFrame()):
            print("returns must be DataFrame")
        else:
            self.returns=returns

        if type(reco_secto) not in [str, list, type(pd.DataFrame())]:
            print("reco_secto must be list or DataFrame")
        else:
            self.reco_secto=copy.deepcopy(reco_secto)

        if type(reco_facto) not in [str, list, type(pd.DataFrame())]:
            print("reco_facto must be list or DataFrame")
        else:
            self.reco_facto=copy.deepcopy(reco_facto)


    def adjust_companies_ponderation(self,df):
        """
        pondération des Benchmarck Market Value pour réduire les effets de taille
        
        """
        df = df.copy()

        if self.ponderation == "Racine cube":
            df.loc[:, 'Benchmark Market Value Millions in EUR '] = df['Benchmark Market Value Millions in EUR ']**(1/3)
        elif self.ponderation == "Racine carrée":
            df.loc[:, 'Benchmark Market Value Millions in EUR '] = df['Benchmark Market Value Millions in EUR ']**(1/2)
        elif self.ponderation == "Market cap":
            df.loc[:, 'Benchmark Market Value Millions in EUR '] = df['Benchmark Market Value Millions in EUR ']
        elif self.ponderation == "Log":
            df.loc[:, 'Benchmark Market Value Millions in EUR '] = np.log(df['Benchmark Market Value Millions in EUR '])
        elif self.ponderation == "Equalweight":
            df['Benchmark Market Value Millions in EUR '] = 1/len(df)

        return df
    
    def filtrage_esg_liste_noire(self, df, date):
        """
        Filtrage en fonction des performances ESG et de la liste noire.
        Retourne uniquement le DataFrame filtré.
        """
        import copy
        df_esg = copy.deepcopy(df)
        # ESG filtering
        if date.year >= 2014 and isinstance(self.score_pivot_esg, float):
            df_esg = df[df['ESG_ANALYST_SCORE'] > self.score_pivot_esg]

        elif date.year >= 2014 and self.esg_exclusion > 0:  # this will entrer only when if date.year >= 2014 and isinstance(self.score_pivot_esg, float) is FALSE
            esg_pct = df['ESG_ANALYST_SCORE'].rank(pct=True)
            df_esg = df.loc[esg_pct >= self.esg_exclusion]


        # Blacklist filtering
        if self._liste_noire != None:
            if isinstance(self._liste_noire, str):
                self._liste_noire = read_liste_noire(self._liste_noire, [], [])

            if 'ISIN' in df_esg.columns:
                df_esg = df_esg[~df_esg['ISIN'].isin(self._liste_noire)]
            elif df_esg.index.name == 'ISIN':
                df_esg = df_esg[~df_esg.index.isin(self._liste_noire)]

        return df_esg


    def find_esg_pivot_file_path(self):
        """
        Trouver le dernier fichier à jour pour le score pivot ESG
        """
        DATE_8DIG_RE = re.compile(r"(\d{8})")

        def parse_yyyymmdd(s: str) -> Optional[datetime.date]:
            """Parse YYYYMMDD string into date, return None if invalid."""
            from datetime import datetime
            try:
                return datetime.strptime(s, "%Y%m%d").date()
            except ValueError:
                return None

        def first_date_in_name(name: str) -> Optional[datetime.date]:
            """
            Return the first valid YYYYMMDD date found in a string (first 8 consecutive digits).
            If no valid 8-digit date exists, return None.
            """
            m = DATE_8DIG_RE.search(name)
            if not m:
                return None
            return parse_yyyymmdd(m.group(1))

        def get_most_recent_dated_subfolder(base_dir: Path) -> Tuple[Optional[Path], Optional[datetime.date], str]:
            """
            Among immediate subfolders of base_dir, select the one with the largest date
            extracted from the first 8 digits in the folder name.

            Fallback: if no subfolder has a valid 8-digit date in the name, pick the
            most recently modified subfolder.

            Returns: (folder_path, extracted_date_or_None, selection_mode)
                    selection_mode in {"by_name_date", "by_mtime", "none"}
            """
            if not base_dir.exists():
                raise FileNotFoundError(f"Base directory does not exist: {base_dir}")
            if not base_dir.is_dir():
                raise NotADirectoryError(f"Not a directory: {base_dir}")

            dated = []
            subdirs = []

            with os.scandir(base_dir) as it:
                for entry in it:
                    if entry.is_dir(follow_symlinks=False):
                        subdirs.append(entry)
                        d = first_date_in_name(entry.name)
                        if d:
                            dated.append((d, Path(entry.path)))

            if dated:
                dated.sort(key=lambda x: x[0], reverse=True)
                chosen_date, chosen_path = dated[0][0], dated[0][1]
                return chosen_path, chosen_date, "by_name_date"

            # Fallback: choose most recently modified directory
            if subdirs:
                latest = max(subdirs, key=lambda e: e.stat().st_mtime)
                return Path(latest.path), None, "by_mtime"

            return None, None, "none"

        def get_most_recent_file_by_first_date(folder: Path) -> Tuple[Optional[Path], Optional[datetime.date]]:
            """
            Inside 'folder', pick the file whose FIRST 8-digit date (YYYYMMDD) in its name is the largest.
            If no files contain a valid first 8-digit date, returns (None, None).
            """
            if not folder.exists() or not folder.is_dir():
                raise NotADirectoryError(f"Folder not found or not a directory: {folder}")

            best_date = None
            best_file = None

            with os.scandir(folder) as it:
                for entry in it:
                    if entry.is_file(follow_symlinks=False):
                        d = first_date_in_name(entry.name)  # STRICT: first 8 digits only
                        if d is not None:
                            if best_date is None or d > best_date:
                                best_date = d
                                best_file = Path(entry.path)

            return best_file, best_date

        base = Path(self.score_pivot_esg_path)

        # 1) Pick subfolder
        folder, folder_date, mode = get_most_recent_dated_subfolder(base)
        if mode == "none" or folder is None:
            print(f"[!] No subfolders found under: {base}")
            return

        # if mode == "by_name_date":
            # print(f"[OK] Selected folder by date in name: {folder.name} (date={folder_date})")
        # else:
        #     mtime = datetime.fromtimestamp(folder.stat().st_mtime)
        #     print(f"[OK] Selected folder by last modified time: {folder.name} (mtime={mtime})")

        # 2) Pick file inside folder using first 8 digits as date
        file_path, file_date = get_most_recent_file_by_first_date(folder)
        if not file_path:
            print(f"[!] No files with a valid first 8-digit date found in folder: {folder}")
            return

        # print(f"[OK] Selected file by first-8-digit date: {file_path.name} (date={file_date})")
        print(f"[RESULT] Full path for ESG Score Pivot File: {file_path}")
        return file_path

    def get_esg_pivot_score(self, bench_name_in_excel="INDEX MSCI WORLD_vs_1330696"):
        """
        Trouver le score pivot avec le mot clé choisi ()
        """
        path = self.find_esg_pivot_file_path()
        ficher_ESG = pd.read_csv(path,
                                encoding="cp1252",
                                sep="|",   # or "\t", "|", etc.
                                engine="python"
                                )
        note_pivot = ficher_ESG[ficher_ESG['sec_id'] == bench_name_in_excel]['note_pivot'].values[0]
        note_pivot = float(note_pivot)
        return note_pivot


    def adjust_bench_weight_with_recommandation(self, df, reco_secto, date):
        """
        1 - Calculer les weights sectoriels du bench 
        2 - Appliquer les reco sectorielles si besoin
        """
        if self.weight_neutral == "ICB 19":
            weight_secto_bench = \
                    df.groupby(' Benchmark ICB Supersector ')['Weight in ' + self.bench].sum() / df['Weight in ' + self.bench].sum()
            icb_missing = set([1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19]) - set(df[' Benchmark ICB Supersector '].unique())
            if len(icb_missing) > 0:
                print(f"Warning: follwing sectors are missing in benchmark: {list(icb_missing)}")
                try:
                    indices_to_delete = [int(icb)-1 for icb in icb_missing] # find out where need to be deleted and then mark it as 1
                    reco_secto = np.delete(np.array(reco_secto), indices_to_delete) # delete function can delete several values at a same time, no need to do iteration
                except:
                    print(date)

            # Recommandation sectorielle
            weight_secto_bench = weight_secto_bench + (np.array(reco_secto)*1)   

            # Adjust small weight sectors
            small_weight_mask = weight_secto_bench < 0.0025
            sectors_to_be_adjusted = weight_secto_bench[small_weight_mask].index
            if not sectors_to_be_adjusted.empty:
                # print(f"Warning: follwing sectors have weight lower than 0.0025, their weight will be replace as 0.0025: {sectors_to_be_adjusted.tolist()}")
                weight_secto_bench[small_weight_mask] = 0.0025

        elif self.weight_neutral == "ICB 11":
            weight_secto_bench = \
                    df.groupby(' Benchmark ICB Industry ')['Weight in ' + self.bench].sum() / df['Weight in ' + self.bench].sum()
        
        return weight_secto_bench

    def neutralise_score_by_secteur(self, df, list_score_col):
        """
        Neutraliser sectoriellement le score pour piocher les tops par secteur par la suite
        """
        df = df.copy()
        scores = df[list_score_col].astype(float).rank(pct=True)
        scores = (scores - scores.min()) / (scores.max() - scores.min())

        sector_column = {
            "ICB 11": ' Benchmark ICB Industry ',
            "ICB 19": ' Benchmark ICB Supersector ',
        }.get(self.score_neutral)
        if sector_column is not None:
            sector_keys = df[sector_column]
            valid_sectors = sector_keys.notna()
            sector_scores = scores.loc[valid_sectors].groupby(
                sector_keys.loc[valid_sectors],
            ).rank(pct=True)
            sector_min = sector_scores.groupby(
                sector_keys.loc[valid_sectors],
            ).transform('min')
            sector_max = sector_scores.groupby(
                sector_keys.loc[valid_sectors],
            ).transform('max')
            scores.loc[valid_sectors] = (
                sector_scores - sector_min
            ) / (sector_max - sector_min)

        df.loc[:, list_score_col] = scores
        return df


    def get_portfolio_name(self, style):
        """
        Automatically select portfolio name based on investment style, benchmark, and ranking position
        
        Parameters:
        style (str): Investment style, choose from list_style
        bench (str): Benchmark, supports "SP500" and "STOXX EUROPE 600"
        top (bool): True for Q1 (top 25%), False for Q5 (bottom 25%)
        
        Returns:
        str: Corresponding portfolio name
        """
        
        if self.mode_monthly_prod:
            if self.ptf_name == "PTF TEST":
                # Define investment style list
                list_style = ['Size Avg Percentile', 'Value Avg Percentile','Quality Avg Percentile',
                            'Mom Avg Percentile','LowVol Avg Percentile','Growth Avg Percentile', 
                            'Multi Avg Percentile']
                
                # Define benchmark to region mapping
                bench_to_region = {
                    'SP500': 'US',
                    'MSCI US': 'US',
                    'STOXX EUROPE 600': 'EU'
                }
                
                # Define style to portfolio type mapping
                style_to_type = {
                    'Size Avg Percentile': 'SIZE',
                    'Value Avg Percentile': 'VALUE',  
                    'Quality Avg Percentile': 'QUALITY',
                    'Mom Avg Percentile': 'MOM',
                    'LowVol Avg Percentile': 'LOWVOL',
                    'Growth Avg Percentile': 'GROWTH',
                    'Multi Avg Percentile': 'MF'
                }
                
                # Validate input parameters
                if style not in list_style:
                    raise ValueError(f"Style '{style}' not in supported style list: {list_style}")
                
                if self.bench not in bench_to_region:
                    raise ValueError(f"Benchmark '{self.bench}' not supported. Supported benchmarks: {list(bench_to_region.keys())}")
                
                # Get region and portfolio type
                region = bench_to_region[self.bench]
                portfolio_type = style_to_type[style]
                
                # Select Q1 or Q5 based on top parameter
                quintile = 'Q1' if self.Top else 'Q5'
                
                # Construct portfolio name
                ptf_name = f"FS_{region}_{portfolio_type}_{quintile}"

                if ptf_name == 'FS_EU_MF_Q1' and self.esg_exclusion>0:
                    ptf_name = "FS_EU_MF_ESG_Q1"
                if ptf_name == 'FS_EU_MF_Q5' and self.esg_exclusion>0:
                    ptf_name = "FS_EU_MF_ESG_Q5"
            
            if self.ptf_name != "PTF TEST":
                ptf_name = self.ptf_name

        if self.mode_monthly_prod != True:
            ptf_name = self.ptf_name

        return ptf_name

    def save_portfolio_data_incremental(self, df_concat, output_dir, date_obj=None):
        """
        Save portfolio data to Excel file incrementally.
        Creates file if it doesn't exist, otherwise appends new data.
        
        Parameters:
        df_concat (DataFrame): New data to save
        output_dir (str): Output directory path
        date_obj (datetime.date): Date object for folder naming (default: current date)
        """
        
        if date_obj is None:
            date_obj = pd.to_datetime(df_concat['Date']).iloc[0]
        
        # Create output file path
        # folder_name = date_obj.strftime("%B %Y")
        folder_name = date_obj.strftime("%m %Y")
        folder_path = os.path.join(output_dir, f"Pour {folder_name}")
        output_file = os.path.join(folder_path, "PTFS TO PUSH.xlsx")
        # Create directory if it doesn't exist
        os.makedirs(folder_path, exist_ok=True)
        
        # Prepare new data
        new_data = df_concat[['PTF', 'ISIN', 'Weight', 'Date']].copy()
        
        # Check if file exists
        if os.path.exists(output_file):
            try:
                # Read existing data
                existing_data = pd.read_excel(output_file)
                print(f"Found existing file with {len(existing_data)} records")
                # Supprimer les lignes de existing_data dont 'PTF' est présent dans new_data
                existing_data = existing_data[~existing_data['PTF'].isin(new_data['PTF'])]


                # Combine existing and new data
                combined_data = pd.concat([existing_data, new_data], ignore_index=True, axis=0)
                
                # Remove duplicates based on all columns (optional)
                # You might want to modify this logic based on your needs
                combined_data = combined_data.drop_duplicates(subset=['PTF', 'ISIN', 'Date'], keep='last')
                
                print(f"After combining and deduplicating: {len(combined_data)} records")
                
            except Exception as e:
                print(f"Error reading existing file: {e}")
                print("Creating new file with current data only")
                combined_data = new_data
        else:
            print("File doesn't exist, creating new file")
            combined_data = new_data
        
        # Write combined data to Excel
        try:
            with pd.ExcelWriter(output_file, datetime_format='dd/mm/yyyy') as writer:
                combined_data.to_excel(writer, index=False)
            
            print(f"Successfully saved {len(combined_data)} records to: {output_file}")
            
        except Exception as e:
            print(f"Error writing to file: {e}")
            raise

    def cap_weight_by_sector(self, ptf, n_iteration=30):
        """
        Cap individual stock weights while redistributing excess weight proportionally
        to other companies within the same sector
    
        Parameters:
        ptf: DataFrame with columns ['Date', 'Secto', 'Weight']
        threshold: Maximum weight limit for individual stocks
        n_iteration: Maximum number of iterations
    
        Returns:
        DataFrame with adjusted weights
        """
        threshold = self.cap_weight_threshold

        result = ptf.copy()
    
        for iteration in range(n_iteration):
            
            # Track if any adjustment was made for early termination
            has_adjustment = False
        
            # Process each date separately
            for date in result['Date'].unique():
                date_mask = result['Date'] == date
                date_data = result[date_mask].copy()
                
                # Process each sector separately
                for sector in date_data['Secto'].unique():
                    sector_mask = date_data['Secto'] == sector
                    sector_data = date_data[sector_mask].copy()
                
                    # Find overweight stocks
                    overweight_mask = sector_data['Weight'] > threshold
                
                    if overweight_mask.any():
                        has_adjustment = True
                    
                        # Calculate total excess weight
                        excess_weight = (sector_data.loc[overweight_mask, 'Weight'] - threshold).sum()
                    
                        # Cap overweight stocks to threshold
                        sector_data.loc[overweight_mask, 'Weight'] = threshold
                    
                        # Find underweight stocks (those not exceeding threshold)
                        underweight_mask = ~overweight_mask
                        underweight_data = sector_data[underweight_mask]
                    
                        if len(underweight_data) > 0:
                            # Calculate total weight of underweight stocks
                            underweight_total = underweight_data['Weight'].sum()
                        
                            if underweight_total > 0:
                                # Distribute excess weight proportionally to underweight stocks
                                allocation_ratio = excess_weight / underweight_total
                                sector_data.loc[underweight_mask, 'Weight'] = (
                                    sector_data.loc[underweight_mask, 'Weight'] * (1 + allocation_ratio)
                                )
                    
                        # Update results
                        result.loc[date_mask & (result['Secto'] == sector), 'Weight'] = sector_data['Weight'].values
        
            # Early termination if no adjustments were made
            if not has_adjustment:
                break
        
        return result

    def select_titles(self, group, max_weight_threshold, column):
        """
            Sélectionne un nombre minimum de titres dans un secteur afin de respecter
            une contrainte de poids maximum par titre.
        
            Cette fonction calcule le poids total du secteur, détermine le nombre
            minimal de titres nécessaires pour que chacun ne dépasse pas le seuil
            de poids maximum défini, puis sélectionne les titres ayant les valeurs
            les plus élevées en mérique choisie (ex. Score ML).
        
            Paramètres
            ----------
            group : pandas.DataFrame
                Sous-ensemble du DataFrame contenant uniquement les titres du secteur
                considéré. Doit contenir au minimum la colonne :
                - "Weight in <benchmark>" : poids de chaque titre dans le benchmark.
        
            max_weight_threshold : float
                Poids maximum toléré pour un titre dans le portefeuille. Le nombre
                minimal de titres est calculé de manière à ce qu'aucun ne dépasse ce seuil.
        
            column : str
                Nom de la colonne utilisée pour sélectionner les titres avec les valeurs
                les plus élevées (utilisée avec `nlargest`).
        
            Retour
            ------
            pandas.DataFrame
                Un DataFrame contenant uniquement les titres sélectionnés selon la
                contrainte de poids et la logique de ranking sur la colonne donnée.
        
            Notes
            -----
            - Le nombre minimal de titres est arrondi à l’entier supérieur.
            - Si le secteur a un poids total de 0, aucun titre ne sera sélectionné.
        
        """
        sector_weight = group['Weight in ' + self.bench].sum()  # Get the sector's total weight
        min_titles_needed = (sector_weight // max_weight_threshold) + (1 if sector_weight % max_weight_threshold != 0 else 0) # Division euclidienne pour connaitre le min de titre a avoir
        
        sector = group[" Benchmark ICB Supersector "].unique()

        # Choisir les minimum de titres pour respecter la contrainte
        selected_titles = group.nlargest(int(min_titles_needed), column)  
        
        return selected_titles

    def _monthly_base_cache_key(self, date):
        """Construit une clé sûre pour les préparations mensuelles réutilisables."""
        if self.monthly_base_cache is None:
            return None
        if not isinstance(self.reco_secto, list):
            return None
        if self.metrics == 'Multi Avg Percentile':
            return None
        return (
            pd.Timestamp(date), self.bench, self.ponderation,
            float(self.cut_mkt_cap), self.weight_neutral,
            tuple(self.reco_secto), float(self.percentile),
        )

    def _preparation_from_monthly_base(self, monthly_base, screen,
                                       list_score_col):
        """Ajoute uniquement les scores courants à une base mensuelle compacte."""
        score_source = screen
        if score_source.index.name != 'ISIN' and 'ISIN' in score_source.columns:
            score_source = score_source.set_index('ISIN')
        if score_source.index.duplicated().any():
            score_source = score_source.loc[
                ~score_source.index.duplicated(keep='first')
            ]
        score_values = score_source.loc[:, list_score_col].reindex(
            monthly_base['df'].index,
        )
        df = monthly_base['df'].copy(deep=True)
        df.loc[:, list_score_col] = score_values.to_numpy()
        df.loc[
            ~monthly_base['eligible_market_cap'], list_score_col,
        ] = np.nan
        return {
            'df': df,
            'date': monthly_base['date'],
            'list_score_col': list_score_col,
            'nb_securities': monthly_base['nb_securities'],
            'weight_secto_bench': monthly_base['weight_secto_bench'],
        }

    def sec_list_spot(self,screen_agg_monthly=None):
        """
        Generate Best Scored Sec List for 1 Month, According to the Metrics Chosen
        """

        if isinstance(screen_agg_monthly, pd.DataFrame):
            source_screen = screen_agg_monthly
        elif screen_agg_monthly==None: # If single month dataframe is not defined, then use the last month data to generate ptf
            source_screen = self.screen[
                self.screen['Date'] == self.screen['Date'].max()
            ]

        if type(self.metrics)==str:
            list_score_col = [self.metrics]
        else:
            list_score_col = self.metrics

        ################ use only the last month's screen (production mode) ################ 
        raw_date = pd.to_datetime(source_screen['Date'].max())
        cache_key = self._monthly_base_cache_key(raw_date)
        if cache_key is not None and cache_key in self.monthly_base_cache:
            preparation = self._preparation_from_monthly_base(
                self.monthly_base_cache[cache_key],
                source_screen,
                list_score_col,
            )
            self._last_monthly_preparation = preparation
            return self._finalize_sec_list_spot(preparation)

        screen=copy.deepcopy(source_screen)
        date = raw_date
        # screen=screen[screen['Date']==date]

        if screen.index.duplicated().any():
            screen = screen[~screen.index.duplicated(keep='first')]

        ################ Merging les tickers secondaires ################
        screen = merge_ticker_secondaire(screen, self.bench)

        ################ Filtrage Bench ################
        df = screen[screen['Weight in ' + self.bench]>0] # on conserve que les weights positifs

        ################ Fixer le nombre de boite à choisir pour plus tard avant que le df soit modifié ################
        if self.percentile > 1:   ##### If percentile is bigger than 1, then this variable means exact number of securities to pick
            nb_securities=self.percentile
        else: ##### If percentile is less than 1, then this variable means the percentage of securities in investable univers to pick
            nb_securities = round(len(df) * self.percentile)

        ################ donne le 1er jour du mois suivant ################
        date +=pd.offsets.MonthBegin(1)

        ################################################################################################################################
        if self.metrics == "Multi Avg Percentile":  ### Reco facto will only activated when using "Multi Avg Percentile" as metric
            # Recommandtions Sectorielles à la date donnée
            if isinstance(self.reco_facto, list):
                reco_facto =np.array(self.reco_facto)
            elif isinstance(self.reco_facto, pd.DataFrame):
                try:
                    reco_facto = self.reco_facto.loc[date] 
                except : 
                    print(f"{date} not in reco_facto")
                    raise KeyError
                
            if (reco_facto.sum() == 0) : 
                reco_facto = np.array([0.2]*5)
            else:
                reco_facto = np.array(reco_facto/reco_facto.sum())

            list_style = ['Growth Avg Percentile','LowVol Avg Percentile','Mom Avg Percentile','Quality Avg Percentile','Value Avg Percentile']
            df.loc[:, 'Multi Avg Percentile'] = df[list_style].dot(reco_facto)

        ################################################################################################################################
        # regression de Benchmark value sur weight in bench pour compléter les valeurs manquantes
        fit = np.polyfit(df.loc[pd.isna(df['Benchmark Market Value Millions in EUR ']) == False, 'Weight in ' + self.bench],df.loc[pd.isna(df['Benchmark Market Value Millions in EUR ']) == False, 'Benchmark Market Value Millions in EUR '], deg = 1)
        func = np.poly1d(fit)
        df.loc[pd.isna(df['Benchmark Market Value Millions in EUR ']),'Benchmark Market Value Millions in EUR '] = func(df.loc[pd.isna(df['Benchmark Market Value Millions in EUR ']),'Weight in ' + self.bench])
        # Market cut filtrage
        eligible_market_cap = (
            df['Benchmark Market Value Millions in EUR '] > self.cut_mkt_cap
        ).to_numpy()
        df.loc[~eligible_market_cap, list_score_col] = np.nan
        
        df = df.copy()
        df.loc[:, 'Date'] = date
    
        # Ajuster le poids des titres dans l'indice (smoothing)
        df=self.adjust_companies_ponderation(df)

        # Recommandtions Sectorielles à la date donnée
        if isinstance(self.reco_secto, list):
            reco_secto =copy.deepcopy(self.reco_secto)
        elif isinstance(self.reco_secto, pd.DataFrame):
            try:
                reco_secto = self.reco_secto.loc[date].to_list() 
            except : 
                print(f"{date} not in reco_secto")
                raise KeyError
        
        # Appilquer les reco sectorielle aux secteurs
        weight_secto_bench = self.adjust_bench_weight_with_recommandation(df, reco_secto, date)

        preparation = {
            'df': df,
            'date': date,
            'list_score_col': list_score_col,
            'nb_securities': nb_securities,
            'weight_secto_bench': weight_secto_bench,
        }
        if cache_key is not None:
            compact_base = df.drop(
                columns=[*list_score_col, 'Company SEDOL'],
                errors='ignore',
            ).copy(deep=True)
            if isinstance(compact_base.index, pd.CategoricalIndex):
                compact_base.index = pd.Index(
                    compact_base.index.astype(object),
                    name=compact_base.index.name,
                )
            self.monthly_base_cache[cache_key] = {
                'df': compact_base,
                'date': date,
                'nb_securities': nb_securities,
                'weight_secto_bench': weight_secto_bench.copy(),
                'eligible_market_cap': eligible_market_cap.copy(),
            }
        self._last_monthly_preparation = preparation
        return self._finalize_sec_list_spot(preparation)

    def _finalize_sec_list_spot(self, preparation):
        """Construit une sélection Top ou Worst depuis une préparation commune."""
        df = preparation['df'].copy(deep=True)
        date = preparation['date']
        list_score_col = preparation['list_score_col']
        nb_securities = preparation['nb_securities']
        weight_secto_bench = preparation['weight_secto_bench']

        # Filtrage ESG only if we choose Top ptf
        if self.Top:
            # if self.score_pivot_esg is string then it will find corresponding item note in excel, 
            # if self.score_pivot_esg is float, it will use it as threshold, 
            # if self.score_pivot_esg is None, filter by esg pivot score will not apply
            if isinstance(self.score_pivot_esg, str):
                print(f"Récupération de Score Pivot ESG avec index {self.score_pivot_esg}......")
                self.score_pivot_esg = self.get_esg_pivot_score(bench_name_in_excel = self.score_pivot_esg) 
            if isinstance(self.score_pivot_esg, float):
                print(f"Score Pivot ESG est {self.score_pivot_esg}")
            df = self.filtrage_esg_liste_noire(df,date)

        df = self.neutralise_score_by_secteur(df, list_score_col) 

        df["Raison Repechage"] = ""

        columns = ['PTF', 'ISIN', 'Weight', 'Date', "Raison Repechage"]
        result_sec_list = pd.DataFrame()

        for i in range(len(list_score_col)):

            if self.Top == True:
                df_top = df.nlargest(nb_securities,list_score_col[i])
                df_top['Raison Repechage'] = list_score_col[i]  # Mettre la métrique de repechage comme raison, ex. Score ML/ Growth Avg Percentile

                # Selection du minimum de titres minimum par secteur pour respecter la contrainte de poid max (self.cap_weight_threshold)
                if self.cap_weight_threshold != None:
                    df_top_sector =  df.groupby(' Benchmark ICB Supersector ').apply(self.select_titles, max_weight_threshold=self.cap_weight_threshold, column = list_score_col[i])
                    df_top_sector = df_top_sector.drop( columns = [" Benchmark ICB Supersector "] )
                    df_top_sector = df_top_sector.reset_index(drop=False)
                    df_top_sector.index = df_top_sector["ISIN"]
                    df_top_sector = df_top_sector.drop( columns = ["ISIN"] )
                    df_top_sector["Raison Repechage"] = "Sector"
                
                    # Concat les deux top list
                    df_top_combined = pd.concat([df_top, df_top_sector], axis=0)
                    df_top = df_top_combined[~df_top_combined.index.duplicated(keep='first')]  # Prioritize top selected with classical way

            if self.Top == False:
                df_top=df.nsmallest(nb_securities,list_score_col[i])
                df_top['Raison Repechage'] = "Worst Metric"

            if isinstance(self.top_mandatory, int) or isinstance(self.top_mandatory, float):
                nb_top_mandatory = int(self.top_mandatory)
                # print(f"Top Mandatory is activated, top {nb_top_mandatory} companies in bench will be added to sec list")
                liste_top_mandatory = df.nlargest(nb_top_mandatory, 'Weight in ' + self.bench)
                liste_top_mandatory['Raison Repechage'] = "Top Obligatoire par Région"

                df_top_combined = pd.concat([liste_top_mandatory, df_top], axis=0)
                df_top = df_top_combined[~df_top_combined.index.duplicated(keep='first')]  # Prioritize top selected with top mandatory


            ##### Ajustement Secteur Neutre
            temp_df = pd.DataFrame(columns = columns)
            temp_df['ISIN'] = np.asarray(df_top.index, dtype=object)
            if self.weight_neutral == "ICB 19":
                temp_df['Secto'] = df_top[
                    ' Benchmark ICB Supersector '
                ].astype(float).to_numpy()
            elif self.weight_neutral == "ICB 11":
                temp_df['Secto'] = df_top[
                    ' Benchmark ICB Industry '
                ].astype(float).to_numpy()
            temp_df['Weight'] = df_top['Benchmark Market Value Millions in EUR '].values

            temp_df['Score'] = df_top[list_score_col[i]].values
            temp_df['Date'] = df_top['Date'].values
            temp_df['Raison Repechage'] = df_top['Raison Repechage'].values


            ###################### Security check : secto in temp_df is a subset of secto in weight_secto_bench ############################
            temp_df_sectors = set(temp_df['Secto'].unique())
            benchmark_sectors = set(weight_secto_bench.index)
            if not temp_df_sectors.issubset(benchmark_sectors):
                missing_sectors = temp_df_sectors.difference(benchmark_sectors)
                raise ValueError(f"Error: Sectors in temp_df {missing_sectors} are not defined in weight_secto_bench")
            #################################################################################################################################

            if self.weight_neutral != None:
                secto_weight_sum = temp_df.groupby('Secto')['Weight'].transform('sum')
                secto_benchmark_weight = temp_df['Secto'].map(weight_secto_bench)
                scaling_factor = secto_benchmark_weight / secto_weight_sum
                temp_df['Weight'] = temp_df['Weight'] * scaling_factor
                temp_df['Weight'] = temp_df['Weight'] / temp_df['Weight'].sum()

            ############ cap weight by sector if necessary ################
            if self.cap_weight_threshold != None:
                # print(f"Capping generated portfolio, no more than {self.cap_weight_threshold} for each title")
                temp_df = self.cap_weight_by_sector(temp_df)

            ##### Give name to ptf generated
            temp_df['PTF'] = self.get_portfolio_name(list_score_col[i])

            result_sec_list = pd.concat([result_sec_list,temp_df], ignore_index=True)
        
        if self.output_dir != None:
            if self.mode_monthly_prod:
                self.save_portfolio_data_incremental(result_sec_list, self.output_dir)
            else:
                result_save_path = os.path.join(self.output_dir, "sec_list_result.xlsx")
                result_sec_list.to_excel(result_save_path)
                print(f"Sec list is generated at path : {result_save_path}")

        self.sec_list_monthly = result_sec_list.copy(deep=True)
        # print(f"Monthly sec list is generated for {date}, you can check 'self.sec_list_monthly' attribute for more details.")

        return result_sec_list

    def drift_weight(self, df_rebal, col_id, df_returns, col_date, col_weight, date_fin_drifter):
        """
        Drifter les weights avec les returns daily.
        """
        df_rebal = df_rebal[df_rebal[col_id].isin(df_returns.columns)]

        liste_rebal_date = list(df_rebal[col_date].unique())

        liste_date_returns = list(df_returns[
                                            (df_returns.index >= liste_rebal_date[0]) & 
                                            (df_returns.index <= date_fin_drifter)
                                            ].index)
        df_rebal.reset_index(inplace=True, drop=True)

        # Boucle dans le cas d'une date de rebalancement non présente dans df_returns -> changement de la date de rebalancement avec la date future la plus proche
        for i in range(len(liste_rebal_date)) :
            # Matcher la liste des dates avec les dates existantes dans la base returns
            if  liste_rebal_date[i] not in liste_date_returns :
                try:
                    # Try pour chercher la date future la plus proche
                    new_date_rebal = min(d for d in liste_date_returns if d > liste_rebal_date[i])
                except ValueError:
                    # Si pas de future date trouvée, on prend la date antérieure la plus proche (cas frequency=1 dernière date est une date de rebalancement)
                    new_date_rebal = max(d for d in liste_date_returns if d < liste_rebal_date[i])

                df_rebal = df_rebal.replace(liste_rebal_date[i], new_date_rebal)   # Change la date en question dans ptf
                liste_rebal_date[i] = new_date_rebal                               # Change la date en question dans liste rebal ['2025-10-01', '2025-11-01', '2025-12-01']

        # Tri avec la fonction sorted()
        liste_date_all = list(set(liste_rebal_date).union(set(liste_date_returns)))
        nouvelle_liste_dates = sorted(liste_date_all)


        new_df=pd.DataFrame(data=nouvelle_liste_dates, columns=['Date_returns'])
        new_df['Date_screen'] = new_df['Date_returns'].apply(lambda x: df_rebal.loc[df_rebal[col_date]<=x, col_date].max())
        # Date_returns  Date_screen
        # 2025-12-29    2025-12-01
        # 2025-12-30    2025-12-01
        # 2025-12-31    2025-12-01
        # 2026-01-01    2025-12-01


        df_rebal[col_date] = pd.to_datetime(df_rebal[col_date])

        df_merge = df_rebal.merge(new_df, how='left', left_on=col_date, right_on = 'Date_screen')
        # PTF        ISIN          Weight     Date        Raison Repechage               Secto  Score     SEDOL     Date_returns  Date_screen
        # ML EU Q1   NL0010273215  0.050000   2025-12-01  Top Obligatoire par Région     16.0   0.833333  JWGWTR-R  2025-12-01    2025-12-01
        # ML EU Q1   NL0010273215  0.050000   2025-12-01  Top Obligatoire par Région     16.0   0.833333  JWGWTR-R  2025-12-02    2025-12-01
        # ML EU Q1   NL0010273215  0.050000   2025-12-01  Top Obligatoire par Région     16.0   0.833333  JWGWTR-R  2025-12-03    2025-12-01
        # ML EU Q1   NL0010273215  0.050000   2025-12-01  Top Obligatoire par Région     16.0   0.833333  JWGWTR-R  2025-12-04    2025-12-01
        # ML EU Q1   NL0010273215  0.050000   2025-12-01  Top Obligatoire par Région     16.0   0.833333  JWGWTR-R  2025-12-05    2025-12-01


        df_merge.drop(columns=col_date,inplace=True)
        df_merge.rename(columns={'Date_returns':col_date},inplace=True)
        df_merge.sort_values(by=col_date,inplace=True)

        # Prendre les returns de la bonne période
        df_returns = df_returns[new_df['Date_screen'].min(): date_fin_drifter]
        df_returns.iloc[0, :] = 0
        returns_cum = (1+df_returns).cumprod()
        #             X5B68T-R  CHKTQ0-R  KQS7WY-R  T8JSPN-R  MF02C5-R
        # 2025-12-01   1.000000  1.000000  1.000000  1.000000  1.000000
        # 2025-12-02   1.013761  1.022564  1.000123  1.020184  1.000123
        # 2025-12-03   0.997901  0.989270  0.995270  1.009678  0.995270
        # 2025-12-04   0.995101  0.994596  1.005961  1.011745  1.005961
        # 2025-12-05   0.996441  0.995082  1.007871  1.012666  1.007871


        # Intuitivement : pour chaque ligne (une date donnée), 
        # on divise l’ensemble de la ligne par le vecteur des rendements cumulés du jour « Date_screen » précédent (dernière date de rebal) ; 
        # cela revient à réinitialiser la valeur de base à 1 à cette date de référence et à obtenir ainsi la performance relative depuis la dernière date de sélection.
        returns_drift = returns_cum.apply(lambda x:x/returns_cum.loc[(new_df.loc[new_df['Date_screen']<=x.name,'Date_screen'].max())], axis=1)
        #             X5B68T-R  CHKTQ0-R  KQS7WY-R  T8JSPN-R  MF02C5-R
        # 2025-12-01   1.000000  1.000000  1.000000  1.000000  1.000000
        # 2025-12-02   1.013761  1.022564  1.000123  1.020184  1.000123
        # 2025-12-03   0.997901  0.989270  0.995270  1.009678  0.995270
        # 2025-12-04   0.995101  0.994596  1.005961  1.011745  1.005961
        # 2025-12-05   0.996441  0.995082  1.007871  1.012666  1.007871

        returns_drift_flat = returns_drift.stack().to_frame().reset_index(names=[col_date,col_id])
        returns_drift_flat.columns=[col_date,col_id,'drift_multiplicator']
        # Date        SEDOL     drift_multiplicator
        # 2025-12-01  JWGWTR-R  1.000000
        # 2025-12-02  JWGWTR-R  1.013381
        # 2025-12-03  JWGWTR-R  1.039603
        # 2025-12-04  JWGWTR-R  1.033020
        # 2025-12-05  JWGWTR-R  1.026869


        returns_flat=df_returns.stack().to_frame().reset_index()
        returns_flat.columns=[col_date+'_shift',col_id, 'Return']
        # # Date_shift  SEDOL     Return
        # # 2025-12-01  JWGWTR-R  0.025791
        # # 2025-12-02  JWGWTR-R  0.013381
        # # 2025-12-03  JWGWTR-R  0.025876
        # # 2025-12-04  JWGWTR-R -0.006332
        # # 2025-12-05  JWGWTR-R -0.005954


        unique_date = df_merge[col_date].unique()
        df_date = pd.DataFrame(data = unique_date,columns=[col_date])
        df_date[col_date+'_shift'] = df_date[col_date].shift(-1) 
        # Date        Date_shift
        # 2025-12-01  2025-12-02
        # 2025-12-02  2025-12-03
        # 2025-12-03  2025-12-04
        # 2025-12-04  2025-12-05
        # 2025-12-05  2025-12-08   


        df_merge = df_merge.merge(df_date, how='left', on =col_date)   #Pour recupérer date shift
        # df_merge = df_merge[df_merge[col_date+'_shift'].notna()]
        # Columns: PTF | ISIN | Weight | SEDOL             | Date | Date_screen | Date_shift
        # 0   | ML EU Q1 | NL0010273215 | 0.05 | JWGWTR-R | 2025-12-01 | 2025-12-01 | 2025-12-02
        # 77  | ML EU Q1 | NL0010273215 | 0.05 | JWGWTR-R | 2025-12-02 | 2025-12-01 | 2025-12-03
        # 87  | ML EU Q1 | NL0010273215 | 0.05 | JWGWTR-R | 2025-12-03 | 2025-12-01 | 2025-12-04
        # 162 | ML EU Q1 | NL0010273215 | 0.05 | JWGWTR-R | 2025-12-04 | 2025-12-01 | 2025-12-05
        # 193 | ML EU Q1 | NL0010273215 | 0.05 | JWGWTR-R | 2025-12-05 | 2025-12-01 | 2025-12-08

        df_merge = df_merge.merge(returns_drift_flat, how='left', on = [col_date, col_id]) #Pour recupérer drift_multiplicator
        # df_merge = df_merge.merge(returns_flat, how='left', on = [col_date+'_shift', col_id])
        # PTF        ISIN           Weight  Date_screen  Date_shift  drift_multiplicator 
        # ML EU Q1   NL0010273215   0.05    2025-12-01   2025-12-02  1.000000            
        # ML EU Q1   NL0010273215   0.05    2025-12-01   2025-12-03  1.013381           
        # ML EU Q1   NL0010273215   0.05    2025-12-01   2025-12-04  1.039603           
        # ML EU Q1   NL0010273215   0.05    2025-12-01   2025-12-05  1.033020          
        # ML EU Q1   NL0010273215   0.05    2025-12-01   2025-12-08  1.026869         

        df_merge[col_weight] = df_merge[col_weight]*df_merge['drift_multiplicator']

        df_merge.drop(columns = ['Date_screen'],inplace=True)
        # df_merge.drop(columns = col_date,inplace=True)
        # df_merge.rename(columns={col_date+'_shift': col_date},inplace=True)

        res = df_merge.loc[df_merge[col_date] == max(liste_date_returns)]

        res = res.copy()
        res = res.drop(columns=['drift_multiplicator', 'Date_shift'], errors='ignore')

        # res['Weight'] = res['Weight'].transform(lambda w: w / w.sum())
        sum_weight = res['Weight'].sum()
        res['Weight'] = res['Weight'] / sum_weight if sum_weight != 0 else 0.0

        return res    

    def update_ptf_with_monthly_drift(self, df):
        """
        For every month present in the portfolio (including the first and last),
        check whether the following month already exists.   
        If it does **not** exist, drift the weight of the current month until the
        date forward by one month, and append it to the dataframe.
        """
        ptf = df.copy()

        # Add SEDOL for drift using returns df ######## raise error if not completly mapping
        sedol_to_isin = dict(zip(self.screen.index, self.screen['Company SEDOL'])) 
        ptf['SEDOL'] = ptf['ISIN'].map(sedol_to_isin)


        existing_dates = ptf.sort_values('Date')['Date'].unique()
        print("Longueur sec_list avant : ", len(existing_dates))

        today = datetime.datetime.now()
        
        for date in existing_dates:  # premier du mois
            next_month = date + pd.DateOffset(months=1)
            date_fin_drifter = date + pd.DateOffset(months=1)
            # print(f"for {date}, date_fin_drifter is {date_fin_drifter}")

            # Si on entre dans le if suivant, c'est à dire, on commence à drifter
            if next_month not in existing_dates:
                if next_month <= today:  # Condition pour sortir : si next month est superieur 
                    # Get the current month's portfolio data
                    current_month_data = ptf[ptf["Date"] == date].copy()
                    
                    # Prepare parameters for drift function
                    col_id = "SEDOL"
                    col_weight = "Weight"
                    col_date = "Date"
                    
                    # Use drift logic to adjust weights
                    next_month_df = self.drift_weight(
                        current_month_data,
                        col_id,
                        self.returns.copy(),
                        col_date,
                        col_weight,
                        date_fin_drifter
                    )
                    # print(next_month_df['Date'].unique())

                    next_month_df['Date'] = date_fin_drifter
                    # print(next_month_df['Date'].unique())
                    # Append the drifted month to the main dataframe
                    ptf = pd.concat([ptf, next_month_df], axis=0).sort_values("Date")
                else:
                    print("Longueur sec_list après : ", len(existing_dates))
                    return ptf
            else:
                return ptf
        return ptf

    def update_ptf_with_monthly_additions(self, df):
        """
        For every month present in the portfolio (including the first and last),
        check whether the following month already exists.  
        If it does **not** exist, create a copy of the current month, shift the
        date forward by one month, and append it to the dataframe.
        """
        # keep a set for O(1) membership checks
        ptf = df.copy()

        existing_dates = set(ptf["Date"].unique())
        print("Longueur sec_list avant : " , len(existing_dates))

        sorted_existing_dates = sorted(existing_dates)  # Exemple : 01-10 , 03-10 , 05-10 ect
        today = datetime.datetime.now()
        for date in sorted_existing_dates:  
            
            next_month = date + pd.DateOffset(months=1)
            
            # Continue adding months until there are no more gaps
            while next_month not in existing_dates :
                if  next_month > today : # Condition pour sortir : si next month est superieur 
                    break
                else :    
                    prev_ptf = ptf[ptf["Date"] == date].copy()
                    prev_ptf["Date"] = next_month
                    ptf = pd.concat([ptf, prev_ptf]).sort_values("Date").reset_index(drop=True)
                    
                    # Update the set so that we don't add the same month twice
                    existing_dates.add(next_month)
                    
                    # Move to the newly added month for further checks
                    date = next_month
                    next_month += pd.DateOffset(months=1)
                
            if next_month > today  :
                break
        
        print("Longueur sec_list après : ", len(existing_dates))
        return ptf
    

    def find_next_closest_date(self, start_date, offset):
        """
        Finds the next closest date to start_date from the given DataFrame.

        Parameters:
        - start_date (datetime): The reference date.
        - screen_agg (pd.DataFrame): A DataFrame containing a 'Date' column with datetime objects.

        Returns:
        - datetime: The next closest date that satisfies the conditions.
        
        Raises:
        - ValueError: If start_date is not a datetime object or if no valid dates are found.
        """
        screen_agg = self.screen
        
        # Calculer les différences absolues entre chaque date et la start_date
        screen_agg = screen_agg[screen_agg["Date"]>=start_date]
        dates = screen_agg["Date"].unique()
        dates = pd.to_datetime(dates)

        closest_date = min(dates, key=lambda d: abs(d - start_date)) # Prend la date du screen_agg après start_date la plus proche


        # Si offset = 0, je rentre dans le if si closest_date est un mois pair cela permettra de prendre la date d'apres qui est forcement un mois impair
        # Si offset = 1, je rentre dans le if si closest_date est un mois impair cela permettra de prendre la date d'apres qui est forcement un mois pair
        if closest_date.month%2==offset:   
            dates = screen_agg[screen_agg["Date"]>closest_date]["Date"].unique()
            dates = pd.to_datetime(dates)

            closest_date = min(dates, key=lambda d: abs(d - start_date))

        return closest_date

    def generic_histo_seclist(self, start_date, freq_rebal=None, screen_start_date = "mois_impair", fill_method="drift"):
        """
        Apply a function to subsets of financial data based on specified frequency.
        
        Parameters:
        -----------
        func : function
            The function to apply to each subset of data
        start_date : str or datetime
            The earliest date to include in the analysis
        *args : 
            First argument is screen_agg (DataFrame or path to parquet file)
            Remaining arguments are passed to func
        freq : int, optional
            The frequency in months for selecting dates
        rebalancing_start_backward : datetime, optional
            If provided, the latest date will be the date in the month before this date
        
        Returns:
        --------
        DataFrame
            Combined results from the function applied to each subset
        """
        
        screen_agg = self.screen
        if type(screen_agg) == str:
            screen_agg = pd.read_parquet(screen_agg)
        

        #START DATE commence en  mois pair
        if screen_start_date == "mois_pair" :
            self.start_date  = self.find_next_closest_date(start_date,1)
        elif screen_start_date == "mois_impair" :
            self.start_date  = self.find_next_closest_date(start_date,0)
        else:
            self.start_date = start_date

        print( "Premiere date du screen_agg prise en compte : " , self.start_date)

        # Filter by start_date
        screen_agg = screen_agg.loc[
            screen_agg['Date'] >= self.start_date
        ].copy()
        all_dates = sorted(screen_agg['Date'].unique())
        
        if not all_dates:
            return pd.DataFrame()  # Return empty DataFrame if no dates
        
        # Determine which dates to keep based on frequency
        if freq_rebal == None:
            # Keep all dates for monthly frequency (original behavior)
            dates_to_keep = all_dates
        else:
            months_step = freq_rebal
            dates_to_keep = all_dates[::months_step]
        
        # Sort the dates to keep
        dates_to_keep = sorted(dates_to_keep)
    
        # Create subsets for each date
        # Apply function - handle possible parallelization issues
        func=self.sec_list_spot

        result_sec_list=[]
        from tqdm import tqdm
        for date_ in tqdm(dates_to_keep, desc="Generation Sec_list"):
            screen = screen_agg.loc[screen_agg['Date'] == date_]
            result_sec_list.append(func(screen_agg_monthly=screen))
            
        # Concatenate seclist results
        if result_sec_list and isinstance(result_sec_list[0], pd.DataFrame):
            df = pd.concat(result_sec_list, ignore_index=True)
        else:
            # Handle the case where func doesn't return DataFrames
            df = pd.DataFrame(result_sec_list)

        # for bimestriel situation
        self.sec_list_historical = df.copy()
        if fill_method=="drift":
            self.sec_list_historical  = self.update_ptf_with_monthly_drift(self.sec_list_historical)
        if fill_method=="copy":
            self.sec_list_historical  = self.update_ptf_with_monthly_additions(self.sec_list_historical)

        print(f"Historical sec list is generated, you can check 'self.sec_list_historical' attribute for more details.")
        
        return df

    def generic_histo_seclists_pair(
        self, builder_bottom, start_date, freq_rebal=None,
        screen_start_date="mois_impair", fill_method="drift",
    ):
        """Construit Top et Worst en partageant la préparation mensuelle."""
        screen_agg = self.screen
        if isinstance(screen_agg, str):
            screen_agg = pd.read_parquet(screen_agg)

        if screen_start_date == "mois_pair":
            resolved_start_date = self.find_next_closest_date(start_date, 1)
        elif screen_start_date == "mois_impair":
            resolved_start_date = self.find_next_closest_date(start_date, 0)
        else:
            resolved_start_date = start_date

        self.start_date = resolved_start_date
        builder_bottom.start_date = resolved_start_date
        print("Premiere date du screen_agg prise en compte : ", resolved_start_date)

        screen_agg = screen_agg.loc[
            screen_agg['Date'] >= resolved_start_date
        ].copy()
        all_dates = sorted(screen_agg['Date'].unique())
        if not all_dates:
            empty = pd.DataFrame()
            self.sec_list_historical = empty.copy()
            builder_bottom.sec_list_historical = empty.copy()
            return empty, empty

        dates_to_keep = (
            all_dates if freq_rebal is None else all_dates[::freq_rebal]
        )
        top_results = []
        bottom_results = []
        weight_column = 'Weight in ' + self.bench
        from tqdm import tqdm
        for date_ in tqdm(
            sorted(dates_to_keep), desc="Generation Sec_list Top/Worst",
        ):
            monthly_screen = screen_agg.loc[screen_agg['Date'] == date_]
            positive_weights = pd.to_numeric(
                monthly_screen[weight_column], errors='coerce',
            ).gt(0)
            if not positive_weights.any():
                target_date = pd.Timestamp(date_) + pd.offsets.MonthBegin(1)
                if not top_results:
                    warnings.warn(
                        f"Aucune pondération positive n'est disponible pour le "
                        f"benchmark {self.bench} au {pd.Timestamp(date_):%Y-%m-%d}. "
                        "Le rééquilibrage est ignoré, car aucune position "
                        "précédente ne peut être prolongée.",
                        RuntimeWarning,
                        stacklevel=2,
                    )
                    continue
                warnings.warn(
                    f"Aucune pondération positive n'est disponible pour le "
                    f"benchmark {self.bench} au {pd.Timestamp(date_):%Y-%m-%d}. "
                    f"Le rééquilibrage prévu au {target_date:%Y-%m-%d} est "
                    "ignoré et les positions précédentes sont prolongées par "
                    "dérive.",
                    RuntimeWarning,
                    stacklevel=2,
                )
                top_results.append(
                    self._drift_selection_to_date(top_results[-1], target_date)
                )
                bottom_results.append(
                    builder_bottom._drift_selection_to_date(
                        bottom_results[-1], target_date,
                    )
                )
                continue
            top_results.append(
                self.sec_list_spot(screen_agg_monthly=monthly_screen)
            )
            bottom_results.append(
                builder_bottom._finalize_sec_list_spot(
                    self._last_monthly_preparation,
                )
            )

        if not top_results:
            empty = pd.DataFrame()
            self.sec_list_historical = empty.copy()
            builder_bottom.sec_list_historical = empty.copy()
            return empty, empty

        top_history = pd.concat(top_results, ignore_index=True)
        bottom_history = pd.concat(bottom_results, ignore_index=True)
        self.sec_list_historical = top_history.copy()
        builder_bottom.sec_list_historical = bottom_history.copy()

        if fill_method == "drift":
            self.sec_list_historical = self.update_ptf_with_monthly_drift(
                self.sec_list_historical,
            )
            builder_bottom.sec_list_historical = (
                builder_bottom.update_ptf_with_monthly_drift(
                    builder_bottom.sec_list_historical,
                )
            )
        elif fill_method == "copy":
            self.sec_list_historical = self.update_ptf_with_monthly_additions(
                self.sec_list_historical,
            )
            builder_bottom.sec_list_historical = (
                builder_bottom.update_ptf_with_monthly_additions(
                    builder_bottom.sec_list_historical,
                )
            )

        print(
            "Les listes historiques Top et Worst sont disponibles dans "
            "l'attribut 'sec_list_historical' de chaque builder."
        )
        return top_history, bottom_history

    def _drift_selection_to_date(self, selection, target_date):
        """Prolonge une sélection existante jusqu'à une date sans rééquilibrage."""
        if 'ISIN' in self.screen.columns:
            references = (
                self.screen.loc[:, ['ISIN', 'Company SEDOL']]
                .dropna(subset=['ISIN', 'Company SEDOL'])
                .drop_duplicates(subset='ISIN', keep='last')
                .set_index('ISIN')['Company SEDOL']
            )
        else:
            references = self.screen['Company SEDOL']
            references = references.loc[
                ~references.index.duplicated(keep='last')
            ]

        selection_to_drift = selection.copy()
        selection_to_drift.loc[:, 'SEDOL'] = selection_to_drift['ISIN'].map(
            references,
        )
        if not selection_to_drift['SEDOL'].isin(self.returns.columns).any():
            raise ValueError(
                "Aucun rendement n'est disponible pour prolonger les positions "
                f"jusqu'au {pd.Timestamp(target_date):%Y-%m-%d}."
            )

        drifted = self.drift_weight(
            selection_to_drift,
            'SEDOL',
            self.returns.copy(),
            'Date',
            'Weight',
            pd.Timestamp(target_date),
        )
        drifted.loc[:, 'Date'] = pd.Timestamp(target_date)
        return drifted.drop(columns='SEDOL', errors='ignore')
    

    def backtest_calcul_all_portfolio(self,df_rebal, df_returns, col_weight,col_sector = ' Benchmark ICB Supersector ', col_date='Date', col_id = 'Company SEDOL'):
        """
        permet de générer les returns des portfolios
        df_rebal: contient les coefficients de poids 
        df_returns: returns des actifs
        """
        
        # Creating a list of available date in sec list (MONTHLY) - premier jour du mois
        liste_rebal_date = list(df_rebal.index.get_level_values(col_date).unique())
        # Creating a list of date (DAILY), but from returns dataframe, starting from the first date of the sec list
        liste_date_returns = list(df_returns[df_returns.index>=liste_rebal_date[0]].index)

        #filtrer pour avoir la période du portefeuille
        df_rebal.reset_index(inplace=True)
    
        df_rebal = df_rebal.loc[
            df_rebal[col_id].isin(df_returns.columns)
        ].copy()  # SUPPRESSION DES TITRES ABSENTS DES RENDEMENTS
        df_rebal.set_index(col_date,inplace=True)

        # Normalisation
        df_rebal['Portfolio weight'] = (
                                        df_rebal.groupby(col_date)['Portfolio weight']
                                        .transform(lambda x: x / x.sum())
                                        )
        
        df_rebal.reset_index(inplace=True)
    
    
        # Boucle dans le cas d'une date de rebalancement non présente dans df_returns -> changement de la date de rebalancement avec la 2eme date future la plus proche
        for i in range(len(liste_rebal_date)) :
            if  liste_rebal_date[i] not in liste_date_returns :

                try:
                    # Try pour chercher la 2eme date future la plus proche

                    # Supposons que liste_date_returns soit une liste de pd.Timestamp
                    serie_date_returns = pd.Series(liste_date_returns)

                    new_date_rebal = serie_date_returns[serie_date_returns > liste_rebal_date[i]].iloc[1]

                except ValueError:
                    # Si pas de future date trouvée, on prend la date antérieure la plus proche (cas frequency=1 dernière date est une date de rebalancement)
                    new_date_rebal = max(d for d in liste_date_returns if d < liste_rebal_date[i])



            else :
                # Supposons que liste_date_returns soit une liste de pd.Timestamp
                serie_date_returns = pd.Series(liste_date_returns)

                new_date_rebal = serie_date_returns[serie_date_returns > liste_rebal_date[i]].iloc[0]

            df_rebal = df_rebal.replace(liste_rebal_date[i], new_date_rebal)
            liste_rebal_date[i] = new_date_rebal

        # Tri avec la fonction sorted()
        liste_date_all = list(set(liste_rebal_date).union(set(liste_date_returns)))
        nouvelle_liste_dates = sorted(liste_date_all)
    
        #df_rebal = df_rebal.set_index([col_id,col_date])
    
        new_df = pd.DataFrame(data=nouvelle_liste_dates, columns=['Date_returns']) #INSTANCIATION dataframe AVEC COLONNE DE DATE DAILY
        
        # Association de chaque date quotidienne à la dernière date de rebalancement.
        """
        Exemple de transformation vectorisée :

        rebalance_dates = [2021-01-04, 2021-02-01]

        Date_returns  position trouvée  Date_screen
        2021-01-03           -1          NaT
        2021-01-04            0          2021-01-04
        2021-01-05            0          2021-01-04
        2021-01-29            0          2021-01-04
        2021-02-01            1          2021-02-01
        2021-02-02            1          2021-02-01

        searchsorted(..., side='right') - 1 renvoie directement la position
        de la dernière date de rebalancement inférieure ou égale à la date
        quotidienne. Une position négative reste NaT et ne crée aucun faux match.
        """
        rebalance_dates = pd.DatetimeIndex(
            sorted(pd.to_datetime(df_rebal[col_date].dropna().unique()))
        )
        return_dates = pd.DatetimeIndex(new_df['Date_returns'])
        rebalance_positions = rebalance_dates.searchsorted(
            return_dates, side='right',
        ) - 1
        screen_dates = np.full(
            len(return_dates), np.datetime64('NaT'), dtype='datetime64[ns]',
        )
        valid_rebalance_positions = rebalance_positions >= 0
        screen_dates[valid_rebalance_positions] = rebalance_dates.to_numpy()[
            rebalance_positions[valid_rebalance_positions]
        ]
        new_df['Date_screen'] = screen_dates
        
        #ON DUPPLIQUE LES DATES DE SCREEN MENSUEL POUR CHAQUE DATE de new DF dont la colonne Date Screen = col_date de df_rebal (date de rebalancement)
        df_merge = pd.merge(df_rebal,new_df, how='left', left_on=col_date, right_on = 'Date_screen')
        df_merge.drop(columns=col_date, inplace=True)
        df_merge.rename(columns={'Date_returns':col_date},inplace=True) #BONNE COLONNE DE DATE
        df_merge.sort_values(by=col_date, inplace=True)
    
        df_returns = df_returns[new_df['Date_screen'].min():] # ON garde LES RETURN A PARTIR DE LA PREMIERE DATE des returns
        returns_cum = (1+df_returns).cumprod() # On a le ttr calculé pour à partir de la 1ère date de rebalancement
        
        # Rebasage matriciel des rendements cumulés à chaque rebalancement.
        cumulative_dates = pd.DatetimeIndex(returns_cum.index)
        cumulative_rebalance_positions = rebalance_dates.searchsorted(
            cumulative_dates, side='right',
        ) - 1
        cumulative_rebalance_dates = rebalance_dates.take(
            cumulative_rebalance_positions,
        )
        reference_positions = returns_cum.index.get_indexer(
            cumulative_rebalance_dates,
        )
        returns_cum_values = returns_cum.to_numpy(copy=False)
        returns_drift = pd.DataFrame(
            returns_cum_values / returns_cum_values[reference_positions],
            index=returns_cum.index,
            columns=returns_cum.columns,
        )
        """
        Matrice returns_cum avant rebasage :

        Date        Asset_A  Asset_B  Date de référence
        2021-01-04     1.00     1.00  2021-01-04
        2021-01-05     1.10     0.95  2021-01-04
        2021-02-01     1.20     1.05  2021-02-01
        2021-02-02     1.26     1.03  2021-02-01

        Chaque ligne est divisée par la ligne de sa date de référence :

        Date        Asset_A  Asset_B  = returns_drift
        2021-01-04     1.00     1.00  = ligne 1 / ligne 1
        2021-01-05     1.10     0.95  = ligne 2 / ligne 1
        2021-02-01     1.00     1.00  = ligne 3 / ligne 3
        2021-02-02     1.05     0.98  = ligne 4 / ligne 3

        La division NumPy traite toute la matrice en une seule opération.
        """

        # Lecture directe des deux matrices sans les convertir en tables longues.
        """
        df_merge avant enrichissement :

        Date        SEDOL    Portfolio weight
        2021-01-05  Asset_A              0.60
        2021-01-05  Asset_B              0.40
        2021-02-02  Asset_A              0.55

        Les indexeurs convertissent chaque clé en coordonnées matricielles :

        Date        SEDOL    position_date  position_titre
        2021-01-05  Asset_A        1               0
        2021-01-05  Asset_B        1               1
        2021-02-02  Asset_A        3               0

        Lecture directe :

        drift_matrix[1, 0]  -> 1.10     return_matrix[1, 0] ->  0.10
        drift_matrix[1, 1]  -> 0.95     return_matrix[1, 1] -> -0.05
        drift_matrix[3, 0]  -> 1.05     return_matrix[3, 0] ->  0.05

        df_merge après enrichissement :

        Date        SEDOL    Portfolio weight  drift_multiplicator  Return
        2021-01-05  Asset_A              0.60                  1.10    0.10
        2021-01-05  Asset_B              0.40                  0.95   -0.05
        2021-02-02  Asset_A              0.55                  1.05    0.05

        Aucune table longue issue de stack() et aucune fusion supplémentaire
        ne sont nécessaires.
        """
        date_positions = returns_drift.index.get_indexer(
            pd.DatetimeIndex(df_merge[col_date]),
        )
        security_positions = returns_drift.columns.get_indexer(
            df_merge[col_id],
        )
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
        df_merge['drift_multiplicator'] = drift_values
        df_merge['Return'] = return_values
        
        #CHAQUE POIDS DAILY est DRIFTé dont celui du rebal qui est aussi drifté par 1
        df_merge[col_weight+'_drifted'] = df_merge[col_weight]*df_merge['drift_multiplicator'] 
        """ EX. the weight af asset A is 0.6, drift_multiplicator is 1.1, then drifted weight : 0.6 * 1.1 = 0.66 """
        

        # Select the date, asset identifier, original weight, drift-adjusted weight, sector (or segment), 
        # and return data to form a new data frame `portfolio_tet`, facilitating subsequent calculations by date and asset.
        columns = [col_date, col_id, col_weight, col_weight+'_drifted', col_sector, 'Return']
        portfolio_tet=df_merge[columns]
        
        # Sum the drifted weights of all assets by date to obtain the total drift weight of all assets for that day
        weight_sum_date = portfolio_tet.groupby(col_date,group_keys=False)[[col_weight+'_drifted']].sum()
        weight_sum_date.columns = ['Weight_sum']
        weight_sum_date.reset_index(inplace=True)
        
        # Merge this total drift weight into portfolio_tet
        portfolio_tet = portfolio_tet.merge(weight_sum_date, how='left', on = col_date)
        
        # Divide the drifted weight of each asset by the total drift weight of that day to obtain the normalized weight W_rebased. 
        # This ensures that the sum of the normalized weights of all assets for each date equals 1.
        portfolio_tet['W_rebased'] = portfolio_tet[col_weight+'_drifted'] / portfolio_tet['Weight_sum']

        """
        Ex:
        Suppose on a given day, the drifted weight of Asset A is 0.66, and the drifted weight of Asset B is 0.38. The total drifted weight is 0.66 + 0.38 = 1.04.

        The normalized weight of Asset A: 0.66 / 1.04 ≈ 0.635  
        The normalized weight of Asset B: 0.38 / 1.04 ≈ 0.365
        """

        # For each asset (grouped by col_id), shift the normalized weight W_rebased down by one row, i.e., retrieve the normalized weight from the previous day.
        # This is typically done to calculate the daily contribution of each asset by multiplying the previous day's weight BY the current day's return, 
        # thereby determining the asset's contribution to the portfolio's daily return.
        portfolio_tet['W_rebased_shift1'] = portfolio_tet.groupby(
            col_id, observed=False,
        )['W_rebased'].shift(1)
        
        # Calculate the contribution of each asset: multiply the previous day's normalized weight by the current day's return.
        # Then, sum the contributions of all assets by date to obtain the total return contribution of the portfolio for each day.
        portfolio_tet['Contrib'] = portfolio_tet['W_rebased_shift1'] * portfolio_tet['Return']
        total_return_by_date = portfolio_tet.groupby(col_date)['Contrib'].sum()
        """
        Ex.
        If on a given day, Asset A's previous day weight is 0.635 and its current day return is 0.10, its contribution is 0.0635.
        If Asset B's previous day weight is 0.365 and its current day return is -0.05, its contribution is -0.01825.
        Total contribution = 0.0635 + (-0.01825) ≈ 0.04525.
        """

        # Starting with an initial value of 1, add the daily total return contribution (filling missing values with 0) and calculate the cumulative product to obtain the cumulative return of the entire portfolio
        total_return_by_date.sort_index(inplace=True)
        serie_ttr=(1 + total_return_by_date.fillna(0)).cumprod() * 100 
        """
        Example:
        Assume the cumulative calculation is as follows:
        Contribution on the first day is 0.00 → (1 + 0.00) = 1.00
        Contribution on the second day is 0.04525 → Cumulative return is 1.00 × 1.04525 ≈ 1.04525
        Contribution on the third day is 0.02 → Cumulative return is 1.04525 × 1.02 ≈ 1.06516
        Multiplying by 100, the cumulative return is 106.516%, representing a growth of 6.516% relative to the initial value.
        """    

        return serie_ttr
        
    def backtest_create_ptf_weight(self,sec_list, 
                        indice_name, 
                        screen_agg,
                        max_weight ,  
                        col_mkt_cap='Benchmark Market Value Millions in EUR ', 
                        col_date = 'Date', 
                        col_sector = ' Benchmark ICB Supersector ', 
                        sector_neutral=False, method='mkt_cap', 
                        col_sedol = 'Company SEDOL', 
                        col_isin= 'ISIN'
                        ):
        """
        Générer les ptfs en duplicant les poids de l'indice => pour backtest la perf de l'indice par la suite
        C'est une version simplifiée de la fonction "sec_list_spot"       
        """
        # INDICE, SCREENAGGREGATE et SECLIST SERONT INVESTI AU 1er du mois
        # Filter Bench related securities and take the weight of bench as sec list
        indice = screen_agg.loc[screen_agg['Weight in '+indice_name]>0, [col_date, col_sedol,col_sector,'Weight in '+indice_name]].reset_index()
        indice.rename(columns={'Weight in '+indice_name:'Indice weight'}, inplace= True)
    
        indice.sort_values(by=col_date,inplace=True)
        sec_list.sort_values(by=col_date,inplace=True)

        indice[col_date] = indice[col_date] + pd.offsets.MonthBegin(1)
        screen_agg[col_date] = screen_agg[col_date] + pd.offsets.MonthBegin(1)

        # Add some columns of screen in sec list
        sec_list = sec_list.merge(right = screen_agg.reset_index()[[col_date,col_isin,col_sedol,col_sector, col_mkt_cap]], on=[col_date,col_isin], how='left')
        sec_list = sec_list[sec_list[col_sedol].notna()]
    
        if method=='EW': # Equal weight
            sec_list.set_index(col_date,inplace=True)
            sec_list['Portfolio weight'] = sec_list.groupby(col_date, group_keys=False).apply(lambda x: 1/len(x))
            sec_list.reset_index(inplace=True)
        else:
            sec_list = sec_list[sec_list[col_mkt_cap].notna()]                                                                      
            if method == "Racine cube":
                sec_list[col_mkt_cap] = sec_list[col_mkt_cap]**(1/3)
            elif method == "Racine carrée":
                sec_list[col_mkt_cap] = sec_list[col_mkt_cap]**(1/2)
            elif method == "Log":
                sec_list[col_mkt_cap] = np.log(sec_list[col_mkt_cap])
            sec_list.set_index(col_date,inplace=True)
            sec_list['Portfolio weight'] = sec_list[col_mkt_cap]/sec_list.groupby(col_date)[col_mkt_cap].sum()
            sec_list.reset_index(inplace=True)
    
        # Calculate the ratio of the benchmark index's total sector weight to the portfolio's total sector weight, which serves as the adjustment factor for each sector.  
        # Adjust the weight of each stock in the portfolio according to this ratio, ensuring that the total sector weight in the adjusted portfolio matches the sector weight of the benchmark index.
        if sector_neutral:
            indice.set_index(col_date,inplace=True)
            indice['Indice weight'] /= indice.groupby(col_date)['Indice weight'].sum()
            indice.reset_index(inplace=True)
            weight_secto_bench = (indice.groupby([col_date,col_sector])['Indice weight'].sum()).reset_index()
        
            sec_list.set_index(col_date,inplace=True)
            sec_list['Portfolio weight'] /= sec_list.groupby(col_date)['Portfolio weight'].sum()
            sec_list.reset_index(inplace=True)
            sec_list.set_index([col_date,col_sector],inplace=True)
            sec_list['weight_secto_ptf'] = sec_list.groupby([col_date,col_sector],group_keys=False)['Portfolio weight'].sum()
            sec_list.reset_index(inplace=True)
    
            sec_list = sec_list.merge(weight_secto_bench[[col_date,col_sector,'Indice weight']], on=[col_date,col_sector], how='left')
            sec_list['Portfolio weight'] = sec_list['Portfolio weight'] * (sec_list['Indice weight']/sec_list['weight_secto_ptf'])
    
        # Handle outliers
        sec_list.set_index(col_date,inplace=True)
        sec_list['Portfolio weight'] /= sec_list.groupby(col_date)['Portfolio weight'].sum()
        sec_list['Portfolio weight'] = sec_list['Portfolio weight'].apply(lambda x : min(x,max_weight))
        sec_list['Portfolio weight'] /= sec_list.groupby(col_date)['Portfolio weight'].sum()
        sec_list.reset_index(inplace=True)
        
        return sec_list[[col_date, col_sedol,col_isin, 'Portfolio weight', col_sector]].set_index([col_date,col_sedol])
    
    def backtest(self,sec_list=None,
                indice_name=None,
                method=None,    
                max_weight = 1, 
                col_sector= ' Benchmark ICB Supersector ', 
                col_sedol='Company SEDOL', 
                col_isin='ISIN', 
                col_date = 'Date', 
                col_mkt_cap = 'Benchmark Market Value Millions in EUR ', 
                sector_neutral=False,
                sec_list_=True, 
                ponderation='mkt_cap', critere='Score ML',
                max_weights= [0.025,0.015,0.02,0.02,0.02,0.02,0.02,0.03,0.015,0.02,0.02,0.035,0.03,0.02,0.02,0.02,0.02,0.02,0.02], 
                list_secto=[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19], repechage_filter=['Sector ICB19'], 
                nb_titres= 150, te_max= 0.03,rebalancing_start_backward=None):
        # if sec_list is not provided used self.sec_list
        if  sec_list is None:
                if self.sec_list_historical != None:
                    sec_list=self.sec_list_historical
                else:
                    sec_list=self.generic_histo_seclist( method=method, critere=critere,max_weights= max_weights, 
                            list_secto=list_secto, repechage_filter=repechage_filter, 
                            nb_titres= nb_titres, te_max=te_max,rebalancing_start_backward=rebalancing_start_backward)

        # if input is path, then read the parquet, if it's a df, then use it directly
        if type(self.screen)==str:
            screen_agg = pd.read_parquet(self.screen)
        else:
            screen_columns = [
                col_date, col_sedol, col_sector, col_mkt_cap,
                'Weight in ' + self.bench,
            ]
            if col_isin in self.screen.columns:
                screen_columns.insert(1, col_isin)
            screen_agg = self.screen.loc[:, screen_columns].copy()
    
        if type(self.returns)==str:
            df_returns = pd.read_parquet(self.returns)
        else:
            df_returns=self.returns

        # Loading sec_list
        buy_list = copy.deepcopy(sec_list)
        
        # For a normal ptf that weight column is included
        if sec_list_ :
            if 'Weight' in buy_list.columns:
                # AVOIR UNE SECLIST AU 1ER DU MOIS pour MATCHER AVEC SCREEN AGGREGATE QUI SERA SHIFTé du 31 du mois au 1er du mois suivant
                sec_list_full = buy_list[[col_date,col_isin,'Weight']].copy()  ## COPY for avoid warning

                ### Rebalancing weight for each date ###
                sec_list_full['Weight'] = (
                                            sec_list_full.groupby(col_date)['Weight']
                                            .transform(lambda w: w / w.sum())
                                            )
                
                # Outliers transformation into [0, 1]
                sec_list_full['Weight'] = sec_list_full['Weight'].apply(lambda x : max(x,0))
                sec_list_full['Weight'] = sec_list_full['Weight'].apply(lambda x : min(x,max_weight))


                ### Redo rebalancing
                sec_list_full["WeightSum"] = sec_list_full.groupby("Date")["Weight"].transform("sum")
                sec_list_full['Weight'] /= sec_list_full["WeightSum"]

                sec_list_full.reset_index(inplace=True)

                sec_list_full.rename(columns={'Weight':'Portfolio weight'},inplace=True) # Rename column of weight

                # Make sure that column of date is datetime format
                screen_agg[col_date] = pd.to_datetime(screen_agg[col_date])

                # Then push the date to the first day of the next month
                screen_agg[col_date] = screen_agg[col_date] + pd.offsets.MonthBegin(1)

                # Generating final seclist
                sec_list_full = sec_list_full.merge(right = screen_agg.reset_index()[[col_date,col_isin,col_sedol,col_sector, col_mkt_cap]], on=[col_date,col_isin], how='left')
                sec_list_full = sec_list_full[sec_list_full[col_sedol].notna()] # Remove empty sedol companies
                sec_list_full = sec_list_full[[col_date, col_sedol,col_isin, 'Portfolio weight', col_sector]].set_index([col_date,col_sedol])
                
                # Calcule TTR
                perf_ttr = self.backtest_calcul_all_portfolio(sec_list_full, df_returns, 'Portfolio weight', col_sector, col_date, col_sedol)
                self.perf_ptf, self.buy_list=perf_ttr, sec_list_full[[col_date,col_isin,'Portfolio weight', col_sector]]
                print('Performance of sec_list is calculated, please check attribute "self.perf_ptf" for more details')
            else:
                print("Is not a sec_list")
            
        # For generating all titles sec list for a BENCHMARK
        else:
            # AVOIR UNE SECLIST AU 1ER DU MOIS pour MATCHER AVEC SCREEN AGGREGATE QUI SERA SHIFTé du 31 du mois au 1er du mois suivant
            sec_list_full = self.backtest_create_ptf_weight(buy_list, indice_name, screen_agg, max_weight, col_mkt_cap, col_date, col_sector, sector_neutral,ponderation,col_sedol, col_isin)
            perf_ttr = self.backtest_calcul_all_portfolio(sec_list_full, df_returns, 'Portfolio weight', col_sector, col_date, col_sedol)
            
            self.perf_bench = perf_ttr
            print('Performance of benchmark is calculated, please check attribute "self.perf_bench" for more details')

        return perf_ttr, self.buy_list

    def backtest_get_bench_perf(self,screen,start_date,bench):
        """
        Calculer la perf de l'indice choisi
        """
        indice_ref = screen[(screen['Date']>=start_date) & (screen['Weight in '+bench]>0)].reset_index()[['Date','ISIN']]
        indice_ref["Date"] = pd.to_datetime(indice_ref["Date"])
        indice_ref["Date"] = indice_ref["Date"] + pd.offsets.MonthBegin(1)
        self.backtest(sec_list=indice_ref,indice_name=bench,sec_list_=False)

    def backtest_plot_ptf_bench(self, perf_ptf=None, perf_bench=None, title=None, save_path="portfolio_performance.html", show_plot=True):
        """
        - Avoir tous les perfs : ptf et bench
        - Ploter les perfs
        """
        if self.perf_ptf is None:
            perf_ptf, buy_list = self.backtest(self.sec_list_historical)
        perf_ptf, buy_list = self.perf_ptf, self.buy_list

        if self.perf_bench is None:
            self.backtest_get_bench_perf(self.screen, self.start_date, self.bench)
        perf_bench = self.perf_bench
        
        # Concatenate dataframes
        df_plot = pd.concat([perf_ptf, perf_bench], axis=1)

        # Create subplots
        fig = make_subplots(rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
                            subplot_titles=("Performance", "Ratio"))

        # Add traces for performance
        for i, col in enumerate(df_plot.columns):
            label = 'Perf PTF' if i == 0 else 'Perf Bench'

            # Add line trace
            fig.add_trace(go.Scatter(
                x=df_plot.index,
                y=df_plot.iloc[:, i],
                mode='lines',
                name=label,
                line=dict(width=2)
            ), row=1, col=1)

            # Add annotation for last value
            last_x = df_plot.index[-1]
            last_y = df_plot.iloc[:, i].iloc[-1]

            fig.add_annotation(
                x=last_x,
                y=last_y,
                text=f'{last_y:.2f}',
                showarrow=False,
                xanchor='left',
                font=dict(size=10),
                bgcolor='rgba(255,255,255,0.8)',
                bordercolor='rgba(0,0,0,0.2)',
                borderwidth=1
            )

        # Add trace for the ratio
        ratio = df_plot.iloc[:, 0] / df_plot.iloc[:, 1]
        fig.add_trace(go.Scatter(
            x=df_plot.index,
            y=ratio,
            mode='lines',
            name='Ratio',
            line=dict(width=2, color='red')
        ), row=2, col=1)

        # Add annotation for last value of the ratio
        last_ratio = ratio.iloc[-1]
        fig.add_annotation(
            x=last_x,
            y=last_ratio,
            text=f'{last_ratio:.2f}',
            showarrow=False,
            xanchor='left',
            font=dict(size=10),
            bgcolor='rgba(255,255,255,0.8)',
            bordercolor='rgba(0,0,0,0.2)',
            borderwidth=1
        )

        # Update layout
        fig.update_layout(
            title=title if title else "",
            width=700,
            height=600,
            showlegend=True,
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1
            ),
            margin=dict(l=50, r=50, t=50, b=50),
            plot_bgcolor='white',
            paper_bgcolor='white'
        )

        # Update axes
        fig.update_xaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )

        fig.update_yaxes(
            showgrid=True,
            gridwidth=1,
            gridcolor='lightgray',
            showline=True,
            linewidth=1,
            linecolor='black'
        )

        # Handle different environments
        if save_path:
            # Save as HTML file
            fig.write_html(save_path)
            print(f"Plot saved as HTML to: {save_path}")

        if show_plot:
            try:
                # Try to show in browser
                fig.show()
            except Exception as e:
                print(f"Cannot display plot directly: {e}")
                # Save as temporary HTML file and provide instructions
                temp_path = "temp_plot.html"
                fig.write_html(temp_path)
                print(f"Plot saved as HTML to: {temp_path}")
                print("Please open this file in your web browser to view the plot.")

    def _calculate_robust_score(self, perf_top, perf_bottom, perf_bench,
                                allow_short_history=False,
                                store_metrics=True):
        """
        Calcule un score de robustesse à partir des performances relatives.

        Le score récompense la surperformance face au benchmark et au portefeuille
        Worst, puis pénalise le drawdown actif, la tracking error annualisée et le
        pire CAGR glissant sur trois ans du ratio Top/Benchmark.
        """
        window = 252 * 3
        performances = pd.concat(
            [perf_top.rename('Top'), perf_bottom.rename('Worst'), perf_bench.rename('Bench')],
            axis=1,
        ).dropna()

        if len(performances) < window and not allow_short_history:
            robust_metrics = {
                'active_max_drawdown': float('nan'),
                'tracking_error_annualized': float('nan'),
                'min_rolling_3y_cagr': float('nan'),
                'observation_count': len(performances),
            }
            if store_metrics:
                self.robust_metrics = robust_metrics
            return float('nan'), float('nan'), float('nan')

        if len(performances) < 2:
            robust_metrics = {
                'active_max_drawdown': float('nan'),
                'tracking_error_annualized': float('nan'),
                'min_rolling_3y_cagr': float('nan'),
                'observation_count': len(performances),
            }
            if store_metrics:
                self.robust_metrics = robust_metrics
            return float('nan'), float('nan'), float('nan')

        perf_top = performances['Top']
        perf_bottom = performances['Worst']
        perf_bench = performances['Bench']

        ret_top = perf_top.iloc[-1] / max(perf_top.iloc[0], 1e-6)
        ret_bench = perf_bench.iloc[-1] / max(perf_bench.iloc[0], 1e-6)
        ret_bottom = perf_bottom.iloc[-1] / max(perf_bottom.iloc[0], 1e-6)

        top_bench_ratio = (ret_top / max(ret_bench, 1e-6)) - 1
        top_worst_ratio = (ret_top / max(ret_bottom, 1e-6)) - 1

        def get_max_dd(series):
            roll_max = series.cummax()
            drawdown = (roll_max - series) / roll_max
            return drawdown.max()

        relative_perf = perf_top / perf_bench
        active_mdd = get_max_dd(relative_perf)

        diff_returns = perf_top.pct_change() - perf_bench.pct_change()
        tracking_error_annualized = diff_returns.std() * (252 ** 0.5)

        rolling_ratio = perf_top / perf_bench
        effective_window = min(window, len(performances))
        effective_years = 3.0 if effective_window == window else (
            (effective_window - 1) / 252
        )
        start_vals = rolling_ratio.shift(effective_window - 1)
        rolling_cagr = (rolling_ratio / start_vals) ** (1 / effective_years) - 1
        min_rolling_cagr = abs(min(rolling_cagr.min(), 0))

        robust_score = (
            top_bench_ratio
            + 0.5 * top_worst_ratio
            - 2 * active_mdd
            - tracking_error_annualized
            - min_rolling_cagr
        )
        robust_metrics = {
            'active_max_drawdown': active_mdd,
            'tracking_error_annualized': tracking_error_annualized,
            'min_rolling_3y_cagr': min_rolling_cagr,
            'observation_count': len(performances),
        }
        if store_metrics:
            self.robust_metrics = robust_metrics
        return robust_score, top_bench_ratio, top_worst_ratio

    def _calculate_classic_metrics(self, performance, benchmark=None,
                                   periods_per_year=252, risk_free_rate=0.0):
        """Calcule les métriques classiques à partir d'une courbe de valeur cumulée."""
        performance = performance.dropna().astype(float)
        metric_names = (
            'total_return', 'annualized_return', 'annualized_volatility',
            'sharpe_ratio', 'max_drawdown', 'sortino_ratio',
            'beta', 'tracking_error', 'information_ratio',
        )
        if len(performance) < 2:
            return {name: float('nan') for name in metric_names}

        returns = performance.pct_change().dropna()
        total_return = performance.iloc[-1] / performance.iloc[0] - 1
        years = (len(performance) - 1) / periods_per_year
        annualized_return = (
            (performance.iloc[-1] / performance.iloc[0]) ** (1 / years) - 1
            if years > 0 else float('nan')
        )
        annualized_volatility = returns.std() * (periods_per_year ** 0.5)
        daily_risk_free = (1 + risk_free_rate) ** (1 / periods_per_year) - 1
        excess_returns = returns - daily_risk_free
        sharpe_ratio = (
            excess_returns.mean() / excess_returns.std() * (periods_per_year ** 0.5)
            if excess_returns.std() > 0 else float('nan')
        )

        running_max = performance.cummax()
        max_drawdown = ((running_max - performance) / running_max).max()
        downside_returns = excess_returns[excess_returns < 0]
        downside_deviation = downside_returns.std()
        sortino_ratio = (
            excess_returns.mean() / downside_deviation * (periods_per_year ** 0.5)
            if pd.notna(downside_deviation) and downside_deviation > 0
            else float('nan')
        )

        beta = float('nan')
        tracking_error = float('nan')
        information_ratio = float('nan')
        if benchmark is not None:
            aligned = pd.concat(
                [performance.rename('Portfolio'), benchmark.rename('Benchmark')], axis=1,
            ).dropna()
            aligned_returns = aligned.pct_change().dropna()
            if not aligned_returns.empty:
                benchmark_variance = aligned_returns['Benchmark'].var()
                if benchmark_variance > 0:
                    beta = (
                        aligned_returns['Portfolio'].cov(aligned_returns['Benchmark'])
                        / benchmark_variance
                    )
                active_returns = (
                    aligned_returns['Portfolio'] - aligned_returns['Benchmark']
                )
                tracking_error = active_returns.std() * (periods_per_year ** 0.5)
                if tracking_error > 0:
                    information_ratio = (
                        active_returns.mean() * periods_per_year / tracking_error
                    )

        return {
            'total_return': total_return,
            'annualized_return': annualized_return,
            'annualized_volatility': annualized_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown,
            'sortino_ratio': sortino_ratio,
            'beta': beta,
            'tracking_error': tracking_error,
            'information_ratio': information_ratio,
        }

    def _calculate_metrics_for_periods(self, performance, periods):
        """Calcule les métriques locales pour des fenêtres datées explicites."""
        columns = (
            'period_id', 'period_label', 'requested_start_date',
            'requested_end_date', 'actual_start_date', 'actual_end_date',
            'observation_count', 'years', 'top_cagr', 'worst_cagr',
            'bench_cagr', 'active_cagr', 'top_worst_cagr',
            'robust_score', 'top_bench_ratio', 'top_worst_ratio',
            'period_group', 'window_months', 'step_months',
            'robust_score_comparable',
        )
        periods = list(periods or [])
        if performance.empty or not periods:
            return pd.DataFrame(columns=columns)

        data = performance.copy().sort_index()
        data.index = pd.to_datetime(data.index, errors='coerce')
        data = data.loc[data.index.notna()]
        rows = []

        def calculate_cagr(series, years):
            series = series.dropna().astype(float)
            if len(series) < 2 or years <= 0:
                return float('nan')
            first_value = series.iloc[0]
            last_value = series.iloc[-1]
            if first_value <= 0 or last_value <= 0:
                return float('nan')
            return (last_value / first_value) ** (1 / years) - 1

        for index, period in enumerate(periods):
            period_id = period.get('id') or f'period_{index + 1}'
            period_label = period.get('label') or period_id
            requested_start = period.get('start')
            requested_end = period.get('end')
            start_date = pd.Timestamp(requested_start) if requested_start else None
            end_date = pd.Timestamp(requested_end) if requested_end else None
            mask = pd.Series(True, index=data.index)
            if start_date is not None:
                mask &= data.index >= start_date
            if end_date is not None:
                mask &= data.index <= end_date
            period_data = data.loc[mask.to_numpy()].dropna()
            row = {
                'period_id': period_id,
                'period_label': period_label,
                'requested_start_date': requested_start,
                'requested_end_date': requested_end,
                'actual_start_date': None,
                'actual_end_date': None,
                'observation_count': len(period_data),
                'years': float('nan'),
                'top_cagr': float('nan'),
                'worst_cagr': float('nan'),
                'bench_cagr': float('nan'),
                'active_cagr': float('nan'),
                'top_worst_cagr': float('nan'),
                'robust_score': float('nan'),
                'top_bench_ratio': float('nan'),
                'top_worst_ratio': float('nan'),
                'period_group': period.get('period_group', 'economic'),
                'window_months': period.get('window_months'),
                'step_months': period.get('step_months'),
                'robust_score_comparable': period.get(
                    'robust_score_comparable', True,
                ),
            }
            if len(period_data) >= 2:
                first_date = period_data.index[0]
                last_date = period_data.index[-1]
                years = max((last_date - first_date).days / 365.25, 1 / 12)
                row.update({
                    'actual_start_date': first_date.date().isoformat(),
                    'actual_end_date': last_date.date().isoformat(),
                    'years': years,
                    'top_cagr': calculate_cagr(period_data['Top'], years),
                    'worst_cagr': calculate_cagr(period_data['Worst'], years),
                    'bench_cagr': calculate_cagr(period_data['Bench'], years),
                    'active_cagr': calculate_cagr(
                        period_data['Top'] / period_data['Bench'], years,
                    ),
                    'top_worst_cagr': calculate_cagr(
                        period_data['Top'] / period_data['Worst'], years,
                    ),
                })
                for portfolio in ('Top', 'Worst', 'Bench'):
                    metrics = self._calculate_classic_metrics(
                        period_data[portfolio], benchmark=period_data['Bench'],
                    )
                    row.update({
                        f'{portfolio.lower()}_{metric}': value
                        for metric, value in metrics.items()
                    })
                robust_score, top_bench_ratio, top_worst_ratio = (
                    self._calculate_robust_score(
                        period_data['Top'],
                        period_data['Worst'],
                        period_data['Bench'],
                        allow_short_history=True,
                        store_metrics=False,
                    )
                )
                row.update({
                    'robust_score': robust_score,
                    'top_bench_ratio': top_bench_ratio,
                    'top_worst_ratio': top_worst_ratio,
                })
            rows.append(row)
        return pd.DataFrame(rows)

    def _calculate_period_metrics(self, performance, period_breakpoints):
        """Calcule les métriques des segments créés par les années de rupture."""
        periods = build_periods_from_breakpoints(period_breakpoints or [])
        return self._calculate_metrics_for_periods(performance, periods)

    def calculate_top_vs_bottom_results(self, builder_bottom, period_breakpoints=None):
        """Calcule les performances et les métriques sans construire de figure."""
        if self.perf_ptf is None:
            if self.sec_list_historical is None:
                self.generic_histo_seclist(
                    start_date=self.start_date,
                    freq_rebal=getattr(self, 'freq_rebal', None),
                    fill_method=getattr(self, 'fill_method', 'drift'),
                )
            self.backtest(self.sec_list_historical)
        perf_top = self.perf_ptf

        if builder_bottom.perf_ptf is None:
            if builder_bottom.sec_list_historical is None:
                builder_bottom.generic_histo_seclist(
                    start_date=self.start_date,
                    freq_rebal=getattr(self, 'freq_rebal', None),
                    fill_method=getattr(self, 'fill_method', 'drift'),
                )
            builder_bottom.backtest(builder_bottom.sec_list_historical)
        perf_bottom = builder_bottom.perf_ptf

        if self.perf_bench is None:
            self.backtest_get_bench_perf(self.screen, self.start_date, self.bench)
        perf_bench = self.perf_bench

        robust_score, top_bench_ratio, top_worst_ratio = self._calculate_robust_score(
            perf_top, perf_bottom, perf_bench,
        )
        performance = pd.concat([perf_top, perf_bottom, perf_bench], axis=1).dropna()
        performance.columns = ['Top', 'Worst', 'Bench']
        ratios = pd.DataFrame({
            'Top/Bench': performance['Top'] / performance['Bench'],
            'Worst/Bench': performance['Worst'] / performance['Bench'],
            'Top/Worst': performance['Top'] / performance['Worst'],
        })
        classic_metrics = {
            portfolio: self._calculate_classic_metrics(
                performance[portfolio], benchmark=performance['Bench'],
            )
            for portfolio in ('Top', 'Worst', 'Bench')
        }
        period_metrics = self._calculate_period_metrics(performance, period_breakpoints)

        result = {
            'robust_score': robust_score,
            'top_bench_ratio': top_bench_ratio,
            'top_worst_ratio': top_worst_ratio,
            'performance': performance,
            'ratios': ratios,
            'classic_metrics': classic_metrics,
            'period_metrics': period_metrics,
        }
        result.update(getattr(self, 'robust_metrics', {}))
        for portfolio, metrics in classic_metrics.items():
            result.update({
                f'{portfolio.lower()}_{metric}': value
                for metric, value in metrics.items()
            })
        return result


    def plot_top_vs_bottom_results(
            self, result, title=None,
            save_path="comparison.html", show_plot=True):
        """Construit uniquement la figure à partir de résultats déjà calculés."""
        performance = result['performance']
        ratios = result['ratios']
        period_metrics = result.get('period_metrics', pd.DataFrame())
        robust_score = result.get('robust_score', float('nan'))
        top_bench_ratio = result.get('top_bench_ratio', float('nan'))
        top_worst_ratio = result.get('top_worst_ratio', float('nan'))

        def rebase_frame(frame, start=None, end=None, base_value=100.0):
            """Rebase chaque série sur la première valeur valide de la fenêtre."""
            rebased = frame.copy()
            window = frame.loc[start:end] if start is not None or end is not None else frame
            for column in frame.columns:
                valid = window[column].dropna()
                valid = valid[valid.ne(0)]
                if not valid.empty:
                    rebased[column] = frame[column] / valid.iloc[0] * base_value
            return rebased

        displayed_performance = rebase_frame(performance, base_value=100.0)
        displayed_ratios = rebase_frame(ratios, base_value=1.0)
        default_title = (
            f"{title if title else 'Comparaison Top/Worst'} | "
            f"Score de robustesse : {robust_score:.4f} | "
            f"T/B: {top_bench_ratio:.4f} | "
            f"T/W: {top_worst_ratio:.4f}"
        )

        fig = make_subplots(
            rows=2, cols=1, shared_xaxes=True, vertical_spacing=0.1,
            subplot_titles=('Performance cumulée', 'Ratios relatifs'),
        )

        colors = {'Top': 'blue', 'Worst': 'orange', 'Bench': 'black'}
        for column in displayed_performance.columns:
            fig.add_trace(
                go.Scatter(
                    x=displayed_performance.index,
                    y=displayed_performance[column],
                    mode='lines',
                    name=column,
                    line=dict(width=2, color=colors[column]),
                ),
                row=1, col=1,
            )

        ratio_colors = {
            'Top/Bench': 'blue',
            'Worst/Bench': 'orange',
            'Top/Worst': 'green',
        }
        for column in displayed_ratios.columns:
            fig.add_trace(
                go.Scatter(
                    x=displayed_ratios.index,
                    y=displayed_ratios[column],
                    mode='lines',
                    name=f'Ratio {column}',
                    line=dict(width=2, color=ratio_colors.get(column)),
                ),
                row=2, col=1,
            )

        fig.update_layout(
            title=default_title,
            width=800,
            height=700,
            template='plotly_white',
            legend=dict(
                orientation='h', yanchor='bottom', y=1.02,
                xanchor='right', x=1,
            ),
        )

        # Le menu conserve une seule figure interactive tout en changeant la fenêtre analysée.
        default_y = [
            displayed_performance[column].tolist()
            for column in displayed_performance.columns
        ] + [
            displayed_ratios[column].tolist()
            for column in displayed_ratios.columns
        ]
        default_x = [
            displayed_performance.index.tolist()
            for _ in displayed_performance.columns
        ] + [
            displayed_ratios.index.tolist()
            for _ in displayed_ratios.columns
        ]
        period_buttons = [{
            'label': 'Période totale',
            'method': 'update',
            'args': [
                {'x': default_x, 'y': default_y},
                {
                    'xaxis.autorange': True,
                    'xaxis2.autorange': True,
                    'title.text': default_title,
                },
            ],
        }]
        for _, period_row in period_metrics.iterrows():
            actual_start = period_row.get('actual_start_date')
            actual_end = period_row.get('actual_end_date')
            if not actual_start or not actual_end:
                continue

            def format_percentage(value):
                return f'{value:.2%}' if pd.notna(value) else 'n.d.'

            period_title = (
                f"{title if title else 'Comparaison Top/Worst'} | "
                f"{period_row['period_label']} | "
                f"Top CAGR: {format_percentage(period_row['top_cagr'])} | "
                f"CAGR actif: {format_percentage(period_row['active_cagr'])} | "
                f"Top/Worst CAGR: {format_percentage(period_row['top_worst_cagr'])}"
            )
            period_performance = rebase_frame(
                performance, start=actual_start, end=actual_end, base_value=100.0,
            ).loc[actual_start:actual_end]
            period_ratios = rebase_frame(
                ratios, start=actual_start, end=actual_end, base_value=1.0,
            ).loc[actual_start:actual_end]
            period_y = [
                period_performance[column].tolist()
                for column in period_performance.columns
            ] + [
                period_ratios[column].tolist()
                for column in period_ratios.columns
            ]
            period_x = [
                period_performance.index.tolist()
                for _ in period_performance.columns
            ] + [
                period_ratios.index.tolist()
                for _ in period_ratios.columns
            ]
            period_buttons.append({
                'label': period_row['period_label'],
                'method': 'update',
                'args': [
                    {'x': period_x, 'y': period_y},
                    {
                        'xaxis.autorange': True,
                        'xaxis2.autorange': True,
                        'title.text': period_title,
                    },
                ],
            })
        if len(period_buttons) > 1:
            fig.update_layout(
                updatemenus=[{
                    'buttons': period_buttons,
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
                margin=dict(t=145),
            )

        if save_path:
            fig.write_html(save_path)
        if show_plot:
            fig.show()
        return fig


    def backtest_plot_top_vs_bottom(
            self, builder_bottom, title=None,
            save_path="comparison.html", show_plot=True,
            period_breakpoints=None):
        """Conserve l'ancien point d'entrée en séparant calcul et affichage."""
        result = self.calculate_top_vs_bottom_results(
            builder_bottom=builder_bottom,
            period_breakpoints=period_breakpoints,
        )
        display_title = title if title else 'Facteur'
        print(
            f"Résultats pour {display_title} : "
            f"score de robustesse {result['robust_score']:.4f}, "
            f"Top/Bench {result['top_bench_ratio']:.4f}, "
            f"Top/Worst {result['top_worst_ratio']:.4f}"
        )
        result['figure'] = self.plot_top_vs_bottom_results(
            result=result,
            title=title,
            save_path=save_path,
            show_plot=show_plot,
        )
        return result

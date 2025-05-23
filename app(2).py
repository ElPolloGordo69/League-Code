import gradio as gr
import pandas as pd
import numpy as np
import os
import tempfile
from fuzzywuzzy import process
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import base64
import datetime
import getpass
import os

def filter_problematic_players(df):
    """Remove players with common names that cause issues"""
    if df is None or len(df) == 0 or 'Name' not in df.columns:
        return df
        
    # Create a copy to avoid modifying the original
    filtered_df = df.copy()
    
    # Define exact problematic name combinations (more specific)
    problematic_names = [
        "jose ramirez", "josé ramirez", "jose ramírez", "josé ramírez",
        "julio rodriguez", "julio rodríguez",
        "will smith",
        "luis garcia", "luis garcía", 
        "luis castillo"
    ]
    
    # Also filter specific players by request
    banned_players = [
        "edwin diaz", "edwin díaz",
        "mason miller",
        "cade smith"
    ]
    
    # Filter out exact name matches (case insensitive)
    for name in problematic_names + banned_players:
        filtered_df = filtered_df[~filtered_df['Name'].str.lower().str.strip().eq(name)]
    
    # Remove duplicates by Name
    filtered_df = filtered_df.drop_duplicates(subset=['Name'])
    
    return filtered_df

def validate_pitcher_data(df):
    """Verify that the dataframe contains pitcher stats"""
    # Check for pitcher-specific columns (case-insensitive)
    pitcher_columns = ['era', 'whip', 'ip', 'w', 'sv', 'k/9', 'k/bb']
    
    # Count how many pitcher columns are found
    found_columns = sum(1 for col in pitcher_columns 
                       if any(pc.lower() == col.lower() for pc in df.columns))
    
    if found_columns >= 2:
        return True
    
    # If not enough columns matched, check for ERA specifically 
    # (most distinctive pitcher stat)
    for col in df.columns:
        if col.lower() == 'era':
            return True
    
    # Otherwise, it's likely not pitcher data
    return False

def analyze_standings(standings_file, my_team_name):
    """
    Analyze standings to find opportunity clusters and create category weights
    """
    # Load standings
    try:
        standings = pd.read_csv(standings_file)
        print(f"DEBUG: Standings file loaded successfully with {len(standings)} rows")
        print(f"DEBUG: Columns found: {standings.columns.tolist()}")
    except Exception as e:
        print(f"Error loading standings file: {str(e)}")
        return None
    
    # Make sure we have a team column - try various common names
    team_cols = ['team', 'team name', 'name', 'teamname', 'tm', 'team_name', 'franchise', 'owner']
    team_col = next((c for c in standings.columns if c.lower() in team_cols or 
                     any(tc in c.lower() for tc in team_cols)), None)
    
    if not team_col:
        print(f"DEBUG: No obvious team column. Trying all columns...")
        # If no obvious team column, try each string column
        string_cols = [col for col in standings.columns if standings[col].dtype == object]
        for col in string_cols:
            if my_team_name in standings[col].values:
                team_col = col
                print(f"DEBUG: Found team name in column '{col}'")
                break
                
    if not team_col:
        print(f"No team name column found in standings. Available columns: {standings.columns.tolist()}")
        return None
    
    print(f"DEBUG: Using '{team_col}' as team column")
    print(f"DEBUG: Teams found: {sorted(standings[team_col].unique().tolist())}")
    
    # Find my team with flexible matching
    my_team = None
    
    # Exact match
    matching_teams = standings[standings[team_col] == my_team_name]
    if len(matching_teams) > 0:
        my_team = matching_teams
        print(f"DEBUG: Found exact match for team '{my_team_name}'")
    else:
        # Case-insensitive match
        matching_teams = standings[standings[team_col].str.lower() == my_team_name.lower()]
        if len(matching_teams) > 0:
            my_team = matching_teams
            print(f"DEBUG: Found case-insensitive match for team '{my_team_name}'")
        else:
            # Strip whitespace and match
            matching_teams = standings[standings[team_col].str.strip().str.lower() == my_team_name.strip().lower()]
            if len(matching_teams) > 0:
                my_team = matching_teams
                print(f"DEBUG: Found match after stripping whitespace for team '{my_team_name}'")
            else:
                # Fuzzy match
                best_match = None
                best_score = 0
                
                for team in standings[team_col].dropna().unique():
                    if not isinstance(team, str):
                        continue
                    score = process.fuzz.ratio(my_team_name.lower(), team.lower())
                    print(f"DEBUG: Fuzzy match score for '{team}': {score}")
                    if score > best_score:
                        best_score = score
                        best_match = team
                
                if best_score > 70:  # Lower threshold for more matches
                    print(f"DEBUG: Using fuzzy-matched team name '{best_match}' (score: {best_score})")
                    my_team = standings[standings[team_col] == best_match]
                else:
                    print(f"DEBUG: No close match found for team '{my_team_name}'")
                    
    if my_team is None or len(my_team) == 0:
        return None
    
    # Get the actual team name as it appears in the data
    actual_team_name = my_team.iloc[0][team_col]
    print(f"DEBUG: Using team '{actual_team_name}' for analysis")
    
    # Get statistical categories (excluding team name, total points, etc.)
    exclude_terms = ['rank', 'total', 'points', 'place', 'position', 'team', 'name', 'wins', 'losses']
    stat_cols = []
    
    for col in standings.columns:
        if col == team_col:
            continue
        
        # Skip columns with excluded terms
        if any(term in col.lower() for term in exclude_terms):
            continue
            
        # Only include numeric columns
        if standings[col].dtype in [np.float64, np.int64]:
            stat_cols.append(col)
    
    if not stat_cols:
        print(f"No statistical categories found in standings. Available columns: {standings.columns.tolist()}")
        return None
        
    print(f"DEBUG: Statistical categories found: {stat_cols}")
    
    # Calculate weights for each category
    weights = {}
    num_teams = len(standings)
    
    for cat in stat_cols:
        # Sort standings by this category - check if higher is better
        # For categories like ERA, WHIP, HR/9 - lower is better
        lower_better = any(term in cat.lower() for term in ['era', 'whip', 'hr/9', 'hr9', 'home runs per'])
        
        cat_standings = standings.sort_values(by=cat, ascending=lower_better)
        
        # Find my team's position (1-indexed)
        my_rows = cat_standings[cat_standings[team_col] == actual_team_name]
        if len(my_rows) == 0:
            print(f"DEBUG: Team '{actual_team_name}' not found in category {cat} rankings")
            continue
            
        my_position = my_rows.index[0] + 1
        
        # Calculate base weight based on position
        normalized_position = my_position / num_teams
        position_weight = 1.0 - 4.0 * (normalized_position - 0.5)**2
        
        # Find clusters in this category
        values = cat_standings[cat].values.reshape(-1, 1)
        
        # Skip if all values are the same
        if np.std(values) == 0:
            weights[cat] = 0.5
            print(f"DEBUG: Category {cat} has no variance, using weight 0.5")
            continue
        
        # Calculate distances and perform hierarchical clustering
        try:
            distances = pdist(values)
            if len(distances) == 0:
                weights[cat] = 0.5
                print(f"DEBUG: Category {cat} has no distances, using weight 0.5")
                continue
                
            Z = linkage(distances, method='ward')
            
            # Determine appropriate number of clusters
            num_clusters = min(5, max(3, int(num_teams / 3)))
            
            # Get cluster assignments
            clusters = fcluster(Z, num_clusters, criterion='maxclust')
            
            # Find which cluster my team is in
            my_cluster = clusters[my_position - 1]
            
            # Count teams in my cluster and adjacent clusters
            my_cluster_size = np.sum(clusters == my_cluster)
            
            # Teams in adjacent positions
            adjacent_positions = 2
            adjacent_teams = sum(1 for i in range(max(0, my_position-adjacent_positions-1), 
                                                min(num_teams, my_position+adjacent_positions)))
            
            # Cluster density weight
            cluster_weight = my_cluster_size / num_teams * 2.0
            
            # Adjacent position weight
            adjacent_weight = adjacent_teams / num_teams * 1.5
            
            # Combine weights
            weight = position_weight * 0.4 + cluster_weight * 0.4 + adjacent_weight * 0.2
            weights[cat] = weight
            
            print(f"DEBUG: Category: {cat}, Position: {my_position}/{num_teams}, Weight: {weight:.3f}")
        except Exception as e:
            print(f"DEBUG: Error calculating weight for {cat}: {str(e)}")
            weights[cat] = 1.0  # Default weight
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {cat: weight * len(weights) / total_weight for cat, weight in weights.items()}
    
    print(f"DEBUG: Final weights: {weights}")
    return weights

def analyze_standings_nfbc(standings_file, my_team=None):
    """
    Analyze NFBC standings to find opportunity clusters and create category weights
    """
    # Load standings
    try:
        standings = pd.read_csv(standings_file)
        print(f"DEBUG: NFBC Standings file loaded successfully with {len(standings)} rows")
        print(f"DEBUG: Columns found: {standings.columns.tolist()}")
    except Exception as e:
        print(f"Error loading NFBC standings file: {str(e)}")
        return None
    
    # Make sure we have a team column - try various common names
    team_cols = ['team', 'team name', 'name', 'teamname', 'tm', 'owner', 'franchise', 'teamname']
    team_col = next((c for c in standings.columns if c.lower() in team_cols or 
                     any(tc in c.lower() for tc in team_cols)), None)
    
    if not team_col:
        print(f"DEBUG: No obvious team column in NFBC standings. Trying all columns...")
        # If no obvious team column, try to find a column with strings that includes the team name
        if my_team:
            for col in standings.columns:
                try:
                    if any(my_team.lower() in str(val).lower() for val in standings[col].dropna().unique()):
                        team_col = col
                        print(f"DEBUG: Found team name in column '{col}'")
                        break
                except:
                    pass
    
    if not team_col:
        print(f"No team name column found in NFBC standings. Available columns: {standings.columns.tolist()}")
        return None
    
    print(f"DEBUG: Using '{team_col}' as team column for NFBC")
    print(f"DEBUG: Teams found: {sorted(standings[team_col].unique().tolist())}")
    
    # Find my team with flexible matching
    my_team_row = None
    
    if my_team:
        # Try exact match
        matching_teams = standings[standings[team_col] == my_team]
        if len(matching_teams) > 0:
            my_team_row = matching_teams.iloc[0]
            print(f"DEBUG: Found exact match for NFBC team '{my_team}'")
        else:
            # Case-insensitive match
            for team in standings[team_col].unique():
                if str(team).lower() == my_team.lower():
                    my_team_row = standings[standings[team_col] == team].iloc[0]
                    print(f"DEBUG: Found case-insensitive match for NFBC team '{my_team}'")
                    break
            
            if my_team_row is None:
                # Fuzzy match
                best_match = None
                best_score = 0
                
                for team in standings[team_col].dropna().unique():
                    score = process.fuzz.ratio(my_team.lower(), str(team).lower())
                    print(f"DEBUG: Fuzzy match score for '{team}': {score}")
                    if score > best_score:
                        best_score = score
                        best_match = team
                
                if best_score > 70:  # Lower threshold for more matches
                    my_team_row = standings[standings[team_col] == best_match].iloc[0]
                    print(f"DEBUG: Using fuzzy-matched team name '{best_match}' (score: {best_score})")
                else:
                    print(f"DEBUG: No close match found for NFBC team '{my_team}'")
    
    if my_team_row is None and len(standings) > 0:
        # If no team specified or found, use the first team in the standings
        my_team_row = standings.iloc[0]
        actual_team_name = my_team_row[team_col]
        print(f"DEBUG: No team specified or found. Using first team in standings: '{actual_team_name}'")
    elif my_team_row is not None:
        actual_team_name = my_team_row[team_col]
        print(f"DEBUG: Using NFBC team '{actual_team_name}' for analysis")
    else:
        print("No teams found in standings data")
        return None
    
    # Get statistical categories (excluding team name, total points, etc.)
    # NFBC specific categories: R, HR, RBI, SB, AVG, K, W, SV, ERA, WHIP
    nfbc_cats = ['R', 'HR', 'RBI', 'SB', 'AVG', 'K', 'W', 'SV', 'ERA', 'WHIP']
    stat_cols = [col for col in standings.columns if col in nfbc_cats]
    
    if not stat_cols:
        # If exact match fails, try case-insensitive
        lower_nfbc_cats = [cat.lower() for cat in nfbc_cats]
        stat_cols = [col for col in standings.columns if col.lower() in lower_nfbc_cats]
    
    if not stat_cols:
        print(f"No relevant NFBC statistical categories found. Available columns: {standings.columns.tolist()}")
        return None
        
    print(f"DEBUG: NFBC statistical categories found: {stat_cols}")
    
    # Calculate weights for each category
    weights = {}
    num_teams = len(standings)
    
    for cat in stat_cols:
        # Determine if lower is better for this category
        lower_better = cat.lower() in ['era', 'whip']
        
        # Get my rank in this category
        if cat not in standings.columns:
            continue
            
        # Sort standings by this category
        cat_standings = standings.sort_values(by=cat, ascending=lower_better)
        
        # Find my row
        my_row = cat_standings[cat_standings[team_col] == actual_team_name]
        if len(my_row) == 0:
            continue
            
        # Get my position (1-indexed)
        my_position = cat_standings.index.get_loc(my_row.index[0]) + 1
        
        # Calculate base weight based on position
        normalized_position = my_position / num_teams
        position_weight = 1.0 - 4.0 * (normalized_position - 0.5)**2
        
        # Find clusters in this category
        values = cat_standings[cat].values.reshape(-1, 1)
        
        # Skip if all values are the same
        if np.std(values) == 0:
            weights[cat] = 0.5
            print(f"DEBUG: NFBC category {cat} has no variance, using weight 0.5")
            continue
        
        # Calculate distances and perform hierarchical clustering
        try:
            distances = pdist(values)
            if len(distances) == 0:
                weights[cat] = 0.5
                print(f"DEBUG: NFBC category {cat} has no distances, using weight 0.5")
                continue
                
            Z = linkage(distances, method='ward')
            
            # Determine appropriate number of clusters
            num_clusters = min(5, max(3, int(num_teams / 3)))
            
            # Get cluster assignments
            clusters = fcluster(Z, num_clusters, criterion='maxclust')
            
            # Find which cluster my team is in
            my_cluster = clusters[my_position - 1]
            
            # Count teams in my cluster and adjacent clusters
            my_cluster_size = np.sum(clusters == my_cluster)
            
            # Teams in adjacent positions
            adjacent_positions = 2
            adjacent_teams = sum(1 for i in range(max(0, my_position-adjacent_positions-1), 
                                                min(num_teams, my_position+adjacent_positions)))
            
            # Cluster density weight
            cluster_weight = my_cluster_size / num_teams * 2.0
            
            # Adjacent position weight
            adjacent_weight = adjacent_teams / num_teams * 1.5
            
            # Combine weights
            weight = position_weight * 0.4 + cluster_weight * 0.4 + adjacent_weight * 0.2
            weights[cat] = weight
            
            print(f"DEBUG: NFBC Category: {cat}, Position: {my_position}/{num_teams}, Weight: {weight:.3f}")
        except Exception as e:
            print(f"DEBUG: Error calculating weight for NFBC category {cat}: {str(e)}")
            weights[cat] = 1.0  # Default weight
    
    # Normalize weights
    total_weight = sum(weights.values())
    if total_weight > 0:
        weights = {cat: weight * len(weights) / total_weight for cat, weight in weights.items()}
    
    print(f"DEBUG: Final NFBC weights: {weights}")
    return weights

def calculate_mash_score(df, category_weights, is_pitcher=False):
    """
    Calculate MASH score based on weighted Z-scores and standings opportunity
    """
    if df is None or len(df) == 0 or not category_weights:
        return df
        
    # Create a copy to avoid modifying the original
    df_mash = df.copy()
    
    # Map fantasy categories to projection categories (expanded with more variations)
    category_map = {
        # Hitting categories
        'AVG': ['AVG', 'BA', 'BATTING AVERAGE'],
        'BA': ['AVG', 'BA', 'BATTING AVERAGE'],
        'BATTING AVERAGE': ['AVG', 'BA', 'BATTING AVERAGE'],
        
        'HR': ['HR', 'HOME RUNS', 'HOMERUNS'],
        'HOME RUNS': ['HR', 'HOME RUNS', 'HOMERUNS'],
        'HOMERUNS': ['HR', 'HOME RUNS', 'HOMERUNS'],
        
        'R': ['R', 'RUNS'],
        'RUNS': ['R', 'RUNS'],
        
        'RBI': ['RBI', 'RUNS BATTED IN'],
        'RUNS BATTED IN': ['RBI', 'RUNS BATTED IN'],
        
        'SB': ['SB', 'STOLEN BASES'],
        'STOLEN BASES': ['SB', 'STOLEN BASES'],
        
        'OPS': ['OPS', 'ON BASE PLUS SLUGGING'],
        'ON BASE PLUS SLUGGING': ['OPS', 'ON BASE PLUS SLUGGING'],
        
        # Pitching categories
        'ERA': ['ERA', 'EARNED RUN AVERAGE'],
        'EARNED RUN AVERAGE': ['ERA', 'EARNED RUN AVERAGE'],
        
        'WHIP': ['WHIP', 'WALKS PLUS HITS PER INNING'],
        
        'W': ['W', 'WINS'],
        'WINS': ['W', 'WINS'],
        
        'SV': ['SV', 'SAVES'],
        'SAVES': ['SV', 'SAVES'],
        
        'K': ['K', 'SO', 'STRIKEOUTS'],
        'SO': ['K', 'SO', 'STRIKEOUTS'],
        'STRIKEOUTS': ['K', 'SO', 'STRIKEOUTS'],
        
        'QS': ['QS', 'QUALITY STARTS'],
        'QUALITY STARTS': ['QS', 'QUALITY STARTS'],
        
        'HLD': ['HLD', 'HOLDS'],
        'HOLDS': ['HLD', 'HOLDS']
    }
    
    # Find Z-score columns
    z_columns = [col for col in df_mash.columns if col.endswith('_z')]
    if not z_columns:
        print("DEBUG: No Z-score columns found")
        return df_mash
        
    # Print debugging info
    print(f"DEBUG: Available category weights for Nuke Laloosh: {list(category_weights.keys())}")
    print(f"DEBUG: Available Z-score columns: {z_columns}")
    
    # Calculate weighted Z-scores for MASH
    mash_components = []
    
    for z_col in z_columns:
        # Extract the base category name
        base_category = z_col[:-2].upper()
        
        # Find if this category has a weight
        weight = None
        for stand_cat in category_weights:
            stand_cat_upper = stand_cat.upper()
            if stand_cat_upper in category_map:
                possible_matches = [match.upper() for match in category_map[stand_cat_upper]]
                if base_category in possible_matches:
                    weight = category_weights[stand_cat]
                    print(f"DEBUG: Matched {base_category} to {stand_cat} with weight {weight}")
                    break
        
        # If no weight found, try simple direct match
        if weight is None and base_category in category_weights:
            weight = category_weights[base_category]
            print(f"DEBUG: Direct match for {base_category} with weight {weight}")
            
        # If still no weight found, use 1.0
        if weight is None:
            print(f"DEBUG: No weight found for {base_category}, using default weight 1.0")
            weight = 1.0
            
        # Create weighted Z-score column
        weighted_col = f"{base_category.lower()}_mash"
        df_mash[weighted_col] = df_mash[z_col] * weight
        mash_components.append(weighted_col)
    
    # Calculate MASH score
    if mash_components:
        df_mash['MASH'] = df_mash[mash_components].sum(axis=1)
        
        # Reorder columns
        cols = list(df_mash.columns)
        cols.remove('MASH')
        
        name_idx = cols.index('Name') if 'Name' in cols else -1
        total_z_idx = cols.index('Total_Z') if 'Total_Z' in cols else -1
        
        if total_z_idx >= 0:
            cols.insert(total_z_idx + 1, 'MASH')
        else:
            cols.insert(name_idx + 1, 'MASH') if name_idx >= 0 else cols.insert(0, 'MASH')
        
        df_mash = df_mash[cols]
        
    return df_mash

def calculate_mash_score_alt(df, category_weights, is_pitcher=False):
    """
    Calculate MASH score based on weighted Z-scores and standings opportunity
    Specifically tailored for Fartolo Cologne league settings
    """
    if df is None or len(df) == 0:
        return df
    
    if not category_weights:
        print("DEBUG: No category weights provided for Fartolo Cologne")
        return df
        
    # Create a copy to avoid modifying the original
    df_mash = df.copy()
    
    # Find Z-score columns
    z_columns = [col for col in df_mash.columns if col.endswith('_z')]
    if not z_columns:
        print("DEBUG: No Z-score columns found for Fartolo Cologne")
        return df_mash
    
    # Print debug information
    print(f"DEBUG: Category weight keys: {list(category_weights.keys())}")
    print(f"DEBUG: Z-score columns: {z_columns}")
    
    # Direct mapping from standings columns to z-score columns
    # This mapping is specifically tailored to your standings file columns
    fartolo_column_map = {
        # Hitting categories
        'HR_z': ['HR'],
        'BB_z': ['BB'],
        'SBN_z': ['SBN'],
        'OBP_z': ['OBP'],
        'SLG_z': ['SLG'],
        
        # Pitching categories
        'K/9_z': ['K'], 
        'K/BB_z': ['K/BB'],
        'ERA_z': ['ERA'],
        'HR/9_z': ['HR.1'], # Maps to HR.1 in standings
        'WHIP_z': ['WHIP']
    }
    
    # Calculate weighted Z-scores for MASH
    mash_components = []
    matched_count = 0
    
    for z_col in z_columns:
        weight = None
        matched_key = None
        
        # Try direct mapping first
        if z_col in fartolo_column_map:
            for possible_key in fartolo_column_map[z_col]:
                if possible_key in category_weights:
                    weight = category_weights[possible_key]
                    matched_key = possible_key
                    print(f"DEBUG: Direct match: {z_col} → {matched_key}, weight: {weight}")
                    matched_count += 1
                    break
        
        # If no direct match, try the base category name without _z
        if weight is None:
            base_category = z_col[:-2]  # Remove _z suffix
            if base_category in category_weights:
                weight = category_weights[base_category]
                matched_key = base_category
                print(f"DEBUG: Base match: {z_col} → {matched_key}, weight: {weight}")
                matched_count += 1
        
        # Special case for HR.1 (HR/9)
        if weight is None and 'HR/9' in z_col.upper():
            if 'HR.1' in category_weights:
                weight = category_weights['HR.1']
                matched_key = 'HR.1'
                print(f"DEBUG: Special case: {z_col} → {matched_key}, weight: {weight}")
                matched_count += 1
        
        # If still no match, use default weight
        if weight is None:
            weight = 1.0
            print(f"DEBUG: No match for {z_col}, using default weight 1.0")
            
        # Create weighted Z-score column
        base_category = z_col[:-2]
        weighted_col = f"{base_category}_mash"
        df_mash[weighted_col] = df_mash[z_col] * weight
        mash_components.append(weighted_col)
    
    print(f"DEBUG: Matched {matched_count}/{len(z_columns)} categories with weights")
    
    # Calculate MASH score
    if mash_components:
        df_mash['MASH'] = df_mash[mash_components].sum(axis=1)
        
        # Reorder columns
        cols = list(df_mash.columns)
        cols.remove('MASH')
        
        name_idx = cols.index('Name') if 'Name' in cols else -1
        total_z_idx = cols.index('Total_Z') if 'Total_Z' in cols else -1
        
        if total_z_idx >= 0:
            cols.insert(total_z_idx + 1, 'MASH')
        else:
            cols.insert(name_idx + 1, 'MASH') if name_idx >= 0 else cols.insert(0, 'MASH')
        
        df_mash = df_mash[cols]
    
    return df_mash

def load_and_aggregate_ros():
    # Load all CSVs from repo
    atc = pd.read_csv('ATC.csv')
    oopsy = pd.read_csv('Oopsy.csv')
    bat_x = pd.read_csv('The Bat X.csv')
    atc_p = pd.read_csv('ATC PItching.csv')
    oopsy_p = pd.read_csv('Oopsy Pitching.csv')
    bat_p = pd.read_csv('The Bat Pitching.csv')
    profiles = pd.read_csv('Combined_Sorted_Profiles.csv')
    botstf = pd.read_csv('BotStf.csv')
    location = pd.read_csv('Location+.csv')
    idmap = pd.read_csv('SFBB Player ID Map - PLAYERIDMAP(1).csv')

    # Normalize IDs
    for df in [atc, oopsy, bat_x, atc_p, oopsy_p, bat_p, botstf, location, idmap]:
        if 'PlayerId' in df.columns:
            df['IDFANGRAPHS'] = df['PlayerId'].astype(str)
        if 'IDFANGRAPHS' in df.columns:
            df['IDFANGRAPHS'] = df['IDFANGRAPHS'].astype(str)
    if 'IDFANGRAPHS' in idmap.columns:
        idmap['IDFANGRAPHS'] = idmap['IDFANGRAPHS'].astype(str)

    # Normalize hitters to 600 PA
    for df in [atc, oopsy, bat_x]:
        if 'PA' in df.columns:
            for stat in ['AB', 'H', '2B', '3B', 'HR', 'R', 'RBI', 'BB', 'SO', 'SB', 'CS']:
                if stat in df.columns:
                    df[stat] = df.apply(lambda row: row[stat] * 600 / row['PA'] if row['PA'] > 0 else row[stat], axis=1)
    # Normalize pitchers to 180 IP
    for df in [atc_p, oopsy_p, bat_p]:
        if 'IP' in df.columns:
            for stat in ['W', 'L', 'SV', 'HLD', 'H', 'ER', 'HR', 'BB', 'SO']:
                if stat in df.columns:
                    df[stat] = df.apply(lambda row: row[stat] * 180 / row['IP'] if row['IP'] > 0 else row[stat], axis=1)

    # Aggregate hitters
    hitter_dfs = []
    for name, df in [('ATC', atc), ('Oopsy', oopsy), ('The Bat X', bat_x)]:
        df_ = df.copy()
        df_['System'] = name
        df_ = df_.rename(columns={"Name":"Name", "Team":"Team"})
        cols = ['Name', 'Team', 'IDFANGRAPHS'] + [c for c in df_.columns if df_[c].dtype in [np.float64, np.int64]]
        cols = [c for c in cols if c in df_.columns]
        hitter_dfs.append(df_[cols])
    all_hitters = pd.concat(hitter_dfs, ignore_index=True)
    numeric_cols = all_hitters.select_dtypes(include=[np.number]).columns
    agg_hitters = all_hitters.groupby(['Name', 'Team', 'IDFANGRAPHS'], dropna=False).agg({col:'mean' for col in numeric_cols})
    agg_hitters = agg_hitters.reset_index()

    # Add SEAGER and 90th Pctile EV
    seager_col = next((c for c in profiles.columns if c.upper() == "SEAGER"), None)
    ev_col = next((c for c in profiles.columns if "90TH" in c.upper() and "EV" in c.upper()), None)
    if seager_col and ev_col:
        profile_dict = {f"{row['Name']} {row['Team']}": (row[seager_col], row[ev_col]) for _, row in profiles.iterrows()}
        seager_avg = profiles[seager_col].mean()
        ev_avg = profiles[ev_col].mean()
        def get_feats(row):
            key = f"{row['Name']} {row['Team']}"
            if key in profile_dict:
                return profile_dict[key]
            match = process.extractOne(key, profile_dict.keys())
            if match and match[1] >= 85:
                return profile_dict[match[0]]
            return (seager_avg, ev_avg)
        agg_hitters[[seager_col, ev_col]] = agg_hitters.apply(lambda row: pd.Series(get_feats(row)), axis=1)

    # Aggregate pitchers
    pitcher_dfs = []
    for name, df in [('ATC', atc_p), ('Oopsy', oopsy_p), ('The Bat', bat_p)]:
        df_ = df.copy()
        df_['System'] = name
        df_ = df_.rename(columns={"Name":"Name", "Team":"Team"})
        cols = ['Name', 'Team', 'IDFANGRAPHS'] + [c for c in df_.columns if df_[c].dtype in [np.float64, np.int64]]
        cols = [c for c in cols if c in df_.columns]
        pitcher_dfs.append(df_[cols])
    all_pitchers = pd.concat(pitcher_dfs, ignore_index=True)
    numeric_cols = all_pitchers.select_dtypes(include=[np.number]).columns
    agg_pitchers = all_pitchers.groupby(['Name', 'Team', 'IDFANGRAPHS'], dropna=False).agg({col:'mean' for col in numeric_cols})
    agg_pitchers = agg_pitchers.reset_index()

    # Add BotStf and Location+
    if 'IDFANGRAPHS' in botstf.columns and 'botStf' in botstf.columns:
        agg_pitchers = agg_pitchers.merge(botstf[['IDFANGRAPHS', 'botStf']], on='IDFANGRAPHS', how='left')
    if 'IDFANGRAPHS' in location.columns and 'Location+' in location.columns:
        agg_pitchers = agg_pitchers.merge(location[['IDFANGRAPHS', 'Location+']], on='IDFANGRAPHS', how='left')
    if 'IDFANGRAPHS' in agg_hitters.columns and 'IDFANGRAPHS' in idmap.columns:
        agg_hitters = agg_hitters.merge(idmap, on='IDFANGRAPHS', how='left')
    if 'IDFANGRAPHS' in agg_pitchers.columns and 'IDFANGRAPHS' in idmap.columns:
        agg_pitchers = agg_pitchers.merge(idmap, on='IDFANGRAPHS', how='left')

    # Filter out problematic players
    agg_hitters = filter_problematic_players(agg_hitters)
    agg_pitchers = filter_problematic_players(agg_pitchers)
    
    # Calculate additional advanced stats for alt league
    # SBN (Net Stolen Bases)
    if 'SBN' not in agg_hitters.columns and 'SB' in agg_hitters.columns:
        if 'CS' in agg_hitters.columns:
            agg_hitters['SBN'] = agg_hitters['SB'] - agg_hitters['CS']
        else:
            agg_hitters['SBN'] = agg_hitters['SB']
        
    # K/9 (Strikeouts per 9 innings)
    if 'K/9' not in agg_pitchers.columns and 'K' in agg_pitchers.columns and 'IP' in agg_pitchers.columns:
        agg_pitchers['K/9'] = 9.0 * agg_pitchers['K'] / agg_pitchers['IP']
        
    # K/BB (Strikeout to Walk ratio)
    if 'K/BB' not in agg_pitchers.columns and 'K' in agg_pitchers.columns and 'BB' in agg_pitchers.columns:
        agg_pitchers['K/BB'] = agg_pitchers['K'] / agg_pitchers['BB'].replace(0, 0.01)
        
    # HR/9 (Home Runs per 9 innings)
    if 'HR/9' not in agg_pitchers.columns and 'HR' in agg_pitchers.columns and 'IP' in agg_pitchers.columns:
        agg_pitchers['HR/9'] = 9.0 * agg_pitchers['HR'] / agg_pitchers['IP']
    
    return agg_hitters, agg_pitchers

def find_matching_columns(df1, df2):
    """Find matching column pairs between two dataframes for player ID matching"""
    matches = []
    # Common player ID columns to check
    id_columns = ["IDFANGRAPHS", "PlayerId", "playerid", "MLBID", "MLBAMID", 
                  "ESPNID", "CBSID", "YAHOOID", "FANTRAXID", "OTTONEU_ID", "NFBCID"]
    
    # Name columns to check
    name_columns = ["Name", "PLAYERNAME", "LASTNAME", "FANGRAPHSNAME", "Player", "Players"]
    
    # First check if there are exact ID column matches
    for col1 in df1.columns:
        for match in id_columns:
            if match.upper() == col1.upper():
                for col2 in df2.columns:
                    if match.upper() == col2.upper():
                        matches.append((col1, col2))
    
    # If no ID matches, try name columns
    if not matches:
        for col1 in df1.columns:
            for match in name_columns:
                if match.upper() == col1.upper():
                    for col2 in df2.columns:
                        if match.upper() == col2.upper():
                            matches.append((col1, col2))
                            
    # If still no matches, try cross-mapping common combinations
    if not matches:
        cross_maps = [
            ("PLAYERNAME", "Player"),
            ("Name", "Player"),
            ("FANTRAXID", "ID"),
            ("FANTRAXNAME", "Player"),
            ("IDFANGRAPHS", "ID"),
            ("Name", "Players"),
            ("NFBCID", "id")
        ]
        for col1, col2 in cross_maps:
            if col1 in df1.columns and col2 in df2.columns:
                matches.append((col1, col2))
    
    return matches

def filter_players(projections_df, league_df, status_values):
    """Filter projections to only players with specified status values"""
    status_col = next((c for c in league_df.columns if c.lower() == "status"), None)
    if not status_col:
        return None, "No 'Status' column found in league file."
    
    if not status_values:  # For lineup optimizer with no specific status filter
        filtered_league = league_df
    else:
        filtered_league = league_df[league_df[status_col].astype(str).str.upper().isin([s.upper() for s in status_values])]
    
    # Find matching columns between the two dataframes
    matches = find_matching_columns(projections_df, filtered_league)
    
    if not matches:
        return None, (
            f"No common player identifier found. "
            f"Projection columns: {projections_df.columns.tolist()[:10]}... | "
            f"League file columns: {filtered_league.columns.tolist()[:10]}..."
        )
    
    # Try each match pair until we get results
    for proj_col, league_col in matches:
        filtered_proj = pd.merge(
            projections_df, 
            filtered_league, 
            left_on=proj_col, 
            right_on=league_col, 
            how='inner'
        )
        if len(filtered_proj) > 0:
            # Add the status column to the filtered projections
            filtered_proj['Status'] = filtered_proj[status_col]
            # Remove duplicates by Name
            name_col = next((c for c in filtered_proj.columns if c.lower() == 'name'), None)
            if name_col:
                filtered_proj = filtered_proj.drop_duplicates(subset=[name_col])
            return filtered_proj, None
    
    return None, f"No players found with status {', '.join(status_values) if status_values else 'any'}"

def calculate_zscores_hitters(df, reference_df=None, top_n=300):
    """
    Calculate Z-scores for hitter categories and add total Z-score
    Uses a reference dataset or top_n players for calculating mean and std
    """
    if df is None or len(df) == 0:
        return df
    
    # Define categories and weights
    hitting_cats = {
        'R': {'higher_better': True, 'weight': 1.0},
        'HR': {'higher_better': True, 'weight': 1.0},
        'RBI': {'higher_better': True, 'weight': 1.0},
        'SB': {'higher_better': True, 'weight': 1.0},
        'AVG': {'higher_better': True, 'weight': 1.0},
        'OPS': {'higher_better': True, 'weight': 1.0}
    }
    
    # Find supplemental stats
    seager_col = next((c for c in df.columns if c.upper() == "SEAGER"), None)
    ev_col = next((c for c in df.columns if "90TH" in c.upper() and "EV" in c.upper()), None)
    
    if seager_col:
        hitting_cats[seager_col] = {'higher_better': True, 'weight': 0.25}
    if ev_col:
        hitting_cats[ev_col] = {'higher_better': True, 'weight': 0.25}
    
    # Create a copy
    df_z = df.copy()
    
    # If no reference dataframe is provided, use the input dataframe
    if reference_df is None:
        reference_df = df.copy()
    
    # Fill missing values with category averages before Z-score calculations
    for cat in hitting_cats:
        if cat in df_z.columns and df_z[cat].isna().any():
            category_mean = reference_df[cat].mean() if cat in reference_df.columns else df_z[cat].mean()
            # Add small random variation (±1%) to avoid identical values for missing data
            import random
            df_z[cat] = df_z[cat].apply(
                lambda x: category_mean * (1 + random.uniform(-0.01, 0.01)) if pd.isna(x) else x
            )
    
    # Filter out bottom performers 
    # Keep only top 80% of players for reference calculations to avoid skewing
    if len(reference_df) > top_n*1.2:  # Only if we have sufficient data
        category_references = {}
        for cat in hitting_cats:
            if cat in reference_df.columns:
                # Sort by category value
                is_higher_better = hitting_cats[cat]['higher_better']
                sorted_ref = reference_df.sort_values(cat, ascending=not is_higher_better)
                # Take top N rows and exclude bottom 20%
                cut_off = int(min(top_n, len(sorted_ref) * 0.8))
                category_references[cat] = sorted_ref.head(cut_off)
    else:
        # Use the whole reference dataset for all categories
        category_references = {cat: reference_df for cat in hitting_cats if cat in reference_df.columns}
    
    # Calculate z-scores for each category
    z_columns = []
    for cat, props in hitting_cats.items():
        if cat in df.columns and cat in category_references:
            # Skip if column is all NaN
            if df[cat].isna().all() or category_references[cat][cat].isna().all():
                continue
                
            col_z = f"{cat}_z"
            mean = category_references[cat][cat].mean()
            std = category_references[cat][cat].std()
            
            # Avoid division by zero
            if std == 0:
                df_z[col_z] = 0
            else:
                # Calculate z-score based on higher/lower better
                if props['higher_better']:
                    df_z[col_z] = (df[cat] - mean) / std
                else:
                    df_z[col_z] = (mean - df[cat]) / std
                
                # Cap z-scores at +/- 1.5
                df_z[col_z] = df_z[col_z].clip(-1.5, 1.5)
                    
                # Apply category weight
                df_z[col_z] = df_z[col_z] * props['weight']
            
            z_columns.append(col_z)
    
    # Calculate total z-score
    if z_columns:
        df_z['Total_Z'] = df_z[z_columns].sum(axis=1)
        
        # Reorder columns
        cols = list(df_z.columns)
        name_idx = cols.index('Name') if 'Name' in cols else -1
        if name_idx >= 0:
            cols.remove('Total_Z')
            cols.insert(name_idx + 1, 'Total_Z')
            df_z = df_z[cols]
    
    return df_z

def calculate_zscores_hitters_alt(df, reference_df=None, top_n=300):
    """
    Calculate Z-scores for hitter categories and add total Z-score
    Uses a reference dataset or top_n players for calculating mean and std
    Adapted for HR, BB, SBN, OBP, SLG categories
    """
    if df is None or len(df) == 0:
        return df
    
    # Define categories and weights - MODIFIED FOR NEW LEAGUE
    hitting_cats = {
        'HR': {'higher_better': True, 'weight': 1.0},
        'BB': {'higher_better': True, 'weight': 1.0},
        'SBN': {'higher_better': True, 'weight': 1.0},
        'OBP': {'higher_better': True, 'weight': 1.0},
        'SLG': {'higher_better': True, 'weight': 1.0}
    }
    
    # Find supplemental stats (still useful even in different format)
    seager_col = next((c for c in df.columns if c.upper() == "SEAGER"), None)
    ev_col = next((c for c in df.columns if "90TH" in c.upper() and "EV" in c.upper()), None)
    
    if seager_col:
        hitting_cats[seager_col] = {'higher_better': True, 'weight': 0.25}
    if ev_col:
        hitting_cats[ev_col] = {'higher_better': True, 'weight': 0.25}
    
    # Create a copy
    df_z = df.copy()
    
    # If no reference dataframe is provided, use the input dataframe
    if reference_df is None:
        reference_df = df.copy()
    
    # Calculate SBN if it's not present but SB and CS are
    if 'SBN' not in df_z.columns and 'SB' in df_z.columns:
        if 'CS' in df_z.columns:
            df_z['SBN'] = df_z['SB'] - df_z['CS']
        else:
            df_z['SBN'] = df_z['SB']  # If CS not available, use SB as approximation
            
    # Fill missing values with category averages before Z-score calculations
    for cat in hitting_cats:
        if cat in df_z.columns and df_z[cat].isna().any():
            category_mean = reference_df[cat].mean() if cat in reference_df.columns else df_z[cat].mean()
            # Add small random variation (±1%) to avoid identical values for missing data
            import random
            df_z[cat] = df_z[cat].apply(
                lambda x: category_mean * (1 + random.uniform(-0.01, 0.01)) if pd.isna(x) else x
            )
    
    # Filter out bottom performers for reference calculations to avoid skewing
    if len(reference_df) > top_n*1.2:  # Only if we have sufficient data
        category_references = {}
        for cat in hitting_cats:
            if cat in reference_df.columns:
                # Sort by category value
                is_higher_better = hitting_cats[cat]['higher_better']
                sorted_ref = reference_df.sort_values(cat, ascending=not is_higher_better)
                # Take top N rows and exclude bottom 20%
                cut_off = int(min(top_n, len(sorted_ref) * 0.8))
                category_references[cat] = sorted_ref.head(cut_off)
    else:
        # Use the whole reference dataset for all categories
        category_references = {cat: reference_df for cat in hitting_cats if cat in reference_df.columns}
    
    # Calculate z-scores for each category
    z_columns = []
    for cat, props in hitting_cats.items():
        if cat in df_z.columns and cat in category_references:
            # Skip if column is all NaN
            if df_z[cat].isna().all() or category_references[cat][cat].isna().all():
                continue
                
            col_z = f"{cat}_z"
            mean = category_references[cat][cat].mean()
            std = category_references[cat][cat].std()
            
            # Avoid division by zero
            if std == 0:
                df_z[col_z] = 0
            else:
                # Calculate z-score based on higher/lower better
                if props['higher_better']:
                    df_z[col_z] = (df_z[cat] - mean) / std
                else:
                    df_z[col_z] = (mean - df_z[cat]) / std
                
                # Cap z-scores at +/- 1.5
                df_z[col_z] = df_z[col_z].clip(-1.5, 1.5)
                    
                # Apply category weight
                df_z[col_z] = df_z[col_z] * props['weight']
            
            z_columns.append(col_z)
    
    # Calculate total z-score
    if z_columns:
        df_z['Total_Z'] = df_z[z_columns].sum(axis=1)
        
        # Reorder columns
        cols = list(df_z.columns)
        name_idx = cols.index('Name') if 'Name' in cols else -1
        if name_idx >= 0:
            cols.remove('Total_Z')
            cols.insert(name_idx + 1, 'Total_Z')
            df_z = df_z[cols]
    
    return df_z

def calculate_zscores_nfbc_hitters(df, reference_df=None, top_n=300):
    """
    Calculate Z-scores for NFBC hitter categories and add total Z-score
    Uses a reference dataset or top_n players for calculating mean and std
    """
    if df is None or len(df) == 0:
        return df
    
    # Define NFBC hitting categories and weights
    hitting_cats = {
        'R': {'higher_better': True, 'weight': 1.0},
        'HR': {'higher_better': True, 'weight': 1.0},
        'RBI': {'higher_better': True, 'weight': 1.0},
        'SB': {'higher_better': True, 'weight': 1.0},
        'AVG': {'higher_better': True, 'weight': 1.0}
    }
    
    # Create a copy
    df_z = df.copy()
    
    # If no reference dataframe is provided, use the input dataframe
    if reference_df is None:
        reference_df = df.copy()
    
    # Fill missing values with category averages before Z-score calculations
    for cat in hitting_cats:
        if cat in df_z.columns and df_z[cat].isna().any():
            category_mean = reference_df[cat].mean() if cat in reference_df.columns else df_z[cat].mean()
            # Add small random variation (±1%) to avoid identical values for missing data
            import random
            df_z[cat] = df_z[cat].apply(
                lambda x: category_mean * (1 + random.uniform(-0.01, 0.01)) if pd.isna(x) else x
            )
    
    # Filter out bottom performers for reference calculations
    if len(reference_df) > top_n*1.2:  # Only if we have sufficient data
        category_references = {}
        for cat in hitting_cats:
            if cat in reference_df.columns:
                # Sort by category value
                is_higher_better = hitting_cats[cat]['higher_better']
                sorted_ref = reference_df.sort_values(cat, ascending=not is_higher_better)
                # Take top N rows and exclude bottom 20%
                cut_off = int(min(top_n, len(sorted_ref) * 0.8))
                category_references[cat] = sorted_ref.head(cut_off)
    else:
        # Use the whole reference dataset for all categories
        category_references = {cat: reference_df for cat in hitting_cats if cat in reference_df.columns}
    
    # Calculate z-scores for each category
    z_columns = []
    for cat, props in hitting_cats.items():
        if cat in df_z.columns and cat in category_references:
            # Skip if column is all NaN
            if df_z[cat].isna().all() or category_references[cat][cat].isna().all():
                continue
                
            col_z = f"{cat}_z"
            mean = category_references[cat][cat].mean()
            std = category_references[cat][cat].std()
            
            # Avoid division by zero
            if std == 0:
                df_z[col_z] = 0
            else:
                # Calculate z-score based on higher/lower better
                if props['higher_better']:
                    df_z[col_z] = (df_z[cat] - mean) / std
                else:
                    df_z[col_z] = (mean - df_z[cat]) / std
                
                # Cap z-scores at +/- 1.5
                df_z[col_z] = df_z[col_z].clip(-1.5, 1.5)
                    
                # Apply category weight
                df_z[col_z] = df_z[col_z] * props['weight']
            
            z_columns.append(col_z)
    
    # Calculate total z-score
    if z_columns:
        df_z['Total_Z'] = df_z[z_columns].sum(axis=1)
        
        # Reorder columns
        cols = list(df_z.columns)
        name_idx = cols.index('Name') if 'Name' in cols else -1
        if name_idx >= 0:
            cols.remove('Total_Z')
            cols.insert(name_idx + 1, 'Total_Z')
            df_z = df_z[cols]
    
    return df_z

def calculate_zscores_pitchers(df, reference_df=None, top_n=300):
    """
    Calculate Z-scores for pitcher categories and add total Z-score
    Uses a reference dataset or top_n players for calculating mean and std
    """
    if df is None or len(df) == 0:
        return df
    
    # Define categories and weights
    pitching_cats = {
        'K': {'higher_better': True, 'weight': 1.0},
        'ERA': {'higher_better': False, 'weight': 1.0},
        'WHIP': {'higher_better': False, 'weight': 1.0},
        'W': {'higher_better': True, 'weight': 1.0},
        'SV': {'higher_better': True, 'weight': 1.0},
        'QS': {'higher_better': True, 'weight': 1.0},
        'botStf': {'higher_better': True, 'weight': 0.25},
        'Location+': {'higher_better': True, 'weight': 0.25}
    }
    
    # Create a copy
    df_z = df.copy()
    
    # Calculate W+QS if both exist
    if 'W' in df.columns and 'QS' in df.columns:
        df_z['W+QS'] = df['W'] + df['QS']
        pitching_cats['W+QS'] = {'higher_better': True, 'weight': 1.0}
    
    # Calculate K/BB if K and BB exist
    if 'K' in df.columns and 'BB' in df.columns:
        df_z['K/BB'] = df['K'] / df['BB'].replace(0, np.nan)
        df_z['K/BB'] = df_z['K/BB'].fillna(df_z['K/BB'].mean())  # Handle division by zero
        pitching_cats['K/BB'] = {'higher_better': True, 'weight': 1.0}
    elif 'K/BB' in df.columns:
        pitching_cats['K/BB'] = {'higher_better': True, 'weight': 1.0}
        
    # Fill missing values with category averages before Z-score calculations
    for cat in pitching_cats:
        if cat in df_z.columns and df_z[cat].isna().any():
            category_mean = reference_df[cat].mean() if (reference_df is not None 
                                                        and cat in reference_df.columns) else df_z[cat].mean()
            # Add small random variation (±1%) to avoid identical values for missing data
            import random
            df_z[cat] = df_z[cat].apply(
                lambda x: category_mean * (1 + random.uniform(-0.01, 0.01)) if pd.isna(x) else x
            )
    
    # If no reference dataframe is provided, use the input dataframe
    if reference_df is None:
        # For ERA and WHIP, exclude extreme outliers
        reference_df = df.copy()
        if 'ERA' in reference_df.columns:
            reference_df['ERA'] = reference_df['ERA'].clip(lower=1.0, upper=7.0)
        if 'WHIP' in reference_df.columns:
            reference_df['WHIP'] = reference_df['WHIP'].clip(lower=0.8, upper=2.0)
    
    # Filter out bottom performers
    # Keep only top 80% of players for reference calculations to avoid skewing
    if len(reference_df) > top_n*1.2:  # Only if we have sufficient data
        category_references = {}
        for cat in pitching_cats:
            if cat in reference_df.columns:
                # Sort by category value
                is_higher_better = pitching_cats[cat]['higher_better']
                sorted_ref = reference_df.sort_values(cat, ascending=not is_higher_better)
                # Take top N rows and exclude bottom 20%
                cut_off = int(min(top_n, len(sorted_ref) * 0.8))
                category_references[cat] = sorted_ref.head(cut_off)
    else:
        # Use the whole reference dataset for all categories
        category_references = {cat: reference_df for cat in pitching_cats if cat in reference_df.columns}
    
    # Calculate z-scores for each category
    z_columns = []
    for cat, props in pitching_cats.items():
        if cat in df_z.columns and cat in category_references:
            # Skip if column is all NaN
            if df_z[cat].isna().all() or category_references[cat][cat].isna().all():
                continue
                
            col_z = f"{cat}_z"
            mean = category_references[cat][cat].mean()
            std = category_references[cat][cat].std()
            
            # Avoid division by zero
            if std == 0:
                df_z[col_z] = 0
            else:
                # Calculate z-score based on higher/lower better
                if props['higher_better']:
                    df_z[col_z] = (df_z[cat] - mean) / std
                else:
                    df_z[col_z] = (mean - df_z[cat]) / std
                
                                # Cap z-scores at +/- 1.5
                df_z[col_z] = df_z[col_z].clip(-1.5, 1.5)
                    
                # Apply category weight
                df_z[col_z] = df_z[col_z] * props['weight']
            
            z_columns.append(col_z)
    
    # Calculate total z-score
    if z_columns:
        df_z['Total_Z'] = df_z[z_columns].sum(axis=1)
        
        # Reorder columns
        cols = list(df_z.columns)
        name_idx = cols.index('Name') if 'Name' in cols else -1
        if name_idx >= 0:
            cols.remove('Total_Z')
            cols.insert(name_idx + 1, 'Total_Z')
            df_z = df_z[cols]
    
    return df_z

def calculate_zscores_pitchers_alt(df, reference_df=None, top_n=300):
    """
    Calculate Z-scores for pitcher categories and add total Z-score
    Uses a reference dataset or top_n players for calculating mean and std
    Adapted for K/9, K/BB, ERA, HR/9, WHIP categories
    """
    if df is None or len(df) == 0:
        return df
    
    # Define categories and weights - MODIFIED FOR NEW LEAGUE
    pitching_cats = {
        'K/9': {'higher_better': True, 'weight': 1.0},
        'K/BB': {'higher_better': True, 'weight': 1.0},
        'ERA': {'higher_better': False, 'weight': 1.0},
        'HR/9': {'higher_better': False, 'weight': 1.0},
        'WHIP': {'higher_better': False, 'weight': 1.0},
        'botStf': {'higher_better': True, 'weight': 0.25},
        'Location+': {'higher_better': True, 'weight': 0.25}
    }
    
    # Create a copy
    df_z = df.copy()
    
    # Calculate needed stats if not present
    if 'K/9' not in df_z.columns and 'K' in df_z.columns and 'IP' in df_z.columns:
        df_z['K/9'] = 9.0 * df_z['K'] / df_z['IP'].replace(0, np.nan)
        df_z['K/9'] = df_z['K/9'].fillna(df_z['K/9'].median())
        
    if 'K/BB' not in df_z.columns and 'K' in df_z.columns and 'BB' in df_z.columns:
        df_z['K/BB'] = df_z['K'] / df_z['BB'].replace(0, np.nan)
        df_z['K/BB'] = df_z['K/BB'].fillna(df_z['K/BB'].median())
    
    if 'HR/9' not in df_z.columns and 'HR' in df_z.columns and 'IP' in df_z.columns:
        df_z['HR/9'] = 9.0 * df_z['HR'] / df_z['IP'].replace(0, np.nan)
        df_z['HR/9'] = df_z['HR/9'].fillna(df_z['HR/9'].median())
    
    # Fill missing values with category averages before Z-score calculations
    for cat in pitching_cats:
        if cat in df_z.columns and df_z[cat].isna().any():
            category_mean = reference_df[cat].mean() if (reference_df is not None 
                                                        and cat in reference_df.columns) else df_z[cat].mean()
            # Add small random variation (±1%) to avoid identical values for missing data
            import random
            df_z[cat] = df_z[cat].apply(
                lambda x: category_mean * (1 + random.uniform(-0.01, 0.01)) if pd.isna(x) else x
            )
    
    # If no reference dataframe is provided, use the input dataframe
    if reference_df is None:
        # For ERA and WHIP, exclude extreme outliers
        reference_df = df.copy()
        if 'ERA' in reference_df.columns:
            reference_df['ERA'] = reference_df['ERA'].clip(lower=1.0, upper=7.0)
        if 'WHIP' in reference_df.columns:
            reference_df['WHIP'] = reference_df['WHIP'].clip(lower=0.8, upper=2.0)
        if 'HR/9' in reference_df.columns:
            reference_df['HR/9'] = reference_df['HR/9'].clip(lower=0, upper=3.0)
    
    # Filter out bottom performers
    if len(reference_df) > top_n*1.2:  # Only if we have sufficient data
        category_references = {}
        for cat in pitching_cats:
            if cat in reference_df.columns:
                # Sort by category value
                is_higher_better = pitching_cats[cat]['higher_better']
                sorted_ref = reference_df.sort_values(cat, ascending=not is_higher_better)
                # Take top N rows and exclude bottom 20%
                cut_off = int(min(top_n, len(sorted_ref) * 0.8))
                category_references[cat] = sorted_ref.head(cut_off)
    else:
        # Use the whole reference dataset for all categories
        category_references = {cat: reference_df for cat in pitching_cats if cat in reference_df.columns}
    
    # Calculate z-scores for each category
    z_columns = []
    for cat, props in pitching_cats.items():
        if cat in df_z.columns and cat in category_references:
            # Skip if column is all NaN
            if df_z[cat].isna().all() or category_references[cat][cat].isna().all():
                continue
                
            col_z = f"{cat}_z"
            mean = category_references[cat][cat].mean()
            std = category_references[cat][cat].std()
            
            # Avoid division by zero
            if std == 0:
                df_z[col_z] = 0
            else:
                # Calculate z-score based on higher/lower better
                if props['higher_better']:
                    df_z[col_z] = (df_z[cat] - mean) / std
                else:
                    df_z[col_z] = (mean - df_z[cat]) / std
                
                # Cap z-scores at +/- 1.5
                df_z[col_z] = df_z[col_z].clip(-1.5, 1.5)
                    
                # Apply category weight
                df_z[col_z] = df_z[col_z] * props['weight']
            
            z_columns.append(col_z)
    
    # Calculate total z-score
    if z_columns:
        df_z['Total_Z'] = df_z[z_columns].sum(axis=1)
        
        # Reorder columns
        cols = list(df_z.columns)
        name_idx = cols.index('Name') if 'Name' in cols else -1
        if name_idx >= 0:
            cols.remove('Total_Z')
            cols.insert(name_idx + 1, 'Total_Z')
            df_z = df_z[cols]
    
    return df_z

def calculate_zscores_nfbc_pitchers(df, reference_df=None, top_n=300):
    """
    Calculate Z-scores for NFBC pitcher categories and add total Z-score
    Uses a reference dataset or top_n players for calculating mean and std
    """
    if df is None or len(df) == 0:
        return df
    
    # Define NFBC pitching categories and weights
    pitching_cats = {
        'K': {'higher_better': True, 'weight': 1.0},
        'ERA': {'higher_better': False, 'weight': 1.0},
        'WHIP': {'higher_better': False, 'weight': 1.0},
        'W': {'higher_better': True, 'weight': 1.0},
        'SV': {'higher_better': True, 'weight': 1.0}
    }
    
    # Create a copy
    df_z = df.copy()
    
    # Fill missing values with category averages before Z-score calculations
    for cat in pitching_cats:
        if cat in df_z.columns and df_z[cat].isna().any():
            category_mean = reference_df[cat].mean() if (reference_df is not None 
                                                        and cat in reference_df.columns) else df_z[cat].mean()
            # Add small random variation (±1%) to avoid identical values for missing data
            import random
            df_z[cat] = df_z[cat].apply(
                lambda x: category_mean * (1 + random.uniform(-0.01, 0.01)) if pd.isna(x) else x
            )
    
    # If no reference dataframe is provided, use the input dataframe
    if reference_df is None:
        # For ERA and WHIP, exclude extreme outliers
        reference_df = df.copy()
        if 'ERA' in reference_df.columns:
            reference_df['ERA'] = reference_df['ERA'].clip(lower=1.0, upper=7.0)
        if 'WHIP' in reference_df.columns:
            reference_df['WHIP'] = reference_df['WHIP'].clip(lower=0.8, upper=2.0)
    
    # Filter out bottom performers
    if len(reference_df) > top_n*1.2:  # Only if we have sufficient data
        category_references = {}
        for cat in pitching_cats:
            if cat in reference_df.columns:
                # Sort by category value
                is_higher_better = pitching_cats[cat]['higher_better']
                sorted_ref = reference_df.sort_values(cat, ascending=not is_higher_better)
                # Take top N rows and exclude bottom 20%
                cut_off = int(min(top_n, len(sorted_ref) * 0.8))
                category_references[cat] = sorted_ref.head(cut_off)
    else:
        # Use the whole reference dataset for all categories
        category_references = {cat: reference_df for cat in pitching_cats if cat in reference_df.columns}
    
    # Calculate z-scores for each category
    z_columns = []
    for cat, props in pitching_cats.items():
        if cat in df_z.columns and cat in category_references:
            # Skip if column is all NaN
            if df_z[cat].isna().all() or category_references[cat][cat].isna().all():
                continue
                
            col_z = f"{cat}_z"
            mean = category_references[cat][cat].mean()
            std = category_references[cat][cat].std()
            
            # Avoid division by zero
            if std == 0:
                df_z[col_z] = 0
            else:
                # Calculate z-score based on higher/lower better
                if props['higher_better']:
                    df_z[col_z] = (df_z[cat] - mean) / std
                else:
                    df_z[col_z] = (mean - df_z[cat]) / std
                
                # Cap z-scores at +/- 1.5
                df_z[col_z] = df_z[col_z].clip(-1.5, 1.5)
                    
                # Apply category weight
                df_z[col_z] = df_z[col_z] * props['weight']
            
            z_columns.append(col_z)
    
    # Calculate total z-score
    if z_columns:
        df_z['Total_Z'] = df_z[z_columns].sum(axis=1)
        
        # Reorder columns
        cols = list(df_z.columns)
        name_idx = cols.index('Name') if 'Name' in cols else -1
        if name_idx >= 0:
            cols.remove('Total_Z')
            cols.insert(name_idx + 1, 'Total_Z')
            df_z = df_z[cols]
    
    return df_z

def format_dataframe_for_display(df, display_cols, is_pitcher=False):
    """Format dataframe for display, including rounding and column selection"""
    if df is None or len(df) == 0:
        return pd.DataFrame()
    
    df_display = df.copy()
    
    # Calculate K/9 and BB/9 for pitchers if not present
    if is_pitcher:
        if "K/9" not in df.columns and "SO" in df.columns and "IP" in df.columns:
            df_display["K/9"] = 9 * df["SO"] / df["IP"].replace(0, np.nan)
        if "BB/9" not in df.columns and "BB" in df.columns and "IP" in df.columns:
            df_display["BB/9"] = 9 * df["BB"] / df["IP"].replace(0, np.nan)
    
    # Round numeric columns
    for col in df_display.select_dtypes(include=[np.number]).columns:
        if col in df_display.columns:
            if col in ["AVG", "OBP", "SLG", "OPS", "ERA", "WHIP"]:
                df_display[col] = df_display[col].round(3)
            elif col in ["K/9", "BB/9", "K/BB", "HR/9", "Total_Z", "MASH"] or col.endswith("_z") or col.endswith("_mash"):
                df_display[col] = df_display[col].round(2)
            else:
                df_display[col] = df_display[col].round(1)
    
    # Sort by MASH if it exists, otherwise by Total_Z
    if 'MASH' in df_display.columns:
        df_display = df_display.sort_values('MASH', ascending=False)
    elif 'Total_Z' in df_display.columns:
        df_display = df_display.sort_values('Total_Z', ascending=False)
    
    # Build a list of columns to display
    available_cols = []
    
    # Always include Name if it exists
    if 'Name' in df_display.columns:
        available_cols.append('Name')
        
    # Include Team if it exists
    if 'Team' in df_display.columns:
        available_cols.append('Team')
    
    # Include MASH if it exists
    if 'MASH' in df_display.columns:
        available_cols.append('MASH')
        
    # Include Total_Z if it exists
    if 'Total_Z' in df_display.columns:
        available_cols.append('Total_Z')
    
    # Add other display columns if they exist
    for col in display_cols:
        if col in df_display.columns and col not in available_cols:
            available_cols.append(col)
    
    # If we found no valid columns, return first 8 columns
    if not available_cols:
        return df_display.iloc[:, :min(8, df_display.shape[1])]
    
    return df_display[available_cols]

def enhance_daily_projections(daily_hitters, daily_pitchers, profiles=None, botstf=None, location=None):
    """Add supplemental stats to daily projections if possible"""
    
    # Function to match players between dataframes
    def match_players(df1, df2, id_col='IDFANGRAPHS'):
        if id_col in df1.columns and id_col in df2.columns:
            return df1.merge(df2, on=id_col, how='left')
        else:
            # Try fuzzy matching based on Name and Team
            df1_copy = df1.copy()
            name_col = next((c for c in df1.columns if c.lower() == 'name'), 'Name')
            team_col = next((c for c in df1.columns if c.lower() == 'team'), 'Team')
            
            result = {}
            for idx, row in df1.iterrows():
                name = row[name_col] if name_col in row else ""
                team = row[team_col] if team_col in row else ""
                key = f"{name} {team}"
                
                # Find best match in df2
                matches = df2[df2[name_col].str.contains(name, case=False, na=False)]
                if len(matches) > 0:
                    best_match = matches.iloc[0]
                    for col in df2.columns:
                        if col not in df1.columns and col != name_col and col != team_col:
                            if idx not in result:
                                result[idx] = {}
                            result[idx][col] = best_match[col]
            
            # Add matched columns to df1_copy
            for idx, cols in result.items():
                for col, val in cols.items():
                    df1_copy.loc[idx, col] = val
                    
            return df1_copy
    
    # Add SEAGER and 90th Pctile EV to hitters if available
    enhanced_hitters = daily_hitters.copy()
    if profiles is not None:
        seager_col = next((c for c in profiles.columns if c.upper() == "SEAGER"), None)
        ev_col = next((c for c in profiles.columns if "90TH" in c.upper() and "EV" in c.upper()), None)
        
        if seager_col and ev_col:
            profile_subset = profiles[[seager_col, ev_col, 'IDFANGRAPHS']].copy()
            enhanced_hitters = match_players(enhanced_hitters, profile_subset)
            
            # Fill missing supplementary stats with averages
            if seager_col in enhanced_hitters.columns and enhanced_hitters[seager_col].isna().any():
                seager_avg = profiles[seager_col].mean()
                enhanced_hitters[seager_col] = enhanced_hitters[seager_col].fillna(seager_avg)
            
            if ev_col in enhanced_hitters.columns and enhanced_hitters[ev_col].isna().any():
                ev_avg = profiles[ev_col].mean()
                enhanced_hitters[ev_col] = enhanced_hitters[ev_col].fillna(ev_avg)
    
    # Add BotStf and Location+ to pitchers if available
    enhanced_pitchers = daily_pitchers.copy()
    
    if botstf is not None and 'botStf' in botstf.columns:
        enhanced_pitchers = match_players(enhanced_pitchers, botstf[['IDFANGRAPHS', 'botStf']])
        
        # Fill missing BotStf with average
        if 'botStf' in enhanced_pitchers.columns and enhanced_pitchers['botStf'].isna().any():
            botstf_avg = botstf['botStf'].mean()
            enhanced_pitchers['botStf'] = enhanced_pitchers['botStf'].fillna(botstf_avg)
    
    if location is not None and 'Location+' in location.columns:
        enhanced_pitchers = match_players(enhanced_pitchers, location[['IDFANGRAPHS', 'Location+']])
        
        # Fill missing Location+ with average
        if 'Location+' in enhanced_pitchers.columns and enhanced_pitchers['Location+'].isna().any():
            location_avg = location['Location+'].mean()
            enhanced_pitchers['Location+'] = enhanced_pitchers['Location+'].fillna(location_avg)
    
    return enhanced_hitters, enhanced_pitchers

def filter_and_export_ros(league_file, standings_file=None, my_team=None):
    """Process ROS projections with league file"""
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "ros_filter")
    os.makedirs(temp_dir, exist_ok=True)
    
    top_n = 300  # Top players to use for mean/std calculations
    
    # Load ROS projections
    ros_hitters, ros_pitchers = load_and_aggregate_ros()
    
    # Add MASH calculation if standings file is provided
    category_weights = None
    if standings_file and my_team:
        try:
            category_weights = analyze_standings(standings_file, my_team)
        except Exception as e:
            print(f"Could not calculate category weights: {str(e)}")
    
    try:
        # Load league file
        league_df = pd.read_csv(league_file)
        
        # Debug status values
        status_col = next((c for c in league_df.columns if c.lower() == "status"), None)
        if not status_col:
            return None, None, None, None, None, "No Status column found in league file"
            
        # Get unique statuses
        status_values = sorted(set([str(s).strip() for s in league_df[status_col].unique() if pd.notna(s)]))
        print(f"Status values found: {status_values[:10]}...")
        
        # Filter FA status players 
        valid_fa = "FA" in status_values
        if not valid_fa:
            return None, None, None, None, None, "No 'FA' status found in league file"
            
        # Apply pre-filtering of problematic players
        ros_hitters = filter_problematic_players(ros_hitters)
        ros_pitchers = filter_problematic_players(ros_pitchers)
        
    except Exception as e:
        return None, None, None, None, None, f"Error reading league file: {str(e)}"
    
    # Filter for AFRO players
    afro_hitters, error_msg = filter_players(ros_hitters, league_df, ["AFRO"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Filter for FA players
    fa_hitters, error_msg = filter_players(ros_hitters, league_df, ["FA"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Filter pitchers for AFRO
    afro_pitchers, error_msg = filter_players(ros_pitchers, league_df, ["AFRO"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Filter pitchers for FA
    fa_pitchers, error_msg = filter_players(ros_pitchers, league_df, ["FA"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Calculate Z-scores for all dataframes using the full dataset as reference
    if afro_hitters is not None and len(afro_hitters) > 0:
        afro_hitters = calculate_zscores_hitters(afro_hitters, ros_hitters, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            afro_hitters = calculate_mash_score(afro_hitters, category_weights, is_pitcher=False)
    
    if fa_hitters is not None and len(fa_hitters) > 0:
        fa_hitters = calculate_zscores_hitters(fa_hitters, ros_hitters, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            fa_hitters = calculate_mash_score(fa_hitters, category_weights, is_pitcher=False)
        
    if afro_pitchers is not None and len(afro_pitchers) > 0:
        afro_pitchers = calculate_zscores_pitchers(afro_pitchers, ros_pitchers, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            afro_pitchers = calculate_mash_score(afro_pitchers, category_weights, is_pitcher=True)
    
    if fa_pitchers is not None and len(fa_pitchers) > 0:
        fa_pitchers = calculate_zscores_pitchers(fa_pitchers, ros_pitchers, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            fa_pitchers = calculate_mash_score(fa_pitchers, category_weights, is_pitcher=True)
            
    # Apply final filtering to each dataset
    if afro_hitters is not None:
        afro_hitters = filter_problematic_players(afro_hitters)
        # Remove duplicates
        afro_hitters = afro_hitters.drop_duplicates(subset=['Name'])
    if fa_hitters is not None:
        fa_hitters = filter_problematic_players(fa_hitters)
        # Remove duplicates
        fa_hitters = fa_hitters.drop_duplicates(subset=['Name'])
    if afro_pitchers is not None:
        afro_pitchers = filter_problematic_players(afro_pitchers)
        # Remove duplicates
        afro_pitchers = afro_pitchers.drop_duplicates(subset=['Name'])
    if fa_pitchers is not None:
        fa_pitchers = filter_problematic_players(fa_pitchers)
        # Remove duplicates
        fa_pitchers = fa_pitchers.drop_duplicates(subset=['Name'])
    
    # Format dataframes for display
    hits_display_cols = ["R", "HR", "RBI", "SB", "AVG", "OPS", "G", "PA", "AB", "BB"]
    pitch_display_cols = ["GS", "IP", "W", "SV", "K", "ERA", "WHIP", "BB", "K/BB", "botStf", "Location+"]
    
    # Add supplemental stats to display if they exist
    seager_col = next((c for c in ros_hitters.columns if c.upper() == "SEAGER"), None)
    ev_col = next((c for c in ros_hitters.columns if "90TH" in c.upper() and "EV" in c.upper()), None)
    if seager_col:
        hits_display_cols.append(seager_col)
    if ev_col:
        hits_display_cols.append(ev_col)
    
    # Safely format dataframes
    afro_hitters_display = format_dataframe_for_display(afro_hitters, hits_display_cols)
    fa_hitters_display = format_dataframe_for_display(fa_hitters, hits_display_cols)
    afro_pitchers_display = format_dataframe_for_display(afro_pitchers, pitch_display_cols, is_pitcher=True)
    fa_pitchers_display = format_dataframe_for_display(fa_pitchers, pitch_display_cols, is_pitcher=True)
    
    # Create download files
    def create_download_file(df, filename):
        if df is None or len(df) == 0:
            empty_df = pd.DataFrame(columns=["No players found"])
            file_path = os.path.join(temp_dir, filename)
            empty_df.to_csv(file_path, index=False)
            return file_path
            
        file_path = os.path.join(temp_dir, filename)
        df.to_csv(file_path, index=False)
        return file_path
    
    afro_hitters_csv = create_download_file(afro_hitters, "ROS_AFRO_hitters.csv")
    fa_hitters_csv = create_download_file(fa_hitters, "ROS_FA_hitters.csv")
    afro_pitchers_csv = create_download_file(afro_pitchers, "ROS_AFRO_pitchers.csv")
    fa_pitchers_csv = create_download_file(fa_pitchers, "ROS_FA_pitchers.csv")
    
    combined_file = os.path.join(temp_dir, "ROS_combined.zip")
    import zipfile
    with zipfile.ZipFile(combined_file, 'w') as zipf:
        if afro_hitters_csv:
            zipf.write(afro_hitters_csv, arcname="ROS_AFRO_hitters.csv")
        if fa_hitters_csv:
            zipf.write(fa_hitters_csv, arcname="ROS_FA_hitters.csv")
        if afro_pitchers_csv:
            zipf.write(afro_pitchers_csv, arcname="ROS_AFRO_pitchers.csv")
        if fa_pitchers_csv:
            zipf.write(fa_pitchers_csv, arcname="ROS_FA_pitchers.csv")
    
    # Count results
    afro_h_count = 0 if afro_hitters is None else len(afro_hitters)
    fa_h_count = 0 if fa_hitters is None else len(fa_hitters)
    afro_p_count = 0 if afro_pitchers is None else len(afro_pitchers)
    fa_p_count = 0 if fa_pitchers is None else len(fa_pitchers)
    
    mash_status = "with MASH scores" if category_weights else "without MASH (no standings file or team name)"
    status_msg = f"Found {afro_h_count} AFRO hitters, {fa_h_count} FA hitters, {afro_p_count} AFRO pitchers, and {fa_p_count} FA pitchers {mash_status}."
    
    return (
        afro_hitters_display, fa_hitters_display, 
        afro_pitchers_display, fa_pitchers_display,
        combined_file, status_msg
    )

def filter_and_export_ros_alt(league_file, standings_file=None, my_team=None):
    """Process ROS projections with league file for alternate league format"""
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "ros_filter_alt")
    os.makedirs(temp_dir, exist_ok=True)
    
    top_n = 300  # Top players to use for mean/std calculations
    
    # Load ROS projections
    ros_hitters, ros_pitchers = load_and_aggregate_ros()
    
    # Add MASH calculation if standings file is provided
    category_weights = None
    if standings_file and my_team:
        try:
            category_weights = analyze_standings(standings_file, my_team)
        except Exception as e:
            print(f"Could not calculate category weights: {str(e)}")
    
    try:
        # Load league file
        league_df = pd.read_csv(league_file)
        
        # Debug status values
        status_col = next((c for c in league_df.columns if c.lower() == "status"), None)
        if not status_col:
            return None, None, None, None, None, "No Status column found in league file"
            
        # Get unique statuses
        status_values = sorted(set([str(s).strip() for s in league_df[status_col].unique() if pd.notna(s)]))
        print(f"Status values found: {status_values[:10]}...")
        
        # Filter FA status players 
        valid_fa = "FA" in status_values
        if not valid_fa:
            return None, None, None, None, None, "No 'FA' status found in league file"
            
        # Apply pre-filtering of problematic players
        ros_hitters = filter_problematic_players(ros_hitters)
        ros_pitchers = filter_problematic_players(ros_pitchers)
        
        # Calculate additional stats needed for alt league
        # SBN (Net Stolen Bases)
        if 'SBN' not in ros_hitters.columns and 'SB' in ros_hitters.columns:
            if 'CS' in ros_hitters.columns:
                ros_hitters['SBN'] = ros_hitters['SB'] - ros_hitters['CS']
            else:
                ros_hitters['SBN'] = ros_hitters['SB']
                
        # K/9 (Strikeouts per 9 innings)
        if 'K/9' not in ros_pitchers.columns and 'K' in ros_pitchers.columns and 'IP' in ros_pitchers.columns:
            ros_pitchers['K/9'] = 9.0 * ros_pitchers['K'] / ros_pitchers['IP'].replace(0, 0.01)
            
        # K/BB (Strikeout to Walk ratio)
        if 'K/BB' not in ros_pitchers.columns and 'K' in ros_pitchers.columns and 'BB' in ros_pitchers.columns:
            ros_pitchers['K/BB'] = ros_pitchers['K'] / ros_pitchers['BB'].replace(0, 0.01)
            
        # HR/9 (Home Runs per 9 innings)
        if 'HR/9' not in ros_pitchers.columns and 'HR' in ros_pitchers.columns and 'IP' in ros_pitchers.columns:
            ros_pitchers['HR/9'] = 9.0 * ros_pitchers['HR'] / ros_pitchers['IP'].replace(0, 0.01)
        
    except Exception as e:
        return None, None, None, None, None, f"Error reading league file: {str(e)}"
    
    # Use my_team parameter instead of hardcoded "AFRO"
    my_team_status = my_team if my_team else "Ffft"
    
    # Filter for players on MY TEAM
    my_team_hitters, error_msg = filter_players(ros_hitters, league_df, [my_team_status])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Filter for FA players
    fa_hitters, error_msg = filter_players(ros_hitters, league_df, ["FA"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Filter pitchers for my team
    my_team_pitchers, error_msg = filter_players(ros_pitchers, league_df, [my_team_status])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Filter pitchers for FA
    fa_pitchers, error_msg = filter_players(ros_pitchers, league_df, ["FA"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Apply further filtering to each dataset
    if my_team_hitters is not None:
        my_team_hitters = filter_problematic_players(my_team_hitters)
    if fa_hitters is not None:
        fa_hitters = filter_problematic_players(fa_hitters)
    if my_team_pitchers is not None:
        my_team_pitchers = filter_problematic_players(my_team_pitchers)
    if fa_pitchers is not None:
        fa_pitchers = filter_problematic_players(fa_pitchers)
    
    # Calculate Z-scores for all dataframes using the ALT functions
    if my_team_hitters is not None and len(my_team_hitters) > 0:
        my_team_hitters = calculate_zscores_hitters_alt(my_team_hitters, ros_hitters, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            my_team_hitters = calculate_mash_score_alt(my_team_hitters, category_weights, is_pitcher=False)
    
    if fa_hitters is not None and len(fa_hitters) > 0:
        fa_hitters = calculate_zscores_hitters_alt(fa_hitters, ros_hitters, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            fa_hitters = calculate_mash_score_alt(fa_hitters, category_weights, is_pitcher=False)
        
    if my_team_pitchers is not None and len(my_team_pitchers) > 0:
        my_team_pitchers = calculate_zscores_pitchers_alt(my_team_pitchers, ros_pitchers, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            my_team_pitchers = calculate_mash_score_alt(my_team_pitchers, category_weights, is_pitcher=True)
    
    if fa_pitchers is not None and len(fa_pitchers) > 0:
        fa_pitchers = calculate_zscores_pitchers_alt(fa_pitchers, ros_pitchers, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            fa_pitchers = calculate_mash_score_alt(fa_pitchers, category_weights, is_pitcher=True)
            
    # Apply final filtering to each dataset
    if my_team_hitters is not None:
        my_team_hitters = filter_problematic_players(my_team_hitters)
        # Remove duplicates
        my_team_hitters = my_team_hitters.drop_duplicates(subset=['Name'])
    if fa_hitters is not None:
        fa_hitters = filter_problematic_players(fa_hitters)
        # Remove duplicates
        fa_hitters = fa_hitters.drop_duplicates(subset=['Name'])
    if my_team_pitchers is not None:
        my_team_pitchers = filter_problematic_players(my_team_pitchers)
        # Remove duplicates
        my_team_pitchers = my_team_pitchers.drop_duplicates(subset=['Name'])
    if fa_pitchers is not None:
        fa_pitchers = filter_problematic_players(fa_pitchers)
        # Remove duplicates
        fa_pitchers = fa_pitchers.drop_duplicates(subset=['Name'])
    
    # Format dataframes for display - Using ALT league categories
    hits_display_cols = ["HR", "BB", "SBN", "OBP", "SLG", "G", "PA", "AB"]
    pitch_display_cols = ["GS", "IP", "K/9", "K/BB", "ERA", "HR/9", "WHIP", "K", "BB", "botStf", "Location+"]
    
    # Add supplemental stats to display if they exist
    seager_col = next((c for c in ros_hitters.columns if c.upper() == "SEAGER"), None)
    ev_col = next((c for c in ros_hitters.columns if "90TH" in c.upper() and "EV" in c.upper()), None)
    if seager_col:
        hits_display_cols.append(seager_col)
    if ev_col:
        hits_display_cols.append(ev_col)
    
    # Safely format dataframes
    team_hitters_display = format_dataframe_for_display(my_team_hitters, hits_display_cols)
    fa_hitters_display = format_dataframe_for_display(fa_hitters, hits_display_cols)
    team_pitchers_display = format_dataframe_for_display(my_team_pitchers, pitch_display_cols, is_pitcher=True)
    fa_pitchers_display = format_dataframe_for_display(fa_pitchers, pitch_display_cols, is_pitcher=True)
    
    # Create download files
    def create_download_file(df, filename):
        if df is None or len(df) == 0:
            empty_df = pd.DataFrame(columns=["No players found"])
            file_path = os.path.join(temp_dir, filename)
            empty_df.to_csv(file_path, index=False)
            return file_path
            
        file_path = os.path.join(temp_dir, filename)
        df.to_csv(file_path, index=False)
        return file_path
    
    team_hitters_csv = create_download_file(my_team_hitters, f"ROS_{my_team_status}_hitters_alt.csv")
    fa_hitters_csv = create_download_file(fa_hitters, "ROS_FA_hitters_alt.csv")
    team_pitchers_csv = create_download_file(my_team_pitchers, f"ROS_{my_team_status}_pitchers_alt.csv")
    fa_pitchers_csv = create_download_file(fa_pitchers, "ROS_FA_pitchers_alt.csv")
    
    combined_file = os.path.join(temp_dir, "ROS_combined_alt.zip")
    import zipfile
    with zipfile.ZipFile(combined_file, 'w') as zipf:
        if team_hitters_csv:
            zipf.write(team_hitters_csv, arcname=f"ROS_{my_team_status}_hitters_alt.csv")
        if fa_hitters_csv:
            zipf.write(fa_hitters_csv, arcname="ROS_FA_hitters_alt.csv")
        if team_pitchers_csv:
            zipf.write(team_pitchers_csv, arcname=f"ROS_{my_team_status}_pitchers_alt.csv")
        if fa_pitchers_csv:
            zipf.write(fa_pitchers_csv, arcname="ROS_FA_pitchers_alt.csv")
    
    # Count results
    team_h_count = 0 if my_team_hitters is None else len(my_team_hitters)
    fa_h_count = 0 if fa_hitters is None else len(fa_hitters)
    team_p_count = 0 if my_team_pitchers is None else len(my_team_pitchers)
    fa_p_count = 0 if fa_pitchers is None else len(fa_pitchers)
    
    mash_status = "with MASH scores" if category_weights else "without MASH (no standings file or team name)"
    status_msg = f"Found {team_h_count} {my_team_status} hitters, {fa_h_count} FA hitters, {team_p_count} {my_team_status} pitchers, and {fa_p_count} FA pitchers {mash_status} for Fartolo Cologne league."
    
    return (
        team_hitters_display, fa_hitters_display, 
        team_pitchers_display, fa_pitchers_display,
        combined_file, status_msg
    )

    def filter_and_export_daily(league_file, daily_hitters_file, daily_pitchers_file, standings_file=None, my_team=None):
        """Process daily projections with league file"""
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "daily_filter")
    os.makedirs(temp_dir, exist_ok=True)

    top_n = 100  # Top players to use for mean/std calculations for daily (smaller pool)

    # Add MASH calculation if standings file is provided
    category_weights = None
    if standings_file and my_team:
        try:
            category_weights = analyze_standings(standings_file, my_team)
        except Exception as e:
            print(f"Could not calculate category weights: {str(e)}")
    
    try:
        league_df = pd.read_csv(league_file)
        daily_hitters = pd.read_csv(daily_hitters_file)
        daily_pitchers = pd.read_csv(daily_pitchers_file)
        
        # Print column names for debugging
        print(f"DEBUG: Daily hitters columns: {daily_hitters.columns.tolist()}")
        print(f"DEBUG: Daily pitchers columns: {daily_pitchers.columns.tolist()}")
        print(f"DEBUG: League file columns: {league_df.columns.tolist()}")
        
        # Find and standardize name columns - helps with matching
        for df, name in [(daily_hitters, "daily_hitters"), (daily_pitchers, "daily_pitchers"), (league_df, "league")]:
            name_col = None
            for col in df.columns:
                if col.lower() == 'name' or 'player' in col.lower():
                    name_col = col
                    break
                    
            if name_col and name_col != 'Name':
                print(f"DEBUG: Renaming {name_col} to Name in {name}")
                df.rename(columns={name_col: 'Name'}, inplace=True)
        
        # Validate that we have pitcher-specific columns in daily_pitchers
        pitcher_specific_columns = ['ERA', 'WHIP', 'IP', 'W', 'SV', 'K', 'QS']
        pitcher_columns_found = [col for col in pitcher_specific_columns if col in daily_pitchers.columns]
        
        if not pitcher_columns_found:
            return None, None, None, None, None, f"Error: Daily pitchers file doesn't contain pitcher stats. Found columns: {daily_pitchers.columns.tolist()}"
        
        # Validate that daily_hitters and daily_pitchers are different files
        if (len(daily_hitters) == len(daily_pitchers) and 
            all(daily_hitters.columns.tolist() == daily_pitchers.columns.tolist())):
            return None, None, None, None, None, "Error: Daily hitters and pitchers files appear to be identical."
        
        # Load supplementary data
        try:
            profiles = pd.read_csv('Combined_Sorted_Profiles.csv')
            botstf = pd.read_csv('BotStf.csv')
            location = pd.read_csv('Location+.csv')
            
            # Add supplementary stats to daily projections
            daily_hitters, daily_pitchers = enhance_daily_projections(daily_hitters, daily_pitchers, profiles, botstf, location)
        except Exception as e:
            print(f"Could not load supplemental stats: {str(e)}")
            # Continue without supplemental stats if files not found
            pass
            
        # Pre-filter problematic players
        daily_hitters = filter_problematic_players(daily_hitters)
        daily_pitchers = filter_problematic_players(daily_pitchers)
        
    except Exception as e:
        return None, None, None, None, None, f"Error reading files: {str(e)}"
    
    # Filter for AFRO players
    afro_hitters, error_msg = filter_players(daily_hitters, league_df, ["AFRO"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Filter for FA players
    fa_hitters, error_msg = filter_players(daily_hitters, league_df, ["FA"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Filter pitchers for AFRO
    afro_pitchers, error_msg = filter_players(daily_pitchers, league_df, ["AFRO"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Filter pitchers for FA
    fa_pitchers, error_msg = filter_players(daily_pitchers, league_df, ["FA"])
    if error_msg and "No players found" not in error_msg:
        return None, None, None, None, None, error_msg
    
    # Apply post-filtering 
    if afro_hitters is not None:
        afro_hitters = filter_problematic_players(afro_hitters)
    if fa_hitters is not None:
        fa_hitters = filter_problematic_players(fa_hitters)
    if afro_pitchers is not None:
        afro_pitchers = filter_problematic_players(afro_pitchers)
    if fa_pitchers is not None:
        fa_pitchers = filter_problematic_players(fa_pitchers)
    
    # Calculate Z-scores for all dataframes - using the full daily dataset as reference
    if afro_hitters is not None and len(afro_hitters) > 0:
        afro_hitters = calculate_zscores_hitters(afro_hitters, daily_hitters, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            afro_hitters = calculate_mash_score(afro_hitters, category_weights, is_pitcher=False)
        
    if fa_hitters is not None and len(fa_hitters) > 0:
        fa_hitters = calculate_zscores_hitters(fa_hitters, daily_hitters, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            fa_hitters = calculate_mash_score(fa_hitters, category_weights, is_pitcher=False)
        
    if afro_pitchers is not None and len(afro_pitchers) > 0:
        afro_pitchers = calculate_zscores_pitchers(afro_pitchers, daily_pitchers, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            afro_pitchers = calculate_mash_score(afro_pitchers, category_weights, is_pitcher=True)
        
    if fa_pitchers is not None and len(fa_pitchers) > 0:
        fa_pitchers = calculate_zscores_pitchers(fa_pitchers, daily_pitchers, top_n=top_n)
        # Add MASH if we have category weights
        if category_weights:
            fa_pitchers = calculate_mash_score(fa_pitchers, category_weights, is_pitcher=True)
            
    # Final filtering and deduplication
    if afro_hitters is not None:
        afro_hitters = filter_problematic_players(afro_hitters)
        afro_hitters = afro_hitters.drop_duplicates(subset=['Name'])
    if fa_hitters is not None:
        fa_hitters = filter_problematic_players(fa_hitters)
        fa_hitters = fa_hitters.drop_duplicates(subset=['Name'])
    if afro_pitchers is not None:
        afro_pitchers = filter_problematic_players(afro_pitchers)
        afro_pitchers = afro_pitchers.drop_duplicates(subset=['Name'])
    if fa_pitchers is not None:
        fa_pitchers = filter_problematic_players(fa_pitchers)
        fa_pitchers = fa_pitchers.drop_duplicates(subset=['Name'])
    
    # Format dataframes for display
    hits_display_cols = ["R", "HR", "RBI", "SB", "AVG", "OPS", "G", "PA", "AB", "BB"]
    pitch_display_cols = ["GS", "IP", "W", "SV", "K", "ERA", "WHIP", "BB", "K/BB", "botStf", "Location+"]
    
    # Add supplemental stats to display if they exist
    seager_col = next((c for c in daily_hitters.columns if c.upper() == "SEAGER"), None)
    ev_col = next((c for c in daily_hitters.columns if "90TH" in c.upper() and "EV" in c.upper()), None)
    if seager_col:
        hits_display_cols.append(seager_col)
    if ev_col:
        hits_display_cols.append(ev_col)
    
    # Safely format dataframes
    afro_hitters_display = format_dataframe_for_display(afro_hitters, hits_display_cols)
    fa_hitters_display = format_dataframe_for_display(fa_hitters, hits_display_cols)
    afro_pitchers_display = format_dataframe_for_display(afro_pitchers, pitch_display_cols, is_pitcher=True)
    fa_pitchers_display = format_dataframe_for_display(fa_pitchers, pitch_display_cols, is_pitcher=True)
    
    # Create download files
    def create_download_file(df, filename):
        if df is None or len(df) == 0:
            empty_df = pd.DataFrame(columns=["No players found"])
            file_path = os.path.join(temp_dir, filename)
            empty_df.to_csv(file_path, index=False)
            return file_path
            
        file_path = os.path.join(temp_dir, filename)
        df.to_csv(file_path, index=False)
        return file_path
    
    afro_hitters_csv = create_download_file(afro_hitters, "Daily_AFRO_hitters.csv")
    fa_hitters_csv = create_download_file(fa_hitters, "Daily_FA_hitters.csv")
    afro_pitchers_csv = create_download_file(afro_pitchers, "Daily_AFRO_pitchers.csv")
    fa_pitchers_csv = create_download_file(fa_pitchers, "Daily_FA_pitchers.csv")
    
    combined_file = os.path.join(temp_dir, "Daily_combined.zip")
    import zipfile
    with zipfile.ZipFile(combined_file, 'w') as zipf:
        if afro_hitters_csv:
            zipf.write(afro_hitters_csv, arcname="Daily_AFRO_hitters.csv")
        if fa_hitters_csv:
            zipf.write(fa_hitters_csv, arcname="Daily_FA_hitters.csv")
        if afro_pitchers_csv:
            zipf.write(afro_pitchers_csv, arcname="Daily_AFRO_pitchers.csv")
        if fa_pitchers_csv:
            zipf.write(fa_pitchers_csv, arcname="Daily_FA_pitchers.csv")
    
    # Count results
    afro_h_count = 0 if afro_hitters is None else len(afro_hitters)
    fa_h_count = 0 if fa_hitters is None else len(fa_hitters)
    afro_p_count = 0 if afro_pitchers is None else len(afro_pitchers)
    fa_p_count = 0 if fa_pitchers is None else len(fa_pitchers)
    
    mash_status = "with MASH scores" if category_weights else "without MASH (no standings file or team name)"
    status_msg = f"Found {afro_h_count} AFRO hitters, {fa_h_count} FA hitters, {afro_p_count} AFRO pitchers, and {fa_p_count} FA pitchers {mash_status} for Nuke Laloosh league."
    
    return (
        afro_hitters_display, fa_hitters_display, 
        afro_pitchers_display, fa_pitchers_display,
        combined_file, status_msg
    )
def filter_and_export_nfbc(league_file, daily_hitters_file, daily_pitchers_file, standings_file=None, my_team=None):
    """Process daily projections with NFBC league file"""
    # Create temp directory if it doesn't exist
    temp_dir = os.path.join(tempfile.gettempdir(), "nfbc_filter")
    os.makedirs(temp_dir, exist_ok=True)

    top_n = 100  # Top players to use for mean/std calculations for daily (smaller pool)

    # Add MASH calculation if standings file is provided
    category_weights = None
    if standings_file and my_team:
        try:
            category_weights = analyze_standings_nfbc(standings_file, my_team)
        except Exception as e:
            print(f"Could not calculate category weights: {str(e)}")
    
    try:
        league_df = pd.read_csv(league_file)
        daily_hitters = pd.read_csv(daily_hitters_file)
        daily_pitchers = pd.read_csv(daily_pitchers_file)
        
        # Print column names for debugging
        print(f"DEBUG: NFBC league columns: {league_df.columns.tolist()}")
        print(f"DEBUG: Daily hitters columns: {daily_hitters.columns.tolist()}")
        print(f"DEBUG: Daily pitchers columns: {daily_pitchers.columns.tolist()}")
        
        # Find and standardize name columns for better matching
        # For league file, the player column is likely "Players"
        if 'Players' in league_df.columns:
            league_df.rename(columns={'Players': 'Name'}, inplace=True)
        
        # For daily projection files, ensure they have a "Name" column
        for df, name in [(daily_hitters, "daily_hitters"), (daily_pitchers, "daily_pitchers")]:
            name_col = None
            for col in df.columns:
                if col.lower() == 'name' or 'player' in col.lower():
                    name_col = col
                    break
                    
            if name_col and name_col != 'Name':
                print(f"DEBUG: Renaming {name_col} to Name in {name}")
                df.rename(columns={name_col: 'Name'}, inplace=True)
        
        # Check for pitcher-specific columns in daily_pitchers
        pitcher_specific_columns = ['ERA', 'WHIP', 'IP', 'W', 'SV', 'K']
        pitcher_columns_found = [col for col in pitcher_specific_columns if col in daily_pitchers.columns]
        
        if not pitcher_columns_found:
            return None, None, None, None, None, f"Error: Daily pitchers file doesn't contain pitcher stats. Found columns: {daily_pitchers.columns.tolist()}"
        
        # Determine owner column in league file - likely "Owner"
        owner_col = 'Owner' if 'Owner' in league_df.columns else None
        if not owner_col:
            owner_candidates = [col for col in league_df.columns if 'own' in col.lower()]
            owner_col = owner_candidates[0] if owner_candidates else None
        
        if not owner_col:
            return None, None, None, None, None, "No owner column found in NFBC league file"
            
        print(f"DEBUG: Using '{owner_col}' as owner column in NFBC league file")
        
        # Determine my team's players - where Owner equals my_team
        if not my_team:
            return None, None, None, None, None, "Team name is required for filtering NFBC players"
            
        # Print unique owners for debugging
        unique_owners = sorted(league_df[owner_col].dropna().unique())
        print(f"DEBUG: Available owners: {unique_owners[:10]}...")
        
        # Check if my_team exists in owners
        if my_team not in unique_owners:
            best_match = process.extractOne(my_team, unique_owners)
            if best_match and best_match[1] > 80:
                my_team = best_match[0]
                print(f"DEBUG: Using closest match '{my_team}' as team name")
            else:
                return None, None, None, None, None, f"Owner '{my_team}' not found in NFBC league file. Available owners: {unique_owners[:5]}..."
        
        # Filter players owned by my team and FA players
        my_players = league_df[league_df[owner_col] == my_team]
        fa_players = league_df[pd.isna(league_df[owner_col]) | (league_df[owner_col] == '')]
        
        print(f"DEBUG: Found {len(my_players)} players owned by {my_team}")
        print(f"DEBUG: Found {len(fa_players)} free agent players")
            
        # Pre-filter problematic players
        daily_hitters = filter_problematic_players(daily_hitters)
        daily_pitchers = filter_problematic_players(daily_pitchers)
        
        # Determine how to match players between files
        # NFBCID is the key identifier
        id_col = 'NFBCID' if 'NFBCID' in daily_hitters.columns and 'id' in league_df.columns else None
        if id_col:
            print(f"DEBUG: Using NFBCID for player matching")
            # Rename league file column from 'id' to 'NFBCID' for easier matching
            league_df.rename(columns={'id': 'NFBCID'}, inplace=True)
            
    except Exception as e:
        return None, None, None, None, None, f"Error reading files: {str(e)}"
    
    # Filter hitters and pitchers for my team and FA
    # Match by NFBCID if available, otherwise use name
    if id_col:
        # Match by NFBCID
        my_hitters = daily_hitters[daily_hitters['NFBCID'].isin(my_players['NFBCID'])]
        fa_hitters = daily_hitters[daily_hitters['NFBCID'].isin(fa_players['NFBCID'])]
        my_pitchers = daily_pitchers[daily_pitchers['NFBCID'].isin(my_players['NFBCID'])]
        fa_pitchers = daily_pitchers[daily_pitchers['NFBCID'].isin(fa_players['NFBCID'])]
    else:
        # Match by name (less reliable)
        my_player_names = set([name.lower().strip() for name in my_players['Name'] if isinstance(name, str)])
        fa_player_names = set([name.lower().strip() for name in fa_players['Name'] if isinstance(name, str)])
        
        my_hitters = daily_hitters[daily_hitters['Name'].str.lower().str.strip().isin(my_player_names)]
        fa_hitters = daily_hitters[daily_hitters['Name'].str.lower().str.strip().isin(fa_player_names)]
        my_pitchers = daily_pitchers[daily_pitchers['Name'].str.lower().str.strip().isin(my_player_names)]
        fa_pitchers = daily_pitchers[daily_pitchers['Name'].str.lower().str.strip().isin(fa_player_names)]
    
    # Apply post-filtering 
    if my_hitters is not None:
        my_hitters = filter_problematic_players(my_hitters)
    if fa_hitters is not None:
        fa_hitters = filter_problematic_players(fa_hitters)
    if my_pitchers is not None:
        my_pitchers = filter_problematic_players(my_pitchers)
    if fa_pitchers is not None:
        fa_pitchers = filter_problematic_players(fa_pitchers)
        
    # Calculate Z-scores with NFBC categories
    if my_hitters is not None and len(my_hitters) > 0:
        my_hitters = calculate_zscores_nfbc_hitters(my_hitters, daily_hitters, top_n=top_n)
        if category_weights:
            my_hitters = calculate_mash_score(my_hitters, category_weights, is_pitcher=False)
    
    if fa_hitters is not None and len(fa_hitters) > 0:
        fa_hitters = calculate_zscores_nfbc_hitters(fa_hitters, daily_hitters, top_n=top_n)
        if category_weights:
            fa_hitters = calculate_mash_score(fa_hitters, category_weights, is_pitcher=False)
    
    if my_pitchers is not None and len(my_pitchers) > 0:
        my_pitchers = calculate_zscores_nfbc_pitchers(my_pitchers, daily_pitchers, top_n=top_n)
        if category_weights:
            my_pitchers = calculate_mash_score(my_pitchers, category_weights, is_pitcher=True)
    
    if fa_pitchers is not None and len(fa_pitchers) > 0:
        fa_pitchers = calculate_zscores_nfbc_pitchers(fa_pitchers, daily_pitchers, top_n=top_n)
        if category_weights:
            fa_pitchers = calculate_mash_score(fa_pitchers, category_weights, is_pitcher=True)
    
    # Format dataframes for display - NFBC specific columns
    hits_display_cols = ["R", "HR", "RBI", "SB", "AVG", "OPS", "G", "PA", "AB", "BB", "SO"]
    pitch_display_cols = ["IP", "W", "SV", "K", "ERA", "WHIP", "BB", "L", "H", "ER"]
    
    # Safely format dataframes
    my_hitters_display = format_dataframe_for_display(my_hitters, hits_display_cols)
    fa_hitters_display = format_dataframe_for_display(fa_hitters, hits_display_cols)
    my_pitchers_display = format_dataframe_for_display(my_pitchers, pitch_display_cols, is_pitcher=True)
    fa_pitchers_display = format_dataframe_for_display(fa_pitchers, pitch_display_cols, is_pitcher=True)
    
    # Create download files
    def create_download_file(df, filename):
        if df is None or len(df) == 0:
            empty_df = pd.DataFrame(columns=["No players found"])
            file_path = os.path.join(temp_dir, filename)
            empty_df.to_csv(file_path, index=False)
            return file_path
            
        file_path = os.path.join(temp_dir, filename)
        df.to_csv(file_path, index=False)
        return file_path
    
    my_hitters_csv = create_download_file(my_hitters, f"NFBC_{my_team}_hitters.csv")
    fa_hitters_csv = create_download_file(fa_hitters, "NFBC_FA_hitters.csv")
    my_pitchers_csv = create_download_file(my_pitchers, f"NFBC_{my_team}_pitchers.csv")
    fa_pitchers_csv = create_download_file(fa_pitchers, "NFBC_FA_pitchers.csv")
    
    combined_file = os.path.join(temp_dir, "NFBC_combined.zip")
    import zipfile
    with zipfile.ZipFile(combined_file, 'w') as zipf:
        if my_hitters_csv:
            zipf.write(my_hitters_csv, arcname=f"NFBC_{my_team}_hitters.csv")
        if fa_hitters_csv:
            zipf.write(fa_hitters_csv, arcname="NFBC_FA_hitters.csv")
        if my_pitchers_csv:
            zipf.write(my_pitchers_csv, arcname=f"NFBC_{my_team}_pitchers.csv")
        if fa_pitchers_csv:
            zipf.write(fa_pitchers_csv, arcname="NFBC_FA_pitchers.csv")
    
    # Count results
    my_h_count = 0 if my_hitters is None else len(my_hitters)
    fa_h_count = 0 if fa_hitters is None else len(fa_hitters)
    my_p_count = 0 if my_pitchers is None else len(my_pitchers)
    fa_p_count = 0 if fa_pitchers is None else len(fa_pitchers)
    
    mash_status = "with MASH scores" if category_weights else "without MASH (no standings file or team name)"
    status_msg = f"Found {my_h_count} {my_team} hitters, {fa_h_count} FA hitters, {my_p_count} {my_team} pitchers, and {fa_p_count} FA pitchers {mash_status} for NFBC league."
    
    return (
        my_hitters_display, fa_hitters_display, 
        my_pitchers_display, fa_pitchers_display,
        combined_file, status_msg
    )
    # Add NFBC League tab after your other tabs
with gr.Tab("NFBC League"):
    gr.Markdown("## NFBC Daily Projections")
    gr.Markdown("Upload your NFBC league file and daily projections to find available players.")
    
    # File uploads - one per line to avoid nesting issues
    nfbc_league_file = gr.File(label="Upload NFBC League File")
    nfbc_hitters_file = gr.File(label="Upload Razzball Daily Hitters")
    nfbc_pitchers_file = gr.File(label="Upload Razzball Daily Pitchers")
    nfbc_standings_file = gr.File(label="Upload Standings File (Optional for MASH)")
    nfbc_team_name = gr.Textbox(label="Your Owner Name (Required)")
    
    # Button and status
    nfbc_run_button = gr.Button("Generate NFBC Reports")
    nfbc_status = gr.Textbox(label="Status", interactive=False)
    
    # Results display - using Markdown headers instead of tabs
    gr.Markdown("### My Hitters")
    nfbc_my_hitters_table = gr.Dataframe(label="My Hitters", interactive=False)
    
    gr.Markdown("### FA Hitters")
    nfbc_fa_hitters_table = gr.Dataframe(label="FA Hitters", interactive=False)
    
    gr.Markdown("### My Pitchers")
    nfbc_my_pitchers_table = gr.Dataframe(label="My Pitchers", interactive=False)
    
    gr.Markdown("### FA Pitchers")
    nfbc_fa_pitchers_table = gr.Dataframe(label="FA Pitchers", interactive=False)
    
    # Download section
    nfbc_download = gr.File(label="Download CSV Files")

# Add the connection in your connections section
nfbc_run_button.click(
    fn=filter_and_export_nfbc,
    inputs=[nfbc_league_file, nfbc_hitters_file, nfbc_pitchers_file, nfbc_standings_file, nfbc_team_name],
    outputs=[nfbc_my_hitters_table, nfbc_fa_hitters_table, nfbc_my_pitchers_table, nfbc_fa_pitchers_table, nfbc_download, nfbc_status]
)


# Fix the lineup handler
lineup_run_button.click(
    fn=optimize_daily_lineup,
    inputs=[lineup_hitters_file, lineup_league_file, lineup_standings_file, lineup_team_name],
    outputs=[lineup_table, lineup_download, lineup_status]
)

# Create NFBC interface function - place this before your Gradio interface definition
def create_nfbc_interface():
    """Create a simple NFBC interface without complex nesting"""
    nfbc_league_file = gr.File(label="Upload NFBC League File")
    nfbc_hitters_file = gr.File(label="Upload Razzball Daily Hitters")
    nfbc_pitchers_file = gr.File(label="Upload Razzball Daily Pitchers")
    nfbc_standings_file = gr.File(label="Upload Standings File (Optional for MASH)")
    nfbc_team_name = gr.Textbox(label="Your Owner Name (Required)")
    nfbc_run_button = gr.Button("Generate NFBC Reports")
    nfbc_status = gr.Textbox(label="Status", interactive=False)
    nfbc_my_hitters_table = gr.Dataframe(label="My Hitters", interactive=False)
    nfbc_fa_hitters_table = gr.Dataframe(label="FA Hitters", interactive=False)
    nfbc_my_pitchers_table = gr.Dataframe(label="My Pitchers", interactive=False)
    nfbc_fa_pitchers_table = gr.Dataframe(label="FA Pitchers", interactive=False)
    nfbc_download = gr.File(label="Download CSV Files")
    
    nfbc_run_button.click(
        fn=filter_and_export_nfbc,
        inputs=[nfbc_league_file, nfbc_hitters_file, nfbc_pitchers_file, nfbc_standings_file, nfbc_team_name],
        outputs=[nfbc_my_hitters_table, nfbc_fa_hitters_table, nfbc_my_pitchers_table, nfbc_fa_pitchers_table, nfbc_download, nfbc_status]
    )
    
    return {
        "league_file": nfbc_league_file,
        "hitters_file": nfbc_hitters_file,
        "pitchers_file": nfbc_pitchers_file,
        "standings_file": nfbc_standings_file,
        "team_name": nfbc_team_name,
        "run_button": nfbc_run_button,
        "status": nfbc_status,
        "my_hitters_table": nfbc_my_hitters_table,
        "fa_hitters_table": nfbc_fa_hitters_table,
        "my_pitchers_table": nfbc_my_pitchers_table,
        "fa_pitchers_table": nfbc_fa_pitchers_table,
        "download": nfbc_download
    }

# At the end of your app.py file, after all other tabs but before demo.launch()
# NFBC Interface (without tabs/rows)
gr.Markdown("## NFBC League Tools")
gr.Markdown("Upload your NFBC league file and daily projections to find available players.")

# Create all NFBC components without tabs or rows
nfbc_components = create_nfbc_interface()

# System Information Display (after the NFBC components)
gr.Markdown("## System Information")
gr.Markdown(f"""
- Current Date and Time (UTC): 2025-05-23 19:21:06
- Current User's Login: ElPolloGordo69
""")

# Additional dynamic system info display:
current_time = datetime.datetime.utcnow().strftime('%Y-%m-%d %H:%M:%S')
username = os.environ.get('USER', getpass.getuser())
gr.Markdown(f"### Current System Time: {current_time}")

if __name__ == "__main__":
    # Launch with server settings appropriate for Hugging Face Spaces
    demo.launch(
        server_name="0.0.0.0",  # Bind to all network interfaces
        server_port=7860,       # Standard port for Gradio
        share=False,            # Don't create a public link
        debug=False,            # Disable debug mode in production
        show_error=True,        # Show detailed error messages
        favicon_path=None,      # Use default favicon
        ssl_verify=True,        # Verify SSL certs
        quiet=False             # Show startup messages
    )
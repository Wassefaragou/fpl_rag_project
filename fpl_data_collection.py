import requests
import pandas as pd
import json
from datetime import datetime

def fetch_fpl_data():
    """Fetch current Fantasy Premier League data from their API"""
    # FPL API base URL
    base_url = "https://fantasy.premierleague.com/api/"
    
    # Get basic player data
    response = requests.get(f"{base_url}bootstrap-static/")
    data = response.json()
    
    # Extract player data
    players_df = pd.DataFrame(data['elements'])
    teams_df = pd.DataFrame(data['teams'])
    
    # Add team name to player data
    players_df = players_df.merge(teams_df[['id', 'name','position']], 
                                  left_on='team', 
                                  right_on='id', 
                                  suffixes=('', '_team'))
    players_df.rename(columns={'name': 'team_name','position':'position_team'}, inplace=True)
   
    
    # Select relevant columns
    selected_columns = [
        'id', 'web_name', 'first_name', 'second_name','team_name',
        'position_team','element_type', 'selected_by_percent', 'now_cost', 'form',
        'points_per_game', 'total_points', 'minutes', 'goals_scored',
        'assists', 'clean_sheets', 'goals_conceded', 'yellow_cards',
        'red_cards', 'saves', 'bonus', 'bps', 'influence',
        'creativity', 'threat', 'ict_index', 'value_season',
        'transfers_in', 'transfers_out'
    ]
    players_df = players_df[selected_columns]
    
    # Replace element_type with actual position
    position_map = {
    1: 'Goalkeeper',
    2: 'Defender',
    3: 'Midfielder',
    4: 'Forward',
    5: 'Manager'
}
    players_df['position'] = players_df['element_type'].map(position_map)
    
    # Convert cost to actual £ value
    players_df['price'] = players_df['now_cost'] / 10
    
    # Add collection timestamp
    players_df['data_date'] = datetime.now().strftime('%Y-%m-%d')
    
    # Fetch detailed player data for each player
    player_details = {}
    for player_id in players_df['id']:  
        try:
            player_history = requests.get(f"{base_url}element-summary/{player_id}/").json()
            player_details[player_id] = player_history
        except:
            print(f"Failed to fetch details for player {player_id}")
    
    return players_df, player_details

def prepare_documents(players_df, player_details):
    """Convert player data into text documents for RAG system"""
    documents = []
    
    # Basic player info
    for _, player in players_df.iterrows():
        doc = f"""
Player: {player['first_name']} {player['second_name']} ({player['web_name']})   
Team: {player['team_name']}
Position team: {player['position_team']}
Position: {player['position']}
Price: £{player['price']}M
Selected by: {player['selected_by_percent']}%
Form: {player['form']}
Points per game: {player['points_per_game']}
Total points: {player['total_points']}
Minutes played: {player['minutes']}
Goals: {player['goals_scored']}
Assists: {player['assists']}
Clean sheets: {player['clean_sheets']}
Goals conceded: {player['goals_conceded']}
Yellow cards: {player['yellow_cards']}
Red cards: {player['red_cards']}
Bonus points: {player['bonus']}
BPS: {player['bps']}
Influence: {player['influence']}
Creativity: {player['creativity']}
Threat: {player['threat']}
ICT Index: {player['ict_index']}
Transfers in: {player['transfers_in']}
Transfers out: {player['transfers_out']}
Data as of: {player['data_date']}
"""
        documents.append({
            'content': doc,
            'metadata': {
                'player_id': player['id'],
                'player_name': f"{player['first_name']} {player['second_name']}",
                'position': player['position'],
                'team_name': player['team_name'],
                'doc_type': 'basic_info'
            }
        })
    
    # Add detailed stats from player history if available
    for player_id, details in player_details.items():
        if 'history' in details and details['history']:
            player = players_df[players_df['id'] == player_id].iloc[0]
            recent_matches = details['history'][-5:]  # Last 5 matches
            
            match_details = []
            for match in recent_matches:
                match_info = f"""
Match against {match.get('opponent_team', 'Unknown')} 
(GW {match.get('round', 'Unknown')}): 
Points: {match.get('total_points', 0)}, 
Minutes: {match.get('minutes', 0)}, 
Goals: {match.get('goals_scored', 0)}, 
Assists: {match.get('assists', 0)}
"""
                match_details.append(match_info)
            
            doc = f"""
Recent performance for {player['first_name']} {player['second_name']}:
{''.join(match_details)}
"""
            documents.append({
                'content': doc,
                'metadata': {
                    'player_id': player_id,
                    'player_name': f"{player['first_name']} {player['second_name']}",
                    'position': player['position'],
                    'doc_type': 'recent_performance'
                }
            })
    
    return documents

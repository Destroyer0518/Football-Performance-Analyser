
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import requests
from bs4 import BeautifulSoup
import os
import base64

# PDF deps
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(layout="wide", page_title="UCL 23-24 — Full Data Fetcher", initial_sidebar_state="expanded")

TEAM_FILE = "ucl_team_venue_agg.csv"
PLAYERS_FILE = "ucl_players.csv"

# ---------- Utilities ----------
@st.cache_data
def generate_sample_data(seed=42):
    np.random.seed(seed)
    teams = ["Real Madrid", "Manchester City", "Bayern Munich", "Paris Saint-Germain",
             "AC Milan", "Barcelona", "Juventus", "Liverpool", "Chelsea", "Dortmund"]
    seasons = [2023, 2024]
    rows = []
    for season in seasons:
        for team in teams:
            for venue in ["Home","Away"]:
                matches = np.random.randint(6,12)
                wins = np.random.randint(0,matches+1)
                draws = np.random.randint(0,matches-wins+1)
                losses = matches-wins-draws
                gf = np.random.randint(0,matches*4+1)
                ga = np.random.randint(0,matches*3+1)
                ast = int(gf * np.random.uniform(0.4,0.9))
                rows.append({
                    "season":season,"team":team,"venue":venue,"matches":matches,"wins":wins,
                    "draws":draws,"losses":losses,"goals_for":gf,"goals_against":ga,
                    "assists":ast,"GplusA":gf+ast,"fouls_committed":np.random.randint(5*matches,20*matches),
                    "fouls_conceded":np.random.randint(5*matches,20*matches),"winners_or_equalisers":np.random.randint(0,gf+1),
                    "xG":round(np.random.uniform(0.5,2.5)*matches,2),"xA":round(np.random.uniform(0.2,1.2)*matches,2),"pass_accuracy":round(np.random.uniform(70,92),1)
                })
    df = pd.DataFrame(rows)
    # players
    players = []
    for season in seasons:
        for team in teams:
            for i in range(1,16):
                player = f"{team.split()[0]} Player {i}"
                goals = np.random.poisson(4)
                assists = max(0,int(goals*np.random.uniform(0.2,0.8)))
                minutes = np.random.randint(200,3500)
                players.append({
                    "season":season,"team":team,"player":player,"goals":goals,"assists":assists,
                    "GplusA":goals+assists,"minutes":minutes,"fouls_committed":np.random.randint(0,40),
                    "xG":round(np.random.uniform(0.1,1.0)*(minutes/90),2),"xA":round(np.random.uniform(0.05,0.6)*(minutes/90),2),
                    "pass_accuracy":round(np.random.uniform(65,95),1),"tackles":np.random.randint(0,80),"saves":np.random.randint(0,150) if i>12 else 0
                })
    df_players = pd.DataFrame(players)
    return df, df_players

def robust_load():
    if os.path.exists(TEAM_FILE) and os.path.exists(PLAYERS_FILE):
        try:
            df_team = pd.read_csv(TEAM_FILE)
            df_players = pd.read_csv(PLAYERS_FILE)
        except Exception:
            df_team, df_players = generate_sample_data()
            df_team.to_csv(TEAM_FILE,index=False)
            df_players.to_csv(PLAYERS_FILE,index=False)
    else:
        df_team, df_players = generate_sample_data()
        df_team.to_csv(TEAM_FILE,index=False)
        df_players.to_csv(PLAYERS_FILE,index=False)
    # ensure player column exists
    if 'player' not in df_players.columns:
        for alt in ['name','player_name','full_name']:
            if alt in df_players.columns:
                df_players = df_players.rename(columns={alt:'player'})
                break
    if 'player' not in df_players.columns:
        df_players['player'] = df_players.index.map(lambda i: f"Player_{i}")
    df_players['player'] = df_players['player'].astype(str)
    return df_team, df_players

df_team, df_players = robust_load()

def enrich_team_df(df):
    df = df.copy()
    if 'goal_diff' not in df.columns:
        df['goal_diff'] = df['goals_for'] - df['goals_against']
    if 'points' not in df.columns:
        df['points'] = df['wins']*3 + df['draws']*1
    return df

df_team = enrich_team_df(df_team)

# ---------- Fetchers ----------
def fetch_fbref_season(season_year=2023):
    """Fetch FBref Champions League season pages and parse player and team stats tables.
    This is best-effort: FBref pages provide many tables; we pick the main stats tables.
    """
    base = "https://fbref.com/en/comps/8/{}/{0}-{}-Champions-League-Stats".format(season_year, season_year)
    # The exact URL pattern can vary; use the season landing page then find stats links.
    landing = f"https://fbref.com/en/comps/8/{season_year}-{season_year+1}/"
    try:
        r = requests.get(landing, timeout=10)
        r.raise_for_status()
        tables = pd.read_html(r.text)
        # FBref landing contains many tables; filter by ones with 'Player' or 'Squad' columns
        players = pd.DataFrame()
        teams = pd.DataFrame()
        for t in tables:
            cols = [c.lower() for c in t.columns.astype(str)]
            if any('player' in str(c).lower() for c in cols) and ('goals' in cols or 'g' in cols):
                if 'Player' in t.columns or 'player' in t.columns:
                    players = t
            if any('squad' in str(c).lower() for c in cols) or any('team' in str(c).lower() for c in cols):
                teams = t if teams.empty else teams
        return teams, players, None
    except Exception as e:
        return None, None, str(e)

def fetch_uefa_team_stats():
    """Fetch basic team stats from the UEFA season statistics page.
    """
    url = "https://www.uefa.com/uefachampionsleague/history/seasons/2024/statistics/"
    try:
        r = requests.get(url, timeout=10)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        # Try to extract club ranking snippet - heavy DOM parsing, best-effort
        tables = pd.read_html(r.text)
        # return first table found as fallback
        return tables[0] if tables else None, None
    except Exception as e:
        return None, str(e)

# ---------- PDF helpers with charts embedded ----------
def fig_to_png_bytes(fig):
    buf = BytesIO()
    fig.savefig(buf, format='png', bbox_inches='tight')
    plt.close(fig)
    buf.seek(0)
    return buf.getvalue()

def create_team_pdf_with_charts(team, season, df_team_sel, players_df):
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=20,leftMargin=20,topMargin=20,bottomMargin=20)
    styles = getSampleStyleSheet()
    flow = []
    title = f"{team} — Season {season} — Team Report"
    flow.append(Paragraph(title, styles['Title']))
    flow.append(Spacer(1,12))
    # Summary table
    summary = {
        "Matches": int(df_team_sel['matches'].sum()),
        "Wins": int(df_team_sel['wins'].sum()),
        "Draws": int(df_team_sel['draws'].sum()),
        "Losses": int(df_team_sel['losses'].sum()),
        "Goals For": int(df_team_sel['goals_for'].sum()),
        "Goals Against": int(df_team_sel['goals_against'].sum()),
        "Goal Diff": int((df_team_sel['goals_for'] - df_team_sel['goals_against']).sum()),
        "G+A": int(df_team_sel['GplusA'].sum()),
    }
    data = [["Metric","Value"]]+[[k,str(v)] for k,v in summary.items()]
    t = Table(data, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#d3d3d3")),('GRID',(0,0),(-1,-1),0.5,colors.black)]))
    flow.append(t); flow.append(Spacer(1,12))
    # Create a matplotlib figure and embed
    fig, ax = plt.subplots(figsize=(6,3))
    ax.bar(df_team_sel['venue'], df_team_sel['wins'], label='Wins')
    ax.set_title("Wins by Venue")
    png = fig_to_png_bytes(fig)
    img = Image(BytesIO(png), width=400, height=200)
    flow.append(img); flow.append(Spacer(1,12))
    # Top players table
    flow.append(Paragraph("Top players (by G+A)", styles['Heading2']))
    top_players = players_df.sort_values('GplusA', ascending=False).head(8)
    tbl = [list(top_players.columns)]
    for _, r in top_players.iterrows():
        tbl.append([str(r[c]) for c in top_players.columns])
    t2 = Table(tbl, hAlign='LEFT', repeatRows=1)
    t2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.3,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0"))]))
    flow.append(t2)
    flow.append(Spacer(1,12))
    flow.append(Paragraph("Generated by UCL 23-24 Explorer (with FBref/UEFA fetch)", styles['Normal']))
    doc.build(flow)
    buffer.seek(0)
    return buffer.getvalue()

# ---------- App UI ----------
st.sidebar.title("Controls")
page = st.sidebar.radio("Go to", ["Overview","Team","Player","Upload CSV","Fetch FBref", "Fetch UEFA", "Quick Wikipedia import"])

if st.sidebar.button("Regenerate sample data"):
    df_team_new, df_players_new = generate_sample_data(seed=np.random.randint(0,9999))
    df_team_new.to_csv(TEAM_FILE,index=False)
    df_players_new.to_csv(PLAYERS_FILE,index=False)
    st.experimental_rerun()

st.title("UCL 2023-24 — Full Fetch + PDF Charts")

if page == "Overview":
    st.header("Season Overview")
    season = st.selectbox("Season", sorted(df_team['season'].unique()))
    df_season = df_team[df_team['season']==season].groupby('team').sum().reset_index()
    df_season['goal_diff'] = df_season['goals_for'] - df_season['goals_against']
    st.dataframe(df_season.sort_values('goal_diff', ascending=False))
elif page == "Team":
    st.header("Team Explorer")
    season = st.selectbox("Season (team)", sorted(df_team['season'].unique()), key='tseason2')
    team = st.selectbox("Team", sorted(df_team['team'].unique()), key='team2')
    df_sel = df_team[(df_team['season']==season)&(df_team['team']==team)].copy()
    if df_sel.empty:
        st.warning("No data")
    else:
        st.dataframe(df_sel[['venue','matches','wins','draws','losses','goals_for','goals_against','GplusA']])
        players_sel = df_players[(df_players['season']==season)&(df_players['team']==team)].copy()
        if not players_sel.empty:
            st.dataframe(players_sel[['player','goals','assists','GplusA']])
        # export PDF with charts
        if st.button("Download team PDF with charts"):
            pdf_bytes = create_team_pdf_with_charts(team, season, df_sel, players_sel if not players_sel.empty else pd.DataFrame())
            st.download_button("Click to download", pdf_bytes, f"{team}_{season}_report_charts.pdf", "application/pdf")
elif page == "Player":
    st.header("Player Explorer")
    season = st.selectbox("Season (player)", sorted(df_players['season'].unique()), key='pseason2')
    team_filter = st.selectbox("Filter by team", ["All"] + sorted(df_players['team'].unique()), key='pteam2')
    df_pl = df_players[df_players['season']==season].copy()
    if team_filter != "All":
        df_pl = df_pl[df_pl['team']==team_filter]
    st.dataframe(df_pl.sort_values('GplusA', ascending=False).reset_index(drop=True))
    sel = st.selectbox("Select player", [""] + sorted(df_pl['player'].unique()))
    if sel:
        p = df_pl[df_pl['player']==sel].squeeze()
        st.write(p.to_frame().T)
elif page == "Upload CSV":
    st.header("Upload CSVs to replace local data")
    up_team = st.file_uploader("Team CSV", type=['csv'])
    up_players = st.file_uploader("Players CSV", type=['csv'])
    if st.button("Load uploaded"):
        if up_team:
            df_t = pd.read_csv(up_team)
            df_t.to_csv(TEAM_FILE,index=False)
            st.success("Team CSV saved")
        if up_players:
            df_p = pd.read_csv(up_players)
            if 'player' not in df_p.columns:
                for alt in ['name','player_name','full_name']:
                    if alt in df_p.columns:
                        df_p = df_p.rename(columns={alt:'player'})
                        break
            if 'GplusA' not in df_p.columns and {'goals','assists'}.issubset(df_p.columns):
                df_p['GplusA'] = df_p['goals'] + df_p['assists']
            df_p.to_csv(PLAYERS_FILE,index=False)
            st.success("Players CSV saved")
        st.experimental_rerun()
elif page == "Fetch FBref":
    st.header("Fetch FBref 2023-24 data (best-effort)")
    st.markdown("This will attempt to fetch tables from FBref for the 2023-24 Champions League season and map common columns.")
    if st.button("Fetch FBref now"):
        with st.spinner("Fetching FBref..."):
            teams_tbl, players_tbl, err = fetch_fbref_season(2023)
            if err:
                st.error(f"Fetch failed: {err}")
            else:
                if teams_tbl is not None and not teams_tbl.empty:
                    st.success("Found FBref tables — attempting to map and save.")
                    # attempt mapping: try to extract basic columns
                    # For players, ensure 'player' and 'goals'
                    try:
                        if 'Player' in players_tbl.columns:
                            players_tbl = players_tbl.rename(columns={'Player':'player'})
                        if 'Squad' in teams_tbl.columns:
                            teams_tbl = teams_tbl.rename(columns={'Squad':'team'})
                        # minimal subset to save
                        keep_p = [c for c in ['player','Nation','Pos','Squad','Comp','Goals','Assists'] if c in players_tbl.columns]
                        # coerce to common names
                        if 'Goals' in players_tbl.columns:
                            players_tbl = players_tbl.rename(columns={'Goals':'goals'})
                        if 'Assists' in players_tbl.columns:
                            players_tbl = players_tbl.rename(columns={'Assists':'assists'})
                        players_tbl['season'] = 2024
                        # map Squad -> team if present
                        if 'Squad' in players_tbl.columns:
                            players_tbl = players_tbl.rename(columns={'Squad':'team'})
                        # compute GplusA where possible
                        if 'GplusA' not in players_tbl.columns and {'goals','assists'}.issubset(players_tbl.columns):
                            players_tbl['GplusA'] = players_tbl['goals'] + players_tbl['assists']
                        # save filtered players to local file
                        cols_to_save = [c for c in ['season','team','player','goals','assists','GplusA'] if c in players_tbl.columns]
                        if cols_to_save:
                            players_tbl[cols_to_save].to_csv(PLAYERS_FILE, index=False)
                            st.success(f"Saved players ({len(players_tbl)} rows) to {PLAYERS_FILE}.")
                        else:
                            st.warning("Couldn't map useful player columns from FBref table; inspect raw table below.")
                            st.dataframe(players_tbl.head(20))
                        # teams mapping
                        if 'team' in teams_tbl.columns:
                            teams_tbl.rename(columns={'team':'team'}, inplace=True)
                        # attempt to extract some team stats and save minimal schema
                        # try to find goals for/against columns
                        tf = teams_tbl.copy()
                        # heuristics: columns like 'GF','GA','Gls' etc.
                        possible_gf = [c for c in tf.columns if str(c).lower() in ['g','gls','gf','goals']]
                        # just save full table as reference
                        tf.to_csv(TEAM_FILE,index=False)
                        st.info(f"Saved raw teams table to {TEAM_FILE} for inspection.")
                        st.experimental_rerun()
                    except Exception as e:
                        st.error(f"Mapping failed: {e}")
                else:
                    st.error("No suitable tables found on FBref landing page.")


elif page == "Fetch UEFA":
    st.header("Fetch UEFA official stats (season-level)")
    if st.button("Fetch UEFA stats now"):
        with st.spinner("Fetching UEFA..."):
            tbl, err = fetch_uefa_team_stats()
            if err:
                st.error(f"UEFA fetch failed: {err}")
            elif tbl is None:
                st.error("No tables found on UEFA page.")
            else:
                tbl.to_csv(TEAM_FILE,index=False)
                st.success(f"Saved UEFA table to {TEAM_FILE}. Please inspect and map columns as needed.")
                st.experimental_rerun()

elif page == "Quick Wikipedia import":
    st.header("Quick import top scorers from Wikipedia")
    st.markdown("This grabs the 'Top goalscorers' table from the Wikipedia article for 2023–24 Champions League and appends to players CSV.")
    if st.button("Import Wikipedia top scorers"):
        url = "https://en.wikipedia.org/wiki/2023%E2%80%9324_UEFA_Champions_League"
        try:
            r = requests.get(url, timeout=10); r.raise_for_status()
            tables = pd.read_html(r.text)
            # heuristics: pick table with 'Player' and 'Goals'
            chosen = None
            for t in tables:
                if 'Player' in t.columns and 'Goals' in t.columns:
                    chosen = t; break
            if chosen is None:
                st.error("Couldn't locate top scorers table automatically.")
            else:
                dfw = chosen.rename(columns={'Player':'player','Club':'team','Goals':'goals'})
                dfw['season'] = 2024
                if 'assists' not in dfw.columns: dfw['assists'] = 0
                dfw['GplusA'] = dfw['goals'] + dfw['assists']
                # append to local players file
                existing = pd.read_csv(PLAYERS_FILE)
                merged = pd.concat([existing, dfw[['season','team','player','goals','assists','GplusA']]], ignore_index=True, sort=False)
                merged.to_csv(PLAYERS_FILE,index=False)
                st.success(f"Appended {len(dfw)} players to {PLAYERS_FILE}.")
                st.experimental_rerun()
        except Exception as e:
            st.error(f"Wikipedia import failed: {e}")

# Footer guidance
st.markdown("""
---
**Notes & limitations**
- FBref fetching is best-effort and depends on FBref's current page structure. If mapping fails, upload your FBref-export CSVs via Upload CSV page.
- Understat does not provide Champions League data publicly in a straightforward API; for xG at competition-level we rely on FBref / UEFA or third-party datasets (Kaggle).
- PDF reports now include simple embedded charts (matplotlib PNGs). If you want full multi-page reports with one page per player, say so and I will add that feature next.
""")

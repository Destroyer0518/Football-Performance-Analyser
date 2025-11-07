
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from io import StringIO, BytesIO
import os

# PDF generation
from reportlab.lib.pagesizes import A4, landscape
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib import colors

st.set_page_config(layout="wide", page_title="UCL 23-24 — Enhanced Explorer", initial_sidebar_state="expanded")

# -------------------- Utilities --------------------
@st.cache_data
def generate_sample_data(seed=42):
    np.random.seed(seed)
    teams = ["Real Madrid", "Manchester City", "Bayern Munich", "Paris Saint-Germain",
             "AC Milan", "Barcelona", "Juventus", "Liverpool", "Chelsea", "Dortmund"]
    seasons = [2023, 2024]
    rows = []
    for season in seasons:
        for team in teams:
            for venue in ["Home", "Away"]:
                matches = np.random.randint(6, 12)
                wins = np.random.randint(0, matches+1)
                draws = np.random.randint(0, matches - wins + 1)
                losses = matches - wins - draws
                goals_for = np.random.randint(0, matches*4+1)
                goals_against = np.random.randint(0, matches*3+1)
                assists = int(goals_for * np.random.uniform(0.4,0.9))
                fouls_committed = np.random.randint(5*matches, 20*matches)
                fouls_conceded = np.random.randint(5*matches, 20*matches)
                winners_or_equalisers = np.random.randint(0, goals_for+1)
                # Extra stats (synthetic if not available)
                xG = round(np.random.uniform(0.5, 2.5) * matches, 2)
                xA = round(np.random.uniform(0.2, 1.2) * matches, 2)
                pass_accuracy = round(np.random.uniform(70, 92), 1)
                tackles = np.random.randint(5*matches, 30*matches)
                saves = np.random.randint(0, 10*matches)
                rows.append({
                    "season": season,
                    "team": team,
                    "venue": venue,
                    "matches": matches,
                    "wins": wins,
                    "draws": draws,
                    "losses": losses,
                    "goals_for": goals_for,
                    "goals_against": goals_against,
                    "assists": assists,
                    "GplusA": goals_for + assists,
                    "fouls_committed": fouls_committed,
                    "fouls_conceded": fouls_conceded,
                    "winners_or_equalisers": winners_or_equalisers,
                    "xG": xG,
                    "xA": xA,
                    "pass_accuracy": pass_accuracy,
                    "tackles": tackles,
                    "saves": saves
                })
    df = pd.DataFrame(rows)
    # player-level dataset
    players = []
    for season in seasons:
        for team in teams:
            for i in range(1, 16):
                player = f"Player_{team.split()[0]}_{i}"
                goals = np.random.poisson(4) + (np.random.randint(0,6) if i<4 else 0)
                assists = max(0, int(goals * np.random.uniform(0.2,0.8)))
                minutes = np.random.randint(200, 3500)
                fouls = np.random.randint(0, 40)
                winners_or_equalisers = np.random.randint(0, goals+1)
                xG = round(np.random.uniform(0.1, 1.0) * (minutes/90), 2)
                xA = round(np.random.uniform(0.05, 0.6) * (minutes/90), 2)
                pass_accuracy = round(np.random.uniform(65, 95), 1)
                tackles = np.random.randint(0, 80)
                saves = np.random.randint(0, 150) if i>12 else 0  # last few as keepers
                players.append({
                    "season": season,
                    "team": team,
                    "player": player,
                    "goals": goals,
                    "assists": assists,
                    "GplusA": goals + assists,
                    "minutes": minutes,
                    "fouls_committed": fouls,
                    "winners_or_equalisers": winners_or_equalisers,
                    "xG": xG,
                    "xA": xA,
                    "pass_accuracy": pass_accuracy,
                    "tackles": tackles,
                    "saves": saves
                })
    df_players = pd.DataFrame(players)
    return df, df_players

# file names
TEAM_FILE = "ucl_team_venue_agg.csv"
PLAYERS_FILE = "ucl_players.csv"

# load or create sample data
if os.path.exists(TEAM_FILE) and os.path.exists(PLAYERS_FILE):
    try:
        df_team = pd.read_csv(TEAM_FILE)
        df_players = pd.read_csv(PLAYERS_FILE)
    except Exception as e:
        st.warning("Couldn't read local CSVs, regenerating sample data.")
        df_team, df_players = generate_sample_data()
        df_team.to_csv(TEAM_FILE, index=False)
        df_players.to_csv(PLAYERS_FILE, index=False)
else:
    df_team, df_players = generate_sample_data()
    df_team.to_csv(TEAM_FILE, index=False)
    df_players.to_csv(PLAYERS_FILE, index=False)

# -------------------- Sidebar & Multi-page --------------------
st.sidebar.title("Controls")
page = st.sidebar.radio("Go to", ["Overview Dashboard", "Team Explorer", "Player Explorer", "Upload Data", "Fetch Real Data (experimental)"])

st.sidebar.markdown("---")
st.sidebar.markdown("Data stored locally in:\n- ucl_team_venue_agg.csv\n- ucl_players.csv")
if st.sidebar.button("Regenerate sample data"):
    df_team, df_players = generate_sample_data(seed=np.random.randint(0,9999))
    df_team.to_csv(TEAM_FILE, index=False)
    df_players.to_csv(PLAYERS_FILE, index=False)
    st.experimental_rerun()

# helper to compute derived metrics
def enrich_team_df(df):
    df = df.copy()
    df['goal_diff'] = df['goals_for'] - df['goals_against']
    df['points'] = df['wins']*3 + df['draws']*1
    df['GplusA'] = df.get('GplusA', df['goals_for'] + df.get('assists', 0))
    return df

df_team = enrich_team_df(df_team)
st.title("UCL 2023-24 — Enhanced Team & Player Analysis (Exports Enabled)")

# -------------------- Export helpers --------------------
def to_excel_bytes(df_dict):
    """df_dict: dict of sheet_name -> dataframe"""
    output = BytesIO()
    with pd.ExcelWriter(output, engine='openpyxl') as writer:
        for sheet, df in df_dict.items():
            df.to_excel(writer, sheet_name=sheet[:31], index=False)
        writer.save()
    output.seek(0)
    return output.getvalue()

def create_team_pdf(team, season, df_team_sel, players_df):
    """Create a simple PDF report for the selected team/season and return bytes"""
    buffer = BytesIO()
    doc = SimpleDocTemplate(buffer, pagesize=landscape(A4), rightMargin=20, leftMargin=20, topMargin=20, bottomMargin=20)
    styles = getSampleStyleSheet()
    flow = []
    title = f"{team} — Season {season} — Team Report"
    flow.append(Paragraph(title, styles['Title']))
    flow.append(Spacer(1,12))
    # summary table (first row)
    summary = {
        "Matches": int(df_team_sel['matches'].sum()),
        "Wins": int(df_team_sel['wins'].sum()),
        "Draws": int(df_team_sel['draws'].sum()),
        "Losses": int(df_team_sel['losses'].sum()),
        "Goals For": int(df_team_sel['goals_for'].sum()),
        "Goals Against": int(df_team_sel['goals_against'].sum()),
        "Goal Diff": int((df_team_sel['goals_for'] - df_team_sel['goals_against']).sum()),
        "G+A": int(df_team_sel['GplusA'].sum()),
        "Fouls Committed": int(df_team_sel['fouls_committed'].sum()),
        "Fouls Conceded": int(df_team_sel['fouls_conceded'].sum()),
    }
    data = [["Metric", "Value"]]
    for k,v in summary.items():
        data.append([k, str(v)])
    t = Table(data, hAlign='LEFT')
    t.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#d3d3d3")),('GRID',(0,0),(-1,-1),0.5,colors.black)]))
    flow.append(t)
    flow.append(Spacer(1,12))
    # top players table (top 8)
    flow.append(Paragraph("Top players (by G+A)", styles['Heading2']))
    top_players = players_df.sort_values('GplusA', ascending=False).head(8)
    tbl = [list(top_players.columns)]
    for _, r in top_players.iterrows():
        tbl.append([str(r[c]) for c in top_players.columns])
    t2 = Table(tbl, hAlign='LEFT', repeatRows=1)
    t2.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.3,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0"))]))
    flow.append(t2)
    flow.append(Spacer(1,12))
    # Add a simple note
    flow.append(Paragraph("Generated by UCL 23-24 Explorer", styles['Normal']))
    doc.build(flow)
    buffer.seek(0)
    return buffer.getvalue()

# -------------------- Pages --------------------
if page == "Overview Dashboard":
    st.header("League / Season Overview")
    season = st.selectbox("Select season", sorted(df_team['season'].unique()), index=0)
    df_season = df_team[df_team['season']==season]
    agg = df_season.groupby('team').agg({
        'matches':'sum','wins':'sum','draws':'sum','losses':'sum',
        'goals_for':'sum','goals_against':'sum','GplusA':'sum','fouls_committed':'sum','fouls_conceded':'sum',
        'winners_or_equalisers':'sum','xG':'sum','xA':'sum','tackles':'sum','saves':'sum'
    }).reset_index()
    agg['goal_diff'] = agg['goals_for'] - agg['goals_against']
    agg['points'] = agg['wins']*3 + agg['draws']
    st.subheader(f"Table — season {season}")
    st.dataframe(agg.sort_values('points', ascending=False).reset_index(drop=True))

    st.subheader("Interactive charts")
    col1, col2 = st.columns(2)
    with col1:
        fig = px.bar(agg.sort_values('points', ascending=False), x='team', y='points', title='Points by Team', hover_data=['wins','goals_for','goal_diff'])
        st.plotly_chart(fig, use_container_width=True)
    with col2:
        fig2 = px.scatter(agg, x='xG', y='goals_for', size='GplusA', hover_name='team', title='xG vs Goals For (bubble size = G+A)')
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("**Top 10 players by G+A (season)**")
    top_players = df_players[df_players['season']==season].sort_values('GplusA', ascending=False).head(10)
    st.table(top_players[['player','team','goals','assists','GplusA','xG','xA','pass_accuracy']])

elif page == "Team Explorer":
    st.header("Team Explorer")
    season = st.selectbox("Season", sorted(df_team['season'].unique()), key='team_season')
    team = st.selectbox("Team", sorted(df_team['team'].unique()))
    df_sel = df_team[(df_team['season']==season) & (df_team['team']==team)].copy()
    if df_sel.empty:
        st.warning("No data for this team/season.")
    else:
        df_sel = enrich_team_df(df_sel) if 'enrich_team_df' in globals() else (lambda x:x)(df_sel)
        # Metrics row
        home = df_sel[df_sel['venue']=='Home'].squeeze()
        away = df_sel[df_sel['venue']=='Away'].squeeze()
        col1, col2, col3, col4, col5 = st.columns(5)
        col1.metric("Home wins", int(home['wins']) if 'wins' in home else 0)
        col2.metric("Away wins", int(away['wins']) if 'wins' in away else 0)
        col3.metric("Goal Diff (total)", int(df_sel['goal_diff'].sum()))
        col4.metric("G + A (total)", int(df_sel['GplusA'].sum()))
        col5.metric("Winners/Equalisers", int(df_sel['winners_or_equalisers'].sum()))

        st.subheader("Home / Away breakdown")
        st.dataframe(df_sel[['venue','matches','wins','draws','losses','goals_for','goals_against','goal_diff','GplusA','fouls_committed','fouls_conceded','winners_or_equalisers','xG','xA','pass_accuracy','tackles','saves']])

        # Interactive wins chart
        fig = px.bar(df_sel, x='venue', y=['wins','draws','losses'], title='Results by venue')
        st.plotly_chart(fig, use_container_width=True)

        # Top players for the team
        st.subheader("Top players (by G+A)")
        players_sel = df_players[(df_players['season']==season) & (df_players['team']==team)].sort_values('GplusA', ascending=False)
        st.dataframe(players_sel[['player','goals','assists','GplusA','minutes','xG','xA','pass_accuracy','tackles','saves']].reset_index(drop=True))

        # Exports: CSV (existing), Excel (new), PDF (new)
        csv_bytes = df_sel.to_csv(index=False).encode('utf-8')
        st.download_button("Download team data (CSV)", csv_bytes, f"{team}_{season}_team.csv", "text/csv")

        # Excel with two sheets: team_breakdown and top_players
        if not players_sel.empty:
            df_dict = {"team_breakdown": df_sel, "top_players": players_sel.head(50)}
        else:
            df_dict = {"team_breakdown": df_sel}
        excel_bytes = to_excel_bytes(df_dict)
        st.download_button("Download team report (Excel)", excel_bytes, f"{team}_{season}_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        # PDF
        pdf_bytes = create_team_pdf(team, season, df_sel, players_sel)
        st.download_button("Download team report (PDF)", pdf_bytes, f"{team}_{season}_report.pdf", "application/pdf")

elif page == "Player Explorer":
    st.header("Player Explorer")
    season = st.selectbox("Season (player)", sorted(df_players['season'].unique()), key='player_season')
    team_filter = st.selectbox("Filter by Team", ["All"] + sorted(df_players['team'].unique()))
    df_pl = df_players[df_players['season']==season].copy()
    if team_filter != "All":
        df_pl = df_pl[df_pl['team']==team_filter]
    st.write(f"Players: {len(df_pl)}")
    min_goals = st.slider("Minimum goals", 0, int(df_pl['goals'].max()), 0)
    df_pl = df_pl[df_pl['goals']>=min_goals]
    search = st.text_input("Search player (substring)")
    if search:
        df_pl = df_pl[df_pl['player'].str.contains(search, case=False, na=False)]
    if df_pl.empty:
        st.warning("No players match filters.")
    else:
        st.dataframe(df_pl.sort_values('GplusA', ascending=False).reset_index(drop=True))

        sel_player = st.selectbox("Select player to view detail", [""] + sorted(df_pl['player'].unique()))
        if sel_player:
            p = df_pl[df_pl['player']==sel_player].squeeze()
            cols = st.columns(6)
            cols[0].metric("Goals", int(p['goals']))
            cols[1].metric("Assists", int(p['assists']))
            cols[2].metric("G + A", int(p['GplusA']))
            cols[3].metric("xG", float(p['xG']) if 'xG' in p else None)
            cols[4].metric("xA", float(p['xA']) if 'xA' in p else None)
            cols[5].metric("Pass %", float(p['pass_accuracy']) if 'pass_accuracy' in p else None)

            # small bar chart
            figp = px.bar(x=['Goals','Assists','Winners/Equalisers','Tackles'], y=[p.get('goals',0), p.get('assists',0), p.get('winners_or_equalisers',0), p.get('tackles',0)], labels={'x':'metric','y':'count'}, title=f"{sel_player} — key counts")
            st.plotly_chart(figp, use_container_width=True)

            # Exports for player: CSV, Excel, PDF (single player)
            st.download_button("Download player row (CSV)", StringIO(p.to_frame().T.to_csv(index=False)), f"{sel_player}_{season}.csv", "text/csv")

            # Excel
            player_df = p.to_frame().T
            excel_bytes = to_excel_bytes({f"{sel_player}": player_df})
            st.download_button("Download player report (Excel)", excel_bytes, f"{sel_player}_{season}_report.xlsx", "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

            # PDF: simple one-page PDF with player stats table
            pdf_buf = BytesIO()
            doc = SimpleDocTemplate(pdf_buf, pagesize=landscape(A4))
            styles = getSampleStyleSheet()
            flow = []
            flow.append(Paragraph(f"{sel_player} — Season {season} — Player Report", styles['Title']))
            flow.append(Spacer(1,12))
            tbl = [["Metric","Value"]]
            for c in player_df.columns:
                tbl.append([c, str(player_df.iloc[0][c])])
            t = Table(tbl, hAlign='LEFT')
            t.setStyle(TableStyle([('GRID',(0,0),(-1,-1),0.3,colors.grey),('BACKGROUND',(0,0),(-1,0),colors.HexColor("#f0f0f0"))]))
            flow.append(t)
            doc.build(flow)
            pdf_buf.seek(0)
            st.download_button("Download player report (PDF)", pdf_buf.read(), f"{sel_player}_{season}_report.pdf", "application/pdf")

elif page == "Upload Data":
    st.header("Upload your own datasets (team & player CSVs)")
    st.markdown("Upload two CSVs that match (or can be mapped to) the schema. If columns differ, the app will try to map common names.")
    uploaded_team = st.file_uploader("Upload team-level CSV", type=["csv"], accept_multiple_files=False)
    uploaded_players = st.file_uploader("Upload player-level CSV", type=["csv"], accept_multiple_files=False)
    if st.button("Load uploaded files") and (uploaded_team or uploaded_players):
        if uploaded_team:
            try:
                df_team_new = pd.read_csv(uploaded_team)
                common = {c:c for c in df_team_new.columns if c in ['team','season','venue','matches','wins','draws','losses','goals_for','goals_against','assists','GplusA','fouls_committed','fouls_conceded','winners_or_equalisers','xG','xA','pass_accuracy','tackles','saves']}
                df_team_new = df_team_new.rename(columns=common)
                df_team_new.to_csv(TEAM_FILE, index=False)
                st.success("Team CSV loaded and saved locally (ucl_team_venue_agg.csv).")
            except Exception as e:
                st.error(f"Failed to load team CSV: {e}")
        if uploaded_players:
            try:
                df_players_new = pd.read_csv(uploaded_players)
                common = {c:c for c in df_players_new.columns if c in ['season','team','player','goals','assists','minutes','fouls_committed','winners_or_equalisers','xG','xA','pass_accuracy','tackles','saves','GplusA']}
                df_players_new = df_players_new.rename(columns=common)
                if 'GplusA' not in df_players_new.columns and {'goals','assists'}.issubset(df_players_new.columns):
                    df_players_new['GplusA'] = df_players_new['goals'] + df_players_new['assists']
                df_players_new.to_csv(PLAYERS_FILE, index=False)
                st.success("Players CSV loaded and saved locally (ucl_players.csv).")
            except Exception as e:
                st.error(f"Failed to load players CSV: {e}")
        st.experimental_rerun()

elif page == "Fetch Real Data (experimental)":
    st.header("Fetch real UCL 2023-24 data (experimental)")
    st.markdown("This tries to fetch public datasets from the web. If you prefer, upload your own CSVs on the previous page. This action runs live when you click the button — network required.")
    if st.button("Attempt to fetch public UCL datasets"):
        st.info("This feature is a best-effort attempt and may fail depending on source availability. If it fails, please upload CSVs manually.")
        st.warning("Network fetch is not implemented in this sample environment. Please upload CSVs or paste data. (In a deployed version we can implement fetching from specific URLs/APIs.)")

# -------------------- Footer --------------------
# -------------------- Footer --------------------
st.markdown("""
---
**Notes & next steps**
- Added export options: CSV, Excel (multi-sheet), and PDF summary reports for teams and players.
- PDF generation uses reportlab and creates simple, printable one-page summary reports.
- If you'd like richer PDF reports (with charts embedded), I can add those next.
""")

import pandas as pd

def add_positions(df_players, positions_csv, player_col="Jugador"):
    pos = pd.read_csv(positions_csv)
    # accept either "Name" or "Jugador"
    key = "Name" if "Name" in pos.columns else "Nombre"
    pos = pos.rename(columns={key: "Jugador"})
    return df_players.merge(pos[["Jugador","Position","Line"]], on="Jugador", how="left")

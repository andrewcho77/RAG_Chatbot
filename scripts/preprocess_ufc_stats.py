import pandas as pd
import uuid


def parse_x_of_y_str(value):
    try:
        x, y = map(int, value.split(" of "))
        return x, y
    except:
        return None, None


def parse_percentage(value):
    try:
        return float(value.strip("%"))
    except:
        return None


def parse_time(value):
    try:
        minutes, seconds = map(int, value.split(":"))
        return minutes, seconds
    except:
        return None, None


def create_uuid(row):
    return str(uuid.uuid4())


def create_custom_numeric_cols(dataframe):
    dataframe[["SIG.STR._LANDED", "SIG.STR._ATTEMPTED"]] = dataframe["SIG.STR."].apply(
        lambda x: pd.Series(parse_x_of_y_str(x))
    )
    dataframe["SIG.STR._PERCENT"] = dataframe["SIG.STR. %"].apply(parse_percentage)
    dataframe[["TOTAL_STR._LANDED", "TOTAL_STR._ATTEMPTED"]] = dataframe[
        "TOTAL STR."
    ].apply(lambda x: pd.Series(parse_x_of_y_str(x)))
    dataframe[["TD_LANDED", "TD_ATTEMPTED"]] = dataframe["TD"].apply(
        lambda x: pd.Series(parse_x_of_y_str(x))
    )
    dataframe["TD_PERCENT"] = dataframe["TD %"].apply(parse_percentage)
    dataframe[["CTRL_MINUTES", "CTRL_SECONDS"]] = dataframe["CTRL"].apply(
        lambda x: pd.Series(parse_time(x))
    )
    dataframe[["HEAD_LANDED", "HEAD_ATTEMPTED"]] = dataframe["HEAD"].apply(
        lambda x: pd.Series(parse_x_of_y_str(x))
    )
    dataframe[["BODY_LANDED", "BODY_ATTEMPTED"]] = dataframe["BODY"].apply(
        lambda x: pd.Series(parse_x_of_y_str(x))
    )
    dataframe[["LEG_LANDED", "LEG_ATTEMPTED"]] = dataframe["LEG"].apply(
        lambda x: pd.Series(parse_x_of_y_str(x))
    )
    dataframe[["DISTANCE_LANDED", "DISTANCE_ATTEMPTED"]] = dataframe["DISTANCE"].apply(
        lambda x: pd.Series(parse_x_of_y_str(x))
    )
    dataframe[["CLINCH_LANDED", "CLINCH_ATTEMPTED"]] = dataframe["CLINCH"].apply(
        lambda x: pd.Series(parse_x_of_y_str(x))
    )
    dataframe[["GROUND_LANDED", "GROUND_ATTEMPTED"]] = dataframe["GROUND"].apply(
        lambda x: pd.Series(parse_x_of_y_str(x))
    )

    dataframe.drop(
        columns=[
            "HEAD",
            "SIG.STR.",
            "SIG.STR. %",
            "TOTAL STR.",
            "TD",
            "TD %",
            "CTRL",
            "HEAD",
            "BODY",
            "LEG",
            "DISTANCE",
            "CLINCH",
            "GROUND",
        ],
        inplace=True,
    )

    return dataframe


def create_stat_summary(row):
    # Core fight details
    fight_summary = [
        f"In {row['EVENT']}, {row['FIGHTER']} fought in {row['BOUT']} during round {row['ROUND']}.",
        f"They landed {row['SIG.STR._LANDED']} significant strikes at an accuracy of {row['SIG.STR._PERCENT']}%.",
        f"Total strikes landed: {row['TOTAL_STR._LANDED']}, attempted: {row['TOTAL_STR._ATTEMPTED']} | Knockdowns: {row['KD']}.",
    ]

    # Grappling stats
    grappling = []
    if row["TD_LANDED"] > 0:
        grappling.append(
            f"Takedowns: {row['TD_LANDED']} of {row['TD_ATTEMPTED']} ({row['TD_PERCENT']}% success rate)."
        )
    if row["SUB.ATT"] > 0:
        grappling.append(f"Submission attempts: {row['SUB.ATT']}.")
    if row["REV."] > 0:
        grappling.append(f"Reversals: {row['REV.']}.")
    if row["CTRL_MINUTES"] > 0 or row["CTRL_SECONDS"] > 0:
        grappling.append(
            f"Control time: {row['CTRL_MINUTES']} min {row['CTRL_SECONDS']} sec."
        )

    if grappling:
        fight_summary.append(" ".join(grappling))

    # Striking breakdown
    striking_summary = [
        f"Head strikes landed: {row['HEAD_LANDED']}, attempted: {row['HEAD_ATTEMPTED']}.",
        f"Body strikes landed: {row['BODY_LANDED']}, attempted: {row['BODY_ATTEMPTED']}.",
        f"Leg strikes landed: {row['LEG_LANDED']}, attempted: {row['LEG_ATTEMPTED']}.",
        f"Distance strikes landed: {row['DISTANCE_LANDED']}, attempted: {row['DISTANCE_ATTEMPTED']}.",
        f"Clinch strikes landed: {row['CLINCH_LANDED']}, attempted: {row['CLINCH_ATTEMPTED']}.",
        f"Ground strikes landed: {row['GROUND_LANDED']}, attempted: {row['GROUND_ATTEMPTED']}.",
    ]

    fight_summary.extend(striking_summary)

    return " ".join(fight_summary)


def preprocess_stats():
    STATS_RAW_DATA_PATH = "./data/ufc_fight_stats_raw.csv"
    STATS_CLEANED_DATA_PATH = "./data/ufc_fight_stats_cleaned.csv"

    ufc_fight_stats = pd.read_csv(
        STATS_RAW_DATA_PATH,
        converters={
            "KD": pd.to_numeric,
            "SUB.ATT": pd.to_numeric,
            "REV.": pd.to_numeric,
        },
    )

    # Add UUID Column here
    ufc_fight_stats["UUID"] = ufc_fight_stats.apply(create_uuid, axis=1)

    uuid_col = ufc_fight_stats.pop("UUID")
    ufc_fight_stats.insert(0, "UUID", uuid_col)

    ufc_fight_stats = create_custom_numeric_cols(ufc_fight_stats)
    ufc_fight_stats["text"] = ufc_fight_stats.apply(create_stat_summary, axis=1)

    ufc_fight_stats.to_csv(STATS_CLEANED_DATA_PATH, index=False)

    return ufc_fight_stats

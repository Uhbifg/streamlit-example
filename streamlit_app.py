from collections import namedtuple
import altair as alt
import math
import pandas as pd
import streamlit as st
from catboost import CatBoostRegressor
import numpy as np

st.title("Предсказание времени простоя поезда!")
X_train = pd.read_parquet("data/train.parquet")
uploaded_file = st.file_uploader("Choose a file (parquet-like)")
if uploaded_file is not None:
    X_test = pd.read_parquet(uploaded_file)



    model = CatBoostRegressor()
    model.load_model("data/model_last")


    @st.experimental_memo
    def convert_df(df):
       return df.to_csv(index=False).encode('utf-8')


    X_test.prev_freight = X_test.prev_freight.fillna("ГАЗЫ УГЛЕВОД ПР")
    X_test.prev_fr_group = X_test.prev_fr_group.fillna("Нефтяные грузы")
    X_test.fr_group = X_test.fr_group.fillna("Лесные грузы")
    X_test.freight = X_test.freight.fillna("БАЛАНСЫ СВ 1,5М")

    X_train.loc[X_train.fr_group.isnull(), "fr_group"] = "Остальные грузы"

    # заполним distance медианой по станциям
    dist = pd.concat([X_train[["snd_st_id", "rsv_st_id", "distance"]], X_test[["snd_st_id", "rsv_st_id", "distance"]]])
    dist = pd.DataFrame(dist.groupby(["snd_st_id", "rsv_st_id"]).distance.median()).rename(columns={"distance": "app_dist"})
    X_test.loc[X_test.distance.isnull(), "distance"] = X_test.loc[X_test.distance.isnull()].join(dist, on=["snd_st_id",
                                                                                                           "rsv_st_id"],
                                                                                                 how="left").app_dist
    X_train.loc[X_train.distance.isnull(), "distance"] = X_train.loc[X_train.distance.isnull()].join(dist, on=["snd_st_id",
                                                                                                               "rsv_st_id"],
                                                                                                     how="left").app_dist

    # уберем Кокс каменноугольный его мало и кажется +-это тоже самое (знатоки не кидайте тапками)
    X_train.fr_group = X_train.fr_group.map({"Кокс каменноугольный": "Уголь каменный"}).fillna(X_train.fr_group)
    X_test.fr_group = X_test.fr_group.map({"Кокс каменноугольный": "Уголь каменный"}).fillna(X_test.fr_group)

    X_train.prev_fr_group = X_train.prev_fr_group.map({"Кокс каменноугольный": "Уголь каменный"}).fillna(
        X_train.prev_fr_group)
    X_test.prev_fr_group = X_test.prev_fr_group.map({"Кокс каменноугольный": "Уголь каменный"}).fillna(X_test.prev_fr_group)


    def replace(text):
        if text is None:
            return None
        if "дист газов конд" == text:
            return "техническое устройство"
        if "семена" in text:
            return "семена"
        if "бумага" in text or "картон" in text or "пергамент" in text:
            return "картон"
        if "свеж" in text or "картофель" in text or "лук" in text or "овощ" in text or "фрукт" in text:
            return "овощи фрукты"
        if "рис" in text or "зерно" in text or "зерна" in text or "нут" in text or "корма " in text or "ячмень" in text or "чечевица" in text or "корма " in text or "просо " in text or "орех" in text:
            return "крупа"
        if "масло" in text or "масла" in text or "петролатум" in text:
            return "масло"
        if "орех" in text:
            return "орех"
        if "изд " in text or "изделия" in text:
            return "изделия"
        if "топливо" in text or "топ" in text or "керосин" in text or "бензин" in text:
            return "топливо"
        if " дерев" in text:
            return "деревянная штука"
        if "мебель" in text or "буфеты" in text or "гарнитур" in text or "кресла" in text or "шкафы" in text or "диваны" in text or "ящики" in text or "столы" in text:
            return "мебель"
        if "табак" in text:
            return "табак"
        if "руда" in text or "руды" in text or "кианит" in text or "барит" in text or "диабаз" in text:
            return "руда"
        if "удоб" in text:
            return "удобрения"
        if "бут" in text or "газ" in text:
            return "газ"
        if "рельсы" in text:
            return "рельсы"
        if "плиты" in text:
            return "плиты"
        if "пилмат" in text or "лесмат" in text:
            return "лесоматериалы"
        if "уголь" in text:
            return "уголь"
        if "нефт" in text:
            return "нефть"
        if "одежда" in text:
            return "одежда"
        if "концентр " in text or "глинопорошок" in text or "аглом тит/марг" in text:
            return "полусухое вещество"
        if "борулин" in text or "паронит" in text or "глинопорошок" in text or "монокорунд" in text or "алигнин" in text:
            return "покрытие"
        return text


    # Приведем id товаров к общему знаменателю:
    total = pd.concat([X_train, X_test])

    ind1 = pd.DataFrame(total[["prev_fr_id", "prev_freight"]].groupby("prev_freight").prev_fr_id.value_counts()).rename(
        columns={"prev_fr_id": "count_1"}).reset_index().rename(
        columns={"prev_fr_id": "id", "prev_freight": "item"}).set_index(["id", "item"])
    ind2 = pd.DataFrame(total[["fr_id", "freight"]].groupby("freight").fr_id.value_counts()).rename(
        columns={"fr_id": "count_2"}).reset_index().rename(columns={"fr_id": "id", "freight": "item"}).set_index(
        ["id", "item"])

    a = ind1.join(ind2, how="outer").reset_index()
    a["count"] = a.count_1.fillna(0) + a.count_2.fillna(0)

    a = a.join(pd.DataFrame(a.groupby("item")["count"].max()).rename(columns={"count": "count_max"}), on="item")
    a = a.loc[a["count"] == a["count_max"]]
    item2id = dict(zip(a.item, a.id))

    X_test.prev_fr_id = X_test.prev_freight.map(item2id)
    X_train.prev_fr_id = X_train.prev_freight.map(item2id)

    X_test.fr_id = X_test.freight.map(item2id)
    X_train.fr_id = X_train.freight.map(item2id)

    # заменим товары на воздух если порожняком
    X_train.loc[X_train.is_load == 0, "fr_id"] = -2
    X_train.loc[X_train.prev_is_load == 0, "prev_fr_id"] = -2

    X_test.loc[X_test.is_load == 0, "fr_id"] = -2
    X_test.loc[X_test.prev_is_load == 0, "prev_fr_id"] = -2

    # удалим название товара
    # X_test.drop(columns=["freight", "prev_freight"], inplace=True, errors="ignore")
    # X_train.drop(columns=["freight", "prev_freight"], inplace=True, errors="ignore")

    X_train.freight = X_train.freight.str.lower().map(lambda x: replace(x))
    X_test.freight = X_test.freight.str.lower().map(lambda x: replace(x))

    X_train.prev_freight = X_train.prev_freight.str.lower().map(lambda x: replace(x))
    X_test.prev_freight = X_test.prev_freight.str.lower().map(lambda x: replace(x))

    X_train.loc[X_train.is_load == 0, "freight"] = "Воздух"
    X_train.loc[X_train.prev_is_load == 0, "prev_freight"] = "Воздух"

    X_test.loc[X_test.is_load == 0, "freight"] = "Воздух"
    X_test.loc[X_test.prev_is_load == 0, "prev_freight"] = "Воздух"

    X_train.loc[X_train.is_load == 0, "fr_group"] = "Воздух"
    X_train.loc[X_train.prev_is_load == 0, "prev_fr_group"] = "Воздух"

    X_test.loc[X_test.is_load == 0, "fr_group"] = "Воздух"
    X_test.loc[X_test.prev_is_load == 0, "prev_fr_group"] = "Воздух"

    a = pd.DataFrame(total.groupby(["snd_st_id"]).fr_group.value_counts(normalize=True))
    a = a.join(a.rename(columns={"fr_group": "fr_share"}).reset_index().groupby("snd_st_id").fr_share.max(), on="snd_st_id",
               how="left").rename(columns={"fr_share": "fr_max_share", "fr_group": "fr_share"}).reset_index()
    st_to_fr_group = a.loc[a.fr_max_share == a.fr_share].drop(columns="fr_max_share").rename(
        columns={"fr_group": "fr_common_group", "fr_share": "fr_common_group_share"}).set_index("snd_st_id")

    a = pd.DataFrame(total.groupby(["snd_st_id"]).prev_fr_group.value_counts(normalize=True))
    a = a.join(a.rename(columns={"prev_fr_group": "fr_share"}).reset_index().groupby("snd_st_id").fr_share.max(),
               on="snd_st_id", how="left").rename(
        columns={"fr_share": "fr_max_share", "prev_fr_group": "fr_share"}).reset_index()
    st_to_prev_fr_group = a.loc[a.fr_max_share == a.fr_share].drop(columns="fr_max_share").rename(
        columns={"prev_fr_group": "prev_fr_common_group", "fr_share": "prev_fr_common_group_share"}).set_index("snd_st_id")

    st_to_unique_group = pd.DataFrame(total.groupby(["snd_st_id"]).fr_group.nunique()).rename(
        columns={"fr_group": "st_number_of_fr_types"})


    def set_send_feat(df):
        df = df.join(st_to_fr_group, on="snd_st_id", how="left").join(st_to_prev_fr_group, on="snd_st_id", how="left").join(
            st_to_unique_group, on="snd_st_id", how="left")

        df["fr_group_is_common"] = (df["fr_group"] == df["fr_common_group"])
        df["prev_fr_group_is_common"] = (df["prev_fr_group"] == df["prev_fr_common_group"])
        return df


    X_train = set_send_feat(X_train)
    X_test = set_send_feat(X_test)

    non_worked_days = [i.date() for i in pd.date_range(start='2022-01-01', end='2022-01-09')] + [i.date() for i in
                                                                                                 pd.date_range(
                                                                                                     start='2022-03-06',
                                                                                                     end='2022-03-08')] + [
                          i.date() for i in pd.date_range(start='2022-04-30', end='2022-05-03')] + [i.date() for i in
                                                                                                    pd.date_range(
                                                                                                        start='2022-05-07',
                                                                                                        end='2022-05-10')] + [
                          i.date() for i in pd.date_range(start='2022-06-11', end='2022-06-13')] + [i.date() for i in
                                                                                                    pd.date_range(
                                                                                                        start='2022-10-04',
                                                                                                        end='2022-10-06')] + [
                          i.date() for i in pd.date_range(start='2023-01-01', end='2023-01-09')]

    X_test.loc[X_test.prev_fr_group == "Черные металлы", "prev_fr_group"] = "Минерально-строит."
    X_test.loc[X_test.fr_group == "Черные металлы", "fr_group"] = "Минерально-строит."

    X_test.common_ch = X_test.common_ch.fillna(9.0)

    X_train.prev_date_depart = pd.to_datetime(X_train.prev_date_depart)
    X_train.prev_date_arrival = pd.to_datetime(X_train.prev_date_arrival)

    X_test.prev_date_depart = pd.to_datetime(X_test.prev_date_depart)
    X_test.prev_date_arrival = pd.to_datetime(X_test.prev_date_arrival)

    # выкидываем наблюдения с приездом в праздничные дни (~6% в трейне, в тесте почти нет)
    X_train = X_train.loc[~X_train.prev_date_arrival.dt.date.isin(non_worked_days)]
    X_train["m"] = X_train.prev_date_arrival.dt.month
    X_train = X_train.loc[X_train.m != 3]
    X_train["m"] = X_train.prev_date_depart.dt.month

    X_train = X_train.loc[~X_train.m.isin([1, 2])]

    X_train.drop(columns=["date_depart", "rod", "m"], inplace=True, errors="ignore")
    X_test.drop(columns=["date_depart", "rod", "m"], inplace=True, errors="ignore")

    new_cat_cols = ["prev_date_depart_10month", "prev_date_arrival_10month", "org_changed", "load_free", "fr_group_changed",
                    "fr_changed", "same_station", "fr_group_is_common", "prev_fr_group_is_common", "fr_common_group",
                    "prev_fr_common_group", "load_hard"]

    new_numeric_cols = ["prev_date_depart_dayofweek", "prev_date_depart_day", "prev_date_arrival_dayofweek",
                        "prev_date_arrival_day", "prev_date_arrival_hour", "prev_hours_in_travel", "prev_speed",
                        "st_number_of_fr_types"]


    def create_new_feats(df):
        df["prev_date_depart_dayofweek"] = df.prev_date_depart.dt.dayofweek
        df["prev_date_depart_10month"] = (df.prev_date_depart.dt.month == 10)
        df["prev_date_depart_day"] = df.prev_date_depart.dt.day

        df["prev_date_arrival_10month"] = (df.prev_date_arrival.dt.month == 10)
        df["prev_date_arrival_dayofweek"] = df.prev_date_arrival.dt.dayofweek
        df["prev_date_arrival_day"] = df.prev_date_arrival.dt.day
        df["prev_date_arrival_hour"] = df.prev_date_arrival.dt.hour

        df["prev_hours_in_travel"] = abs(df.prev_date_arrival - df.prev_date_depart) / np.timedelta64(1, 'h')
        df["prev_speed"] = df["prev_distance"] / df["prev_hours_in_travel"]

        df.loc[df.prev_hours_in_travel < 1, "prev_speed"] = None
        df.loc[(df.prev_speed > 35) | (df.prev_speed < 1), "prev_speed"] = None
        df.loc[df.prev_hours_in_travel > 300, ["prev_hours_in_travel", "prev_speed"]] = None

        df.drop(columns=["prev_date_depart", "prev_date_arrival"], inplace=True, errors="ignore")

        df["org_changed"] = (df["snd_org_id"] == df["rsv_org_id"])
        df["load_free"] = ((df["prev_is_load"] == 0) & (df["is_load"] == 0))
        df["load_hard"] = ((df["prev_is_load"] == 1) & (df["is_load"] == 1))
        df["fr_group_changed"] = (df["prev_fr_group"] == df["fr_group"])
        df["fr_changed"] = (df["prev_fr_id"] == df["fr_id"])
        df["same_station"] = (df["snd_st_id"] == df["rsv_st_id"])

        df.drop(columns=["prev_is_load"], inplace=True, errors="ignore")

        return df


    X_test = create_new_feats(X_test)
    X_train = create_new_feats(X_train)


    def reset_unknowns(X_train, X_test, col):
        good_items = set(X_test[col].unique()) & set(X_train[col].unique())
        print(
            f"{col}, reset: train={(~X_train[col].isin(good_items)).sum() / X_train.shape[0]}%, test={(~X_test[col].isin(good_items)).sum() / X_test.shape[0]}%")
        X_test.loc[~X_test[col].isin(good_items), col] = -1
        X_train.loc[~X_train[col].isin(good_items), col] = -1
        return X_train, X_test


    cols_to_reset_unknowns = ["wagnum", "prev_fr_id", "prev_snd_org_id", "prev_rsv_org_id", "snd_st_id",
                              "rsv_st_id", "fr_id", "snd_org_id", "rsv_org_id", "freight", "prev_freight"]

    for col in cols_to_reset_unknowns:
        X_train, X_test = reset_unknowns(X_train, X_test, col)

    category_columns = ["prev_fr_id", "prev_snd_org_id", "prev_rsv_org_id",
                        "snd_st_id", "rsv_st_id", "fr_id", "is_load", "common_ch", "vidsobst",
                        "snd_org_id", "rsv_org_id", "prev_fr_group", "wagnum", "fr_group", "freight",
                        "prev_freight"] + new_cat_cols

    for col in category_columns:
        if len(set(X_test[col].unique()) - set(X_train[col].unique())) == 0 and len(
                set(X_train[col].unique()) - set(X_test[col].unique())) == 0:
            continue
        print(
            f"{col}, Есть в тесте, нет в трейне: {len(set(X_test[col].unique()) - set(X_train[col].unique()))}, В трейне, нет в тесте: {len(set(X_train[col].unique()) - set(X_test[col].unique()))}, total_test={X_test[col].nunique()}")

    X_test[category_columns] = X_test[category_columns].fillna("").astype(str)
    test_res = np.exp(model.predict(X_test))
    answer = pd.DataFrame()
    answer["target"] = test_res

    csv = convert_df(answer)

    st.download_button(
       "Press to Download",
       csv,
       "file.csv",
       "text/csv",
       key='download-csv'
    )

from typing import List

from influxdb import InfluxDBClient
import pandas as pd


def load_amount(df: pd.DataFrame, client: InfluxDBClient):
    for _, row in df.iterrows():
        json_body = [{
            "measurement": "sales",
            "tags": {
                "code": int(row["code"]),
                "class": int(row["class"]),
                "sort": int(row["sort"]),
                "mid_sort": int(row["mid_sort"]),
                "small_sort": int(row["smal_sort"])
            },
            "time": row["busdate"],
            "fields": {
                "amount": float(row["amount"]) if row["amount"] > 0 else 0.0,
                "price_statu": int(row["price_statu"])
            }
        }]

        client.write_points(json_body)


def load_calendar_indicators(df: pd.DataFrame, indicator_list: List[str], client: InfluxDBClient):
    df = df[df["code"] == 2900012024354]
    for _, row in df.iterrows():
        json_body = [{
            "measurement": "calendar_indicators",
            "time": row["busdate"],
            "fields": {k: int(row[k]) for k in indicator_list}
        }]

        client.write_points(json_body)


def main():
    raw = pd.read_csv("data/UnionWestTestData.csv")
    raw = raw.fillna(0.0)
    client = InfluxDBClient(database="xilian", username="admin", password="2much4ME")
    load_amount(raw, client)

    calendar_indicators = [
        'cj_pre3', 'cj_pre2', 'cj_pre1', 'cj_mid', 'cj_aft', 'yd_pre', 'yd_aft',
        'ld_pre', 'ld_aft', 'dw_pre', 'dw_mid', 'dw_aft', 'et_pre', 'et_aft',
        'qx_pre', 'qx_mid', 'qx_aft', 'zq_pre', 'zq_mid', 'zq_aft', 'gq_pre',
        'gq_aft', 'ssy_pre', 'ssy_aft', 'sse_pre', 'sse_aft', 'sd_pre',
        'sd_aft', 'month', 'day', 'week', 'weekday'
    ]

    # load_calendar_indicators(raw, indicator_list=calendar_indicators, client=client)


if __name__ == '__main__':
    main()

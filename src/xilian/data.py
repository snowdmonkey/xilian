from datetime import date
from typing import List, Dict

from influxdb import InfluxDBClient

client = InfluxDBClient(database="xilian", username="admin", password="2much4ME")


def get_target_series(start_date: date, end_date: date, code: str):
    results = client.query("select amount from sales where code=$code and time>=$start_date and time<$end_date",
                           bind_params={"code": str(code), "start_date": str(start_date), "end_date": str(end_date)})

    return [x["amount"] for x in results.get_points()]


def get_promotion_series(start_date: date, end_date: date, code: str):
    results = client.query("select price_statu from sales where code=$code and time>=$start_date and time<$end_date",
                           bind_params={"code": str(code), "start_date": str(start_date), "end_date": str(end_date)})

    return [x["price_statu"] for x in results.get_points()]


def get_calendar_indicators(start_date: date, end_date: date, indicator_names: List[str]) -> List[List[float]]:
    sql = f"select {','.join(indicator_names)} from calendar_indicators where time>=$start_date and time<$end_date"
    results = client.query(sql, bind_params={"start_date": str(start_date), "end_date": str(end_date)})
    d = {k: [] for k in indicator_names}
    for point in results.get_points():
        for k, v in point.items():
            if k != "time":
                d[k].append(v)
    return [d[k] for k in indicator_names]


def get_calendar_indicator(start_date: date, end_date: date, indicator_name: str) -> List[float]:
    sql = f"select {indicator_name} from calendar_indicators where time>=$start_date and time<$end_date"
    results = client.query(sql, bind_params={"start_date": str(start_date), "end_date": str(end_date)})
    return [point[indicator_name] for point in results.get_points()]


def get_dynamic_feat(start_date: date, end_date: date, code: str, feat_name: str) -> List[float]:

    if feat_name == 'price_statu':
        return get_promotion_series(start_date, end_date, code)
    else:
        return get_calendar_indicator(start_date, end_date, feat_name)


def get_good_category(code: str, category: str) -> str:
    sql = f"select amount,{category} from sales where code=$code limit 1"
    result = client.query(sql, bind_params={"code": code})
    point = next(result.get_points())
    return point[category]


def get_category_values(cat: str) -> List[str]:
    sql = f"SHOW TAG VALUES FROM sales WITH KEY = {cat}"
    results = client.query(sql)
    return [point["value"] for point in results.get_points()]


def write_forecast_result(code: str, date_: date, data: Dict[str, float]):
    json_body = [{
        "measurement": "sales",
        "tags": {
            "code": code,
        },
        "time": str(date_),
        "fields": data
    }]
    client.write_points(json_body)

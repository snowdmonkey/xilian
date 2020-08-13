import logging
from datetime import date, timedelta
from pathlib import Path

from xilian.predict import DeepARPredictor
from xilian import data

PREDICT_START = date(2019, 7, 1)
PREDICT_END = date(2020, 7, 1)
# TRAIN_META = "data/meta/meta.json"

logger = logging.getLogger(__name__)


codes = data.get_category_values("code")
codes.sort()


def forecast(code: str, predictor: DeepARPredictor):
    for i in range(0, (PREDICT_END-PREDICT_START).days, predictor.predict_length):
        predict_start = PREDICT_START + timedelta(days=i)
        logger.info(f"code {code} starts from {predict_start}")
        results = predictor.predict(code=code, predict_start=predict_start)
        for j in range(predictor.predict_length):
            data_ = {"deepar_mean": results["mean"][j]}
            for k, v in results["quantiles"].items():
                data_[f"deepar_quantile_{k}"] = v[j]
            data.write_forecast_result(
                code=code, date_=predict_start+timedelta(days=j),
                data=data_
            )


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("meta_path")

    args = parser.parse_args()

    predictor = DeepARPredictor(train_meta_path=Path(args.meta_path), quantiles=[0.95])

    for code in codes:
        forecast(code, predictor)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(process)d - %(name)s - %(levelname)s - %(message)s")
    main()
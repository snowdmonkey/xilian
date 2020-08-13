import json
import logging
from dataclasses import dataclass
from datetime import datetime, date, timedelta
from pathlib import Path
from typing import Union, List, ByteString, Tuple, Dict

import sagemaker as sm
from sagemaker.serializers import JSONSerializer

from . import data


logger = logging.getLogger(__name__)


@dataclass
class Instance:
    start: str
    target: List[float]
    cat: List[int] = None
    dynamic_feat: List[List[float]] = None

    def to_dict(self) -> Dict:
        d = {"start": self.start, "target": self.target}
        if self.cat is not None:
            d.update({"cat": self.cat})
        if self.dynamic_feat is not None:
            d.update({"dynamic_feat": self.dynamic_feat})
        return d


@dataclass
class Configuration:
    output_types: List[str] = ("mean", "quantiles")
    quantiles: List[str] = ("0.1", "0.9")

    def to_dict(self) -> Dict:
        return {k: v for k, v in self.__dict__.items() if v is not None}


@dataclass
class ForecastRequest:
    instance: Instance
    configuration: Configuration

    def to_dict(self):
        return {
            "instances": [self.instance.to_dict()],
            "configuration": self.configuration.to_dict()
        }


class DeepARPredictor:

    def __init__(self, train_meta_path: Path, quantiles: List[float] = None):
        with train_meta_path.open("r") as f:
            self._train_meta = json.load(f)

        self._sm_predictor = sm.predictor.Predictor(
            endpoint_name=self._train_meta["endpoint_name"],
            serializer=JSONSerializer(),
            sagemaker_session=sm.Session()
        )
        self._earliest_date = date.fromisoformat(self._train_meta["train_config"]["train_start"])
        self._predict_quantiles = quantiles
        self._predict_length = int(self._train_meta["train_config"]["hyper_parameters"]["prediction_length"])
        self._cat_mapping = self._train_meta["cat_mapping"]
        self._cat_keys = self._train_meta["train_config"]["cat"]
        self._feat_names = self._train_meta["train_config"]["dynamic_feat"]
        self._config = Configuration(quantiles=[str(x) for x in quantiles])

    @property
    def predict_length(self):
        return self._predict_length

    def _get_instance(self, code: str, predict_start: date) -> Instance:
        start = max(self._earliest_date, predict_start-timedelta(days=800))
        target = data.get_target_series(start_date=start, end_date=predict_start, code=code)
        cat = [
            self._cat_mapping[category_key][data.get_good_category(code, category_key)]
            for category_key in self._cat_keys
        ]
        dynamic_feat = [
            data.get_dynamic_feat(
                start_date=start,
                end_date=predict_start+timedelta(days=self._predict_length),
                code=code,
                feat_name=feat_name
            ) for feat_name in self._feat_names
        ]
        return Instance(start=str(start), target=target, cat=cat, dynamic_feat=dynamic_feat)

    def predict(self, code: str, predict_start: date) -> Dict:
        req = ForecastRequest(instance=self._get_instance(code, predict_start), configuration=self._config)

        res = self._sm_predictor.predict(req.to_dict()).decode()
        return json.loads(res)["predictions"][0]

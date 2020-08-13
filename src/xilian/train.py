import json
import logging
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import List, Dict, Iterable

import boto3
from botocore.exceptions import ClientError
import sagemaker as sm
from sagemaker import image_uris

from . import data
from .utils import upload_to_s3

S3_BUCKET = "mlsl-data"
S3_TRAIN_DATA_PATH = "xilian/data/processed/train.json"
S3_TEST_DATA_PATH = "xilian/data/processed/test.json"
S3_CAT_MAPPING_PATH = "xilian/data/processed/mapping.json"
S3_MODEL_OUTPUT_PATH = "xilian/model/"

LOCAL_TRAIN_DATA_PATH = Path("data/processed/train.json")
LOCAL_TEST_DATA_PATH = Path("data/processed/test.json")
LOCAL_CAT_MAPPING_PATH = Path("data/processed/mapping.json")

SM_ROLE = "arn:aws:iam::799473255825:role/service-role/AmazonSageMaker-ExecutionRole-20191231T212617"


logger = logging.getLogger(__name__)


@dataclass
class DeepARHyperParameters:
    context_length: int
    epochs: int
    prediction_length: int
    time_freq: str = "1D"
    dropout_rate: float = 0.1
    embedding_dimension: int = 64
    learning_rate: float = 1e-3
    mini_batch_size: int = 128
    num_cells: int = 64
    num_layers: int = 2

    def to_dict(self):
        return {k: str(v) for k, v in self.__dict__.items() if v is not None}


@dataclass
class DeepARTrainConfig:
    cat: List[str]
    dynamic_feat: List[str]
    train_start: date
    train_end: date
    hyper_parameters: DeepARHyperParameters

    def to_dict(self):
        return {
            "cat": self.cat,
            "dynamic_feat": self.dynamic_feat,
            "train_start": str(self.train_start),
            "train_end": str(self.train_end),
            "hyper_parameters": self.hyper_parameters.to_dict()
        }


class DeepARTrainer:

    def __init__(self, config: DeepARTrainConfig, name: str = None):

        self._config = config
        self._cat_mapping: Dict[str, Dict[str, int]] = {}
        self._model_path: str = ""
        self._endpoint: str = ""
        self._name = name
        self._sm_session = sm.Session()

    @property
    def name(self):
        return self._name

    @property
    def meta(self) -> Dict:
        return {
            "cat_mapping": self._cat_mapping,
            "train_config": self._config.to_dict(),
            "model_path": self._model_path,
            "endpoint_name": self._endpoint
        }

    def _get_cat_mapping(self):
        m = dict()
        config = self._config
        for cat_key in config.cat:
            cats = data.get_category_values(cat_key)
            cats.sort()
            d = {cat_value: i for i, cat_value in enumerate(cats)}
            m.update({cat_key: d})
        self._cat_mapping = m

    def _generate_json_lines(self) -> Iterable[str]:
        codes = data.get_category_values("code")
        for code in codes:
            d = self._generate_json_line_for_good(code)
            yield json.dumps(d)

    def _generate_json_line_for_good(self, code: str) -> Dict:
        logger.info(f"start to generate json line for code {code}")
        config = self._config
        target = data.get_target_series(start_date=config.train_start, end_date=config.train_end, code=code)
        dynamic_feat = [
            data.get_dynamic_feat(config.train_start, config.train_end, code, x)
            for x in config.dynamic_feat
        ]
        cat = [self._cat_mapping[x][data.get_good_category(code, x)] for x in config.cat]
        return {
            "start": str(config.train_start),
            "target": target,
            "cat": cat,
            "dynamic_feat": dynamic_feat
        }

    @staticmethod
    def _save_json_lines_to_local(path: Path, lines: Iterable[str]):
        if path.exists():
            logger.warning(f"{path} already exists, deleting it first")
            path.unlink()
        with path.open("a") as f:
            for line in lines:
                f.write(line)
                f.write("\n")

    def _fit_deepar(self):
        config = self._config
        sm_session = self._sm_session
        image_name = image_uris.retrieve("forecasting-deepar", boto3.Session().region_name)
        estimator = sm.estimator.Estimator(
            sagemaker_session=sm_session,
            image_uri=image_name,
            role=SM_ROLE,
            instance_count=1,
            instance_type="ml.c4.4xlarge",
            base_job_name="xilian-deepar",
            output_path=f"s3://{S3_BUCKET}/{S3_MODEL_OUTPUT_PATH}",
        )
        estimator.set_hyperparameters(**config.hyper_parameters.to_dict())
        estimator.fit(inputs={
            "train": f"s3://{S3_BUCKET}/{S3_TRAIN_DATA_PATH}"
        })
        self._model_path = estimator.model_data

    def dump_meta(self, path: Path):
        with path.open("w") as f:
            json.dump(self.meta, f)

    def deploy(self):
        endpoint_name = self._name+"-"+datetime.now().strftime("%Y%m%d-%H%M%S")
        logger.info(f"start to deploy model {self._model_path}")
        model = sm.model.Model(
            image_uri=image_uris.retrieve("forecasting-deepar", boto3.Session().region_name),
            model_data=self._model_path,
            role=SM_ROLE,
            name=endpoint_name
        )
        model.deploy(initial_instance_count=1, instance_type="ml.c4.xlarge", endpoint_name=endpoint_name)
        self._endpoint = model.endpoint_name
        logger.info(f"model deployed at {self._endpoint}")

    def train(self):
        logger.info("start to generate cat mapping")
        self._get_cat_mapping()
        logger.info("start to generate train json lines")
        self._save_json_lines_to_local(LOCAL_TRAIN_DATA_PATH, self._generate_json_lines())
        logger.info("start to upload train json lines")
        upload_to_s3(LOCAL_TRAIN_DATA_PATH, S3_BUCKET, S3_TRAIN_DATA_PATH)
        logger.info("start to train model")
        self._fit_deepar()
        logger.info(f"model saved to {self._model_path}")
        logger.info("complete")


def main():
    config = DeepARTrainConfig(
        cat=["small_sort"],
        dynamic_feat=[
            'cj_pre3', 'cj_pre2', 'cj_pre1', 'cj_mid', 'cj_aft', 'yd_pre', 'yd_aft', 'ld_pre', 'ld_aft', 'dw_pre',
            'dw_mid', 'dw_aft', 'et_pre', 'et_aft', 'qx_pre', 'qx_mid', 'qx_aft', 'zq_pre', 'zq_mid', 'zq_aft',
            'gq_pre', 'gq_aft', 'ssy_pre', 'ssy_aft', 'sse_pre', 'sse_aft', 'sd_pre', 'sd_aft', 'price_statu'
        ],
        train_start=date(2016, 7, 1),
        train_end=date(2019, 7, 1),
        hyper_parameters=DeepARHyperParameters(
            context_length=20,
            epochs=1_000,
            prediction_length=3
        )
    )

    trainer = DeepARTrainer(config, name="xilian-deepar")
    trainer.train()
    trainer.deploy()
    meta_output_path = Path(f"data/meta/meta_{trainer.name}_{datetime.now().strftime('%Y%m%d-%H%M%S')}.json")
    meta_output_path.parent.mkdir(exist_ok=True, parents=True)
    trainer.dump_meta(meta_output_path)


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    main()

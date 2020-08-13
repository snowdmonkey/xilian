from pathlib import Path

import boto3


def upload_to_s3(local_file: Path, s3_bucket_name: str, s3_path: str):
    s3 = boto3.resource("s3")
    bucket = s3.Bucket(s3_bucket_name)
    with local_file.open("rb") as f:
        bucket.put_object(Key=s3_path, Body=f)

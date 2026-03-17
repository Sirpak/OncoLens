import sagemaker
from sagemaker.inputs import TrainingInput
from sagemaker.pytorch import PyTorch

ROLE_ARN = "PUT_YOUR_SAGEMAKER_ROLE_ARN_HERE"
BUCKET = "oncolens-david-2026"
DATA_PREFIX = "oncolens/data"
OUTPUT_PREFIX = "oncolens/output"


def main() -> None:
    session = sagemaker.Session(default_bucket=BUCKET)

    estimator = PyTorch(
        entry_point="src/train.py",
        source_dir=".",
        role=ROLE_ARN,
        framework_version="2.1",
        py_version="py310",
        instance_count=1,
        instance_type="ml.g4dn.xlarge",
        output_path=f"s3://{BUCKET}/{OUTPUT_PREFIX}",
        base_job_name="oncolens-train",
        sagemaker_session=session,
    )

    estimator.fit(
        {
            "training": TrainingInput(
                s3_data=f"s3://{BUCKET}/{DATA_PREFIX}",
                input_mode="File",
            )
        }
    )


if __name__ == "__main__":
    main()
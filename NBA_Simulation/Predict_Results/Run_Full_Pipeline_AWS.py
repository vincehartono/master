import os
import sys

# Ensure parent (NBA_Simulation) is importable when run from subfolder or Lambda
PARENT_DIR = os.path.dirname(os.path.dirname(__file__))
if PARENT_DIR not in sys.path:
    sys.path.insert(0, PARENT_DIR)

try:
    # When used as a package (e.g., Lambda handler: NBA_Simulation.Predict_Results.Run_Full_Pipeline_AWS.lambda_handler)
    from Predict_Results.Run_Full_Pipeline import main as run_full_pipeline_main
except Exception:
    # When run as a plain script from this folder
    from Run_Full_Pipeline import main as run_full_pipeline_main


def lambda_handler(event, context):
    """
    AWS Lambda entrypoint.

    Invokes the existing Run_Full_Pipeline.main() and returns a simple status payload.
    """
    run_full_pipeline_main()
    return {"statusCode": 200, "body": "NBA full pipeline run complete"}


if __name__ == "__main__":
    # Local execution entrypoint (useful for testing the AWS wrapper)
    run_full_pipeline_main()


from roboflow import Roboflow
from roboflow.models.inference import InferenceModel
import json
import os
from pathlib import Path

OUT_FILE = "./results.json"


def init_model():
    rf = Roboflow(api_key="enter-your-key")
    project = rf.workspace().project("tennis-annotations")
    model = project.version(3).model

    return model


def run_inference_on_video(model: InferenceModel, filename, output_file) -> bool:
    res = False
    job_id, _signed_url, _expire_time = model.predict_video(
        filename,
        fps=5,
        prediction_type="batch-video",
    )

    results = model.poll_until_video_results(job_id)

    # Write results to file:
    with open(output_file, "w") as f:
        json.dump(results, f)
        res = True

    return res


def run_inference_on_reference_image(model: InferenceModel):
    return model.predict("depth-reference.png", confidence=40, overlap=30).json()


def inference(filename):
    model = init_model()
    filename_without_extension = Path(filename).stem

    return run_inference_on_video(
        model, f"./uploads/{filename}", f"./outputs/{filename_without_extension}.json"
    )


if __name__ == "__main__":
    # for testing
    model = init_model()

    # For now, we cache.
    if not os.path.exists(OUT_FILE):
        run_inference_on_video(model, "demo-rally.mp4", OUT_FILE)
        print("Finished inference on video")
    else:
        print("Skipping inference on video")

    print("Inferencing on reference image...")
    print(run_inference_on_reference_image(model))

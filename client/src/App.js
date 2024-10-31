import "./App.css";
import { useDropzone } from "react-dropzone";
import Uppy from "@uppy/core";
import XHRUpload from "@uppy/xhr-upload";
import "@uppy/core/dist/style.css";
import { useState, useMemo, useEffect } from "react";
import { TailSpin } from "react-loader-spinner";
import axios from "axios";

function App() {
  const [file, setFile] = useState(null);
  const [runningInference, setRunningInference] = useState(false);
  const [rawHeatmapImageData, setRawHeatmapImageData] = useState("");
  const [hasSuccessfulInferenceRun, setHasSuccessfulInferenceRun] =
    useState(false);

  const uppy = useMemo(
    () =>
      new Uppy({
        restrictions: {
          maxFileSize: 1024 * 1024 * 500, // 500 MB
        },
        autoProceed: true,
      }),
    []
  );

  useEffect(() => {
    uppy.use(XHRUpload, {
      endpoint: "http://127.0.0.1:5000/upload",
      fieldName: "file",
      formData: true,
    });

    uppy.on("upload-success", (file) => {
      setFile(file.name);
      console.log(file.data);
    });

    return () => uppy.destroy();
  }, [uppy]);

  const onDrop = (acceptedFiles) => {
    acceptedFiles.forEach((file) => {
      uppy.addFile({
        name: file.name,
        type: file.type,
        data: file,
      });
    });
  };

  const { getRootProps, getInputProps } = useDropzone({ onDrop });

  const submitInferenceRequest = () => {
    setRunningInference(true);
    axios
      .post("http://127.0.0.1:5000/run-inference", {
        fileName: file,
      })
      .then((response) => {
        setRunningInference(false);
        setHasSuccessfulInferenceRun(true);
      })
      .catch((error) => {
        setRunningInference(false);
        console.log(error);
      });
  };

  const fetchHeatmap = () => {
    axios
      .get(`http://127.0.0.1:5000/heatmap/${file}`, {
        responseType: "blob",
      })
      .then((response) => {
        const url = URL.createObjectURL(response.data);
        setRawHeatmapImageData(url);
      })
      .catch((error) => {
        console.log(error);
      });
  };

  const clearData = () => {
    uppy.removeFile(file);
    setFile(null);
    setHasSuccessfulInferenceRun(false);
    setRawHeatmapImageData("");
    setRunningInference(false);
  };

  return (
    <div>
      <h3>Upload your video</h3>
      <div {...getRootProps()} className="dropzone input">
        <input {...getInputProps()} />
        <p>Drag & drop files here, or click to select files</p>
      </div>
      {file != null && (
        <div className="file-upload-container">
          <p>{file}</p>
          <div className="button-container">
            {runningInference && (
              <TailSpin
                visible={true}
                height="30"
                width="30"
                color="white"
                ariaLabel="tail-spin-loading"
                radius="1"
                wrapperStyle={{}}
                wrapperClass=""
              />
            )}
            <button
              disabled={runningInference}
              onClick={submitInferenceRequest}
              className="custom-button"
            >
              Run inference
            </button>
          </div>
        </div>
      )}
      {hasSuccessfulInferenceRun && (
        <div>
          <button
            disabled={runningInference}
            className="custom-button"
            onClick={fetchHeatmap}
          >
            Get heatmap
          </button>
          {rawHeatmapImageData != null && rawHeatmapImageData !== "" && (
            <img src={rawHeatmapImageData} className="heatmap-image" />
          )}
        </div>
      )}
      <button className="clear-button" onClick={clearData}>
        Clear Data
      </button>
    </div>
  );
}

export default App;

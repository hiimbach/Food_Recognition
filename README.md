# Food_Recognition
A part of the Capstone project in Bootcamp Techainer 2022. This model is a transfer learning YOLOv7 model.

## Preparation
To run it, you need to download the weight (and a sample image incase you need it) by:
``` 
pip install 'dvc[gdrive]' && dvc pull
```
In case the Drive requires you to have permission to access, please contact me.

## How to run
- To test the model, run:
``` 
python test.py
```

- To run it on MLChain server, run:
``` 
mlchain run -c mlconfig.yaml 
```
An API server will be hosted at http://0.0.0.0:8001


### Docker

Run with cpu
```
DOCKER_BUILDKIT=1 docker build -t food_detection_cpu -f cpu.Dockerfile .
```

```
docker run --rm -it -p 8001:8001 food_detection_cpu
```

Run with gpu
```
DOCKER_BUILDKIT=1 docker build -t food_detection_gpu -f gpu.Dockerfile .
```

```
docker run --gpus=all --rm -it -p 8001:8001 food_detection_gpu
```

If you want test with streamlit, let's create a new terminal, and run:
```
streamlit run streamlit/streamlit.py
```
Hosted at http://0.0.0.0:8001

---
If you have any question or encouter any problem regarding this repo. Please open an issue and cc me. Thank you.


FROM tensorflow/tfx:1.11.0
WORKDIR /pipeline

COPY requirements.txt .

RUN pip install -U -r requirements.txt

COPY . .

ENV PYTHONPATH="/pipeline:${PYTHONPATH}"

ENTRYPOINT ["python", "tfx/scripts/run_executor.py"]

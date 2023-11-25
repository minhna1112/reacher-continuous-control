FROM pytorch/pytorch:latest
# RUN apt-get update && apt-get install -y libopenmpi-dev

WORKDIR /workspace
COPY requirements.txt ./requirements.txt
COPY setup.py ./setup.py

RUN python3 -m pip install .

CMD ["jupyter", "notebook", "Continuous_Control_solution.ipynb"]
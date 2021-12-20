FROM python:3.8-slim

COPY --chown=root:root src/app /root/app/
COPY --chown=root:root src/ensembles /root/app/ensembles

WORKDIR /root/app

RUN pip install --no-cache-dir -r requirements.txt
RUN chmod +x run.py

ENV SECRET_KEY stuntmanmike

CMD ["python", "run.py"]

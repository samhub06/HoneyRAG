from prefect import flow, task
from prefect.logging import get_run_logger

@task
def ingest_logs():
    logger = get_run_logger()
    logger.info("Stage 1: ingesting Cowrie logs")
    return {"raw": "sample log"}

@task
def preprocess(raw):
    logger = get_run_logger()
    logger.info("Stage 2: preprocessing")
    return {"processed": raw["raw"] + "_processed"}

@task
def analyze(processed):
    logger = get_run_logger()
    logger.info("Stage 3: LLM analysis")
    return {"analysis": processed["processed"] + "_analysis"}

@task
def output(analysis):
    logger = get_run_logger()
    logger.info("Stage 4: output generation")
    return {"result": analysis["analysis"] + "_output"}

@flow(name="HoneyRAG demo flow")
def honey_flow():
    raw = ingest_logs()
    processed = preprocess(raw)
    analysis = analyze(processed)
    return output(analysis)

if __name__ == "__main__":
    print(honey_flow())
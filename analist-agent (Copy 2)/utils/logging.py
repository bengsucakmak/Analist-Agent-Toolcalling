import logging, os, sys
from rich.logging import RichHandler

def setup_logging(log_dir: str="logs", level=logging.INFO):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, "run.log")
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
        datefmt="[%H:%M:%S]",
        handlers=[RichHandler(rich_tracebacks=True), logging.FileHandler(log_path, encoding="utf-8")]
    )
    return logging.getLogger("analist_agent")
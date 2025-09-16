import logging
from utils.types import AgentState

log = logging.getLogger("guardian")

def run(state: AgentState, pii_columns=None) -> AgentState:
    # Şimdilik pasif; ihtiyaç olursa name/surname maskelenebilir.
    # ör: if pii_columns: satırlarda bu kolonları *** ile maskele
    log.info("Guardian check (pasif).")
    return state

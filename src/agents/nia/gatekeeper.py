import re
from src.core.logger import setup_logger

logger = setup_logger("NIA.Gatekeeper")

class RoutingGatekeeper:
    """
    Quality Control Layer.
    Ensures commands are valid and strips conversational filler.
    """
    def __init__(self):
        # Regex to catch "ROUTE:TARGET:COMMAND"
        self.route_pattern = r"(?i)route\s*[:\->]\s*(TARA|IRIS|DOCKER)"

    def validate(self, llm_response: str) -> dict:
        match = re.search(self.route_pattern, llm_response)
        if not match:
            return {"valid": True, "target": None, "command": llm_response, "error": None}

        target = match.group(1).upper()
        raw_command = llm_response[match.end():].strip()

        if raw_command.startswith(":"):
            raw_command = raw_command[1:].strip()

        # Chatter Check: Commands shouldn't be long sentences
        if len(raw_command.split()) > 15:
            return {"valid": False, "error": "COMMAND TOO LONG. Send ONLY the command."}

        # Filler Check
        garbage_starts = ("i will", "sure", "okay", "here is", "closing", "opening")
        if raw_command.lower().startswith(garbage_starts):
            return {"valid": False, "error": f"DO NOT CHAT. Remove filler words like '{raw_command.split()[0]}'."}

        return {"valid": True, "target": target, "command": raw_command, "error": None}
